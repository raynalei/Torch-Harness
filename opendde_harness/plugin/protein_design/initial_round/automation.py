"""Detached deterministic initial rounds: the worker, not Qwen, polls Slurm."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path

import yaml

from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig
from opendde_harness.plugin.protein_design.initial_round.policy import choose_scorer
from opendde_harness.plugin.protein_design.initial_round.storage import (
    SharedFiles,
    config_difference_paths,
    status_profile_identity,
    write_json,
)
from opendde_harness.utils.portable_lock import file_lock


def prepare_run(
    config_path: Path | dict,
    *,
    pdb_path: Path | None = None,
    name: str | None = None,
    registrations: dict | None = None,
    plugin_config: dict | None = None,
):
    value = copy.deepcopy(config_path) if isinstance(config_path, dict) else yaml.safe_load(config_path.read_text())
    if not isinstance(value, dict):
        raise ValueError("workflow configuration must be an object")
    if plugin_config is None:
        if registrations is None:
            from opendde_harness.config.loader import load_config

            plugin_config = load_config().plugins.config.get("protein-design", {})
        else:
            plugin_config = {}
    from .defaults import deployment_settings, load_deployment_defaults, workflow_values

    plugin_config = load_deployment_defaults(plugin_config)
    plugin_config = deployment_settings(plugin_config)
    # Deployment fallback must not turn an explicit legacy single round into autonomous.
    if "autonomous" not in value:
        value["autonomous"] = None
    explicit_evaluation = value.get("evaluation", {})
    if value.get("autonomous") is None and "scoring_backend" not in explicit_evaluation:
        configured = {kind for kind in ("af3score", "torchscore") if explicit_evaluation.get(kind) is not None}
        if configured:
            explicit_evaluation["scoring_backend"] = choose_scorer(None, configured, set())

    value = workflow_values(value, plugin_config)
    aliases = value.pop("model_aliases", {})
    evaluation = value.setdefault("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")
    known = registrations if registrations is not None else plugin_config.get("model_services", {})

    def registration(kind, explicit=None):
        if kind in aliases:
            match = known.get(aliases[kind])
            if not match or match.get("kind") != kind:
                raise ValueError(f"invalid registered model alias for {kind}")
            return match
        matches = [item for item in known.values() if item.get("kind") == kind]
        if explicit is not None and len(matches) > 1:
            matches = [item for item in matches if item["service"].get("url") == explicit.get("url")]
            if len(matches) != 1:
                return None
        if len(matches) > 1:
            raise ValueError(f"multiple registered {kind} services; select model_aliases or an explicit task service")
        return matches[0] if matches else None

    for kind, generator in value.get("generators", {}).items():
        model = registration(kind, generator.get("service"))
        if model:
            if generator.get("service") is None:
                generator["service"] = copy.deepcopy(model["service"])
            generator["request"] = {**copy.deepcopy(model.get("request_defaults", {})), **generator.get("request", {})}
    configured = {kind for kind in ("af3score", "torchscore") if evaluation.get(kind) is not None}
    registered = {item.get("kind") for item in known.values() if item.get("kind") in {"af3score", "torchscore"}}
    evaluation["scoring_backend"] = choose_scorer(
        evaluation.get("scoring_backend") or ("torchscore" if value.get("autonomous") is not None else None),
        configured,
        registered,
    )
    selected = evaluation["scoring_backend"]
    kinds = [selected, "torchfold"]
    if value.get("autonomous") is not None:
        kinds.extend(["rosetta", "torchopt"])
    for kind in kinds:
        model = registration(kind, evaluation.get(kind))
        if model:
            if evaluation.get(kind) is None:
                evaluation[kind] = copy.deepcopy(model["service"])
            defaults = copy.deepcopy(model.get("request_defaults", {}))
            if defaults:
                option_key = kind + "_options"
                evaluation[option_key] = {**defaults, **evaluation.get(option_key, {})}
    config = InitialRoundConfig.model_validate(value)
    value = config.model_dump(mode="json")
    if name:
        value["name"] = name
    if pdb_path is not None:
        local = pdb_path.expanduser().resolve()
        if not local.is_file():
            raise ValueError("input PDB must exist on the client's mounted shared filesystem")
        remote = SharedFiles(config.path_mappings).remote(local)
        if "ppiflow" in value["generators"]:
            generator = value["generators"]["ppiflow"]
            generator["request"]["input_pdb" if generator["request"]["mode"] == "binder" else "antigen_pdb"] = remote
        if "torchcraft" in value["generators"]:
            value["generators"]["torchcraft"]["request"]["pdb_path"] = remote
    return InitialRoundConfig.model_validate(value)


def _alive(pid: int, root: Path):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, TypeError):
        return False
    proc = Path(f"/proc/{pid}")
    if proc.exists():
        try:
            arguments = (proc / "cmdline").read_bytes().split(b"\0")
            if b"opendde_harness.plugin.protein_design.initial_round.worker" not in arguments:
                return False
            if str(root).encode() not in arguments:
                return False
            if ") Z " in (proc / "stat").read_text():
                return False
        except OSError:
            return False
    return True


def round_status(root: Path):
    root = root.expanduser().resolve()
    if not (root / "launch.json").is_file():
        raise ValueError("no detached initial round found in this run directory")
    launch = json.loads((root / "launch.json").read_text())
    state = (
        json.loads((root / "worker-status.json").read_text())
        if (root / "worker-status.json").is_file()
        else {"state": "starting"}
    )
    local_host = launch.get("hostname", socket.gethostname()) == socket.gethostname()
    if state["state"] in {"running", "starting"} and local_host and not _alive(launch["pid"], root):
        state = {
            "state": "interrupted",
            "error": "worker exited; inspect worker.log, then restart the same run to resume recorded remote jobs",
        }
    checkpoint = json.loads((root / "checkpoint.json").read_text()) if (root / "checkpoint.json").is_file() else {}
    jobs = [
        {"stage": key, "state": job.get("state"), "remote_status": job.get("remote_status"), "remote_id": job.get("id")}
        for key, job in checkpoint.get("jobs", {}).items()
    ]
    if "summary" in state:
        summary = state["summary"]
        state["summary"] = {key: value for key, value in summary.items() if key not in {"ranking", "failed"}}
        state["summary"]["failed_count"] = len(summary.get("failed", []))
    return {
        **state,
        "run_dir": str(root),
        "pid": launch["pid"],
        "worker_host": launch.get("hostname"),
        "jobs": jobs[-8:],
        "total_jobs": len(jobs),
        "pending_regions": checkpoint.get("pending_regions"),
        "resume_tool": "protein_design_resume",
        "job_counts": dict(Counter(job["state"] for job in jobs)),
        "api_cleanup": checkpoint.get("api_cleanup", {}),
        "summary_path": str(root / "summary.json"),
        "log_path": str(root / "worker.log"),
    }


def _start_round(config: InitialRoundConfig, *, attach_jobs: dict[str, str] | None = None, retry_failed: bool = False):
    attach_jobs = dict(attach_jobs or {})
    root = (config.work_dir / config.name).resolve()
    SharedFiles(config.path_mappings).remote(root)
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / "launch.lock", blocking=False):
        files = SharedFiles(config.path_mappings)
        inputs = {}
        for source, generator in config.generators.items():
            paths = [
                files.local(generator.request[field])
                for field in (
                    "antigen_pdb",
                    "input_pdb",
                    "framework_dir",
                    "pdb_path",
                    "target_full_seq",
                    "framework_cif_path",
                )
                if field in generator.request
            ]
            if generator.fold_template:
                paths.append(generator.fold_template)
            for path in paths:
                if not path.exists():
                    raise ValueError(f"pipeline input does not exist: {path}")
                for item in sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]:
                    inputs[f"{source}/{item.resolve()}"] = hashlib.sha256(item.read_bytes()).hexdigest()
        input_snapshot = root / "launch-inputs.json"
        if input_snapshot.exists() and json.loads(input_snapshot.read_text()) != inputs:
            raise ValueError("pipeline input contents changed; choose a new run name")
        snapshot = root / "worker-config.json"
        value = config.model_dump(mode="json")
        if snapshot.exists():
            differences = config_difference_paths(
                status_profile_identity(json.loads(snapshot.read_text())), status_profile_identity(value)
            )
            if differences:
                raise ValueError(
                    "run configuration changed at "
                    + ", ".join(differences[:32])
                    + f"; compare {snapshot} with the requested config before choosing a new run name"
                )
        if (root / "launch.json").exists():
            current = round_status(root)
            if current["state"] in {"starting", "running", "completed"}:
                if attach_jobs:
                    raise ValueError("cannot attach jobs to an active or completed worker; stop it before recovery")
                return current
        # A foreground runner also holds this lock. Do not start a second one.
        with file_lock(root / "run.lock", blocking=False):
            if attach_jobs:
                checkpoint_path = root / "checkpoint.json"
                jobs = json.loads(checkpoint_path.read_text()).get("jobs", {}) if checkpoint_path.exists() else {}
                for key, job_id in attach_jobs.items():
                    if key not in jobs:
                        raise ValueError(f"--attach-job key must exist in checkpoint: {key}")
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
                        raise ValueError("invalid attached job ID")
                    if jobs[key].get("id") and jobs[key]["id"] != job_id:
                        raise ValueError(f"cannot replace existing job ID for {key}")
            write_json(snapshot, value)
            write_json(input_snapshot, inputs)
        (root / "worker-status.json").unlink(missing_ok=True)
        command = [sys.executable, "-m", "opendde_harness.plugin.protein_design.initial_round.worker", str(root)]
        if attach_jobs:
            command.extend(["--attach-jobs", json.dumps(attach_jobs)])
        if retry_failed:
            command.append("--retry-failed")
        with (root / "worker.log").open("ab") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        write_json(root / "launch.json", {"pid": process.pid, "hostname": socket.gethostname()})
        threading.Thread(target=process.wait, daemon=True).start()
        return {"state": "starting", "run_dir": str(root), "pid": process.pid, "log_path": str(root / "worker.log")}


def resume_round(root: Path):
    """Resume the immutable launch snapshot rather than reconstructing task inputs."""
    root = root.expanduser().resolve()
    snapshot = root / "worker-config.json"
    if not snapshot.is_file():
        snapshot = root / "run-config.json"
    config = InitialRoundConfig.model_validate(json.loads(snapshot.read_text()))
    if (config.work_dir / config.name).resolve() != root:
        raise ValueError("run directory does not match the saved worker configuration")
    return start_round(config, retry_failed=True)


def run_index_path():
    from opendde_harness.config.loader import get_config_path

    return get_config_path().parent / "protein_initial_round_runs.json"


def remember_run(root: Path):
    path = run_index_path()
    root = root.expanduser().resolve()
    with file_lock(path.with_suffix(".lock")):
        runs = json.loads(path.read_text()) if path.is_file() else {}
        runs[str(root)] = {"run_dir": str(root)}
        write_json(path, runs)


def start_round(config: InitialRoundConfig, *, attach_jobs: dict[str, str] | None = None, retry_failed: bool = False):
    # Fail closed before launching: a task must be discoverable without chat history.
    remember_run(config.work_dir / config.name)
    return _start_round(config, attach_jobs=attach_jobs, retry_failed=retry_failed)


def list_rounds(work_dir: Path | None = None):
    path = run_index_path()
    indexed = json.loads(path.read_text()) if path.is_file() else {}
    roots = {Path(item["run_dir"]).expanduser().resolve() for item in indexed.values()}
    if work_dir is not None:
        directory = work_dir.expanduser().resolve()
        roots.update(item.parent for item in directory.glob("*/worker-config.json"))
    runs = []
    for root in sorted(roots, key=str):
        try:
            value = round_status(root)
            value.pop("jobs", None)
            value["name"] = root.name
            value["local_worker_check"] = value["worker_host"] == socket.gethostname()
            if not value["local_worker_check"] and value["state"] in {"starting", "running"}:
                value["state"] = "remote_worker_unverified"
            runs.append(value)
        except (OSError, ValueError, KeyError) as exc:
            if (root / "run-config.json").is_file():
                summary = root / "summary.json"
                value = json.loads(summary.read_text()) if summary.is_file() else {}
                runs.append(
                    {
                        "run_dir": str(root),
                        "name": root.name,
                        "state": value.get("status", "foreground_unverified"),
                        "resume_tool": "protein_design_resume",
                    }
                )
                continue
            runs.append({"run_dir": str(root), "state": "unavailable", "error_type": type(exc).__name__})
    return {
        "index_path": str(path),
        "runs": runs,
        "hint": "Use run_dir with protein_design_status/protein_design_resume; no new task name is needed. "
        "Remote worker liveness cannot be proven from a different host. "
        "If this instance has no index, provide the original shared work_dir to discover older runs.",
    }
