"""Detached deterministic initial rounds: the worker, not Qwen, polls Slurm."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path

from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig
from opendde_harness.plugin.protein_design.initial_round.storage import SharedFiles, write_json
from opendde_harness.utils.portable_lock import file_lock


def prepare_run(config_path: Path, *, pdb_path: Path | None = None, name: str | None = None):
    config = InitialRoundConfig.load(config_path)
    value = config.model_dump(mode="json")
    if name:
        value["name"] = name
    if pdb_path is not None:
        local = pdb_path.expanduser().resolve()
        if not local.is_file():
            raise ValueError("input PDB must exist on the client's mounted shared filesystem")
        remote = SharedFiles(config.path_mappings).remote(local)
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
        "job_counts": dict(Counter(job["state"] for job in jobs)),
        "summary_path": str(root / "summary.json"),
        "log_path": str(root / "worker.log"),
    }


def start_round(config: InitialRoundConfig):
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
        if snapshot.exists() and json.loads(snapshot.read_text()) != value:
            raise ValueError("run configuration changed; choose a new run name")
        if (root / "launch.json").exists():
            current = round_status(root)
            if current["state"] in {"starting", "running", "completed"}:
                return current
        # A foreground runner also holds this lock. Do not start a second one.
        with file_lock(root / "run.lock", blocking=False):
            write_json(snapshot, value)
            write_json(input_snapshot, inputs)
        (root / "worker-status.json").unlink(missing_ok=True)
        with (root / "worker.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "opendde_harness.plugin.protein_design.initial_round.worker", str(root)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        write_json(root / "launch.json", {"pid": process.pid, "hostname": socket.gethostname()})
        threading.Thread(target=process.wait, daemon=True).start()
        return {"state": "starting", "run_dir": str(root), "pid": process.pid, "log_path": str(root / "worker.log")}
