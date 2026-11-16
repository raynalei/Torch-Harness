"""Atomic checkpoints and explicit paths for a mounted HPC filesystem."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import InitialRoundConfig, PathMapping


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class SharedFiles:
    def __init__(self, mappings: list[PathMapping]):
        self.mappings = mappings

    def local(self, remote: str) -> Path:
        source = Path(remote)
        for mapping in sorted(self.mappings, key=lambda m: len(m.remote), reverse=True):
            try:
                tail = source.relative_to(mapping.remote)
            except ValueError:
                continue
            root = mapping.local.resolve()
            result = (root / tail).resolve()
            if not result.is_relative_to(root):
                raise ValueError(f"artifact escapes its shared root: {remote}")
            return result
        raise ValueError(f"no shared filesystem mapping for remote path: {remote}")

    def remote(self, local: Path) -> str:
        source = local.resolve()
        for mapping in sorted(self.mappings, key=lambda m: len(str(m.local)), reverse=True):
            root = mapping.local.resolve()
            if source.is_relative_to(root):
                return str(Path(mapping.remote) / source.relative_to(root))
        raise ValueError(f"path is not on the shared filesystem: {local}")


def select(value: Any, field: str, default=None):
    """Select documented response fields; profiles can use dotted object paths."""
    for key in field.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def status_profile_identity(value):
    """Compare validated snapshot meaning, allowing response-state selector corrections."""
    normalized = copy.deepcopy(value)
    workflow = normalized.get("config", normalized)
    if not isinstance(workflow, dict):
        return normalized
    if {"name", "work_dir", "generators", "evaluation"} <= workflow.keys():
        try:
            canonical = InitialRoundConfig.model_validate(workflow).model_dump(mode="json")
        except ValidationError:
            # Do not drop unknown/removed scientific settings to force an old run through.
            pass
        else:
            if "config" in normalized:
                normalized["config"] = canonical
            else:
                normalized = canonical
            workflow = canonical
    services = [g.get("service") for g in workflow.get("generators", {}).values() if isinstance(g, dict)]
    evaluation = workflow.get("evaluation", {})
    services.extend(evaluation.get(kind) for kind in ("af3score", "torchscore", "torchfold", "rosetta", "torchopt"))
    for service in services:
        if isinstance(service, dict) and "url" in service:
            service.pop("status_field", None)
    return normalized


def config_difference_paths(previous, current, prefix=""):
    """Report keys only: config values may contain private paths or credentials."""
    if isinstance(previous, dict) and isinstance(current, dict):
        paths = []
        for key in sorted(previous.keys() | current.keys()):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in previous or key not in current:
                paths.append(child)
            else:
                paths.extend(config_difference_paths(previous[key], current[key], child))
        return paths
    if isinstance(previous, list) and isinstance(current, list):
        paths = [f"{prefix}.length"] if len(previous) != len(current) else []
        for index, (old, new) in enumerate(zip(previous, current)):
            paths.extend(config_difference_paths(old, new, f"{prefix}[{index}]"))
        return paths
    return [prefix or "<root>"] if previous != current else []


def response_state(value, field="status"):
    """Read status/state aliases, rejecting contradictory evidence."""

    def normalize(raw):
        parts = str(raw or "UNKNOWN").upper().split()
        state = parts[0].rstrip("+") if parts else "UNKNOWN"
        return {"SUCCESS": "COMPLETED", "SUCCEEDED": "COMPLETED", "CANCELED": "CANCELLED"}.get(state, state)

    selected = select(value, field)
    if field not in {"status", "state"}:
        return normalize(selected)
    alternate = select(value, "state" if field == "status" else "status")
    if selected is not None and alternate is not None and normalize(selected) != normalize(alternate):
        raise ValueError("API response has conflicting status/state fields; correct the response profile")
    return normalize(selected if selected is not None else alternate)


class Checkpoint:
    def __init__(self, root: Path, config: dict[str, Any]):
        self.path = root / "checkpoint.json"
        signature = fingerprint(config)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data.get("config_hash") != signature:
                if self.data.get("jobs") or self.data.get("candidates"):
                    previous = self.data.get("config_snapshot")
                    if previous is None and "config" in config and "inputs" in config:
                        legacy = root / "run-config.json"
                        if legacy.is_file():
                            previous = {"config": json.loads(legacy.read_text()), "inputs": config["inputs"]}
                    if previous is None or fingerprint(previous) != self.data.get("config_hash"):
                        raise ValueError(
                            "configuration changed or original hash evidence is missing; preserve checkpoint and run-config.json for recovery"
                        )
                    differences = config_difference_paths(
                        status_profile_identity(previous), status_profile_identity(config)
                    )
                    if differences:
                        raise ValueError(
                            "configuration changed at "
                            + ", ".join(differences[:32])
                            + "; inspect saved inputs/settings before starting a new experiment"
                        )
                    self.data.setdefault("response_profile_history", []).append(
                        {
                            "previous_hash": self.data["config_hash"],
                            "current_hash": signature,
                            "change": "compatible snapshot serialization/status_field",
                            "serialized_differences": config_difference_paths(previous, config),
                        }
                    )
                self.data["config_hash"] = signature
        else:
            self.data = {"version": 1, "config_hash": signature, "jobs": {}}
        self.data["config_snapshot"] = copy.deepcopy(config)
        self.save()

    def save(self):
        write_json(self.path, self.data)


def retry_failed_jobs(checkpoint):
    """Archive only confirmed terminal failures; unknown submissions keep their IDs."""
    retried = []
    for key, job in list(checkpoint.data["jobs"].items()):
        if job.get("state") != "failed":
            continue
        status = job.get("status", {})
        terminal = str(job.get("remote_status", "")).upper() in {
            "FAILED",
            "CANCELLED",
            "CANCELED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
            "REVOKED",
        }
        nonzero = job.get("remote_status") in {"COMPLETED", "SUCCEEDED", "SUCCESS"} and (
            status.get("exit_code") is not None and str(status["exit_code"]) not in {"0", "0:0"}
        )
        if not (terminal or nonzero or job.get("retry_safe") is True):
            continue
        checkpoint.data.setdefault("job_attempts", {}).setdefault(key, []).append(job)
        del checkpoint.data["jobs"][key]
        retried.append(key)
    checkpoint.save()
    return retried
