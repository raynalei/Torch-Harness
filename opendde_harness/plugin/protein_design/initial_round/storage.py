"""Atomic checkpoints and explicit paths for a mounted HPC filesystem."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import PathMapping


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


class Checkpoint:
    def __init__(self, root: Path, config: dict[str, Any]):
        self.path = root / "checkpoint.json"
        signature = fingerprint(config)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data.get("config_hash") != signature:
                if self.data.get("jobs") or self.data.get("candidates"):
                    raise ValueError("configuration changed; use a new work_dir rather than resuming incompatible jobs")
                self.data["config_hash"] = signature
                self.save()
        else:
            self.data = {"version": 1, "config_hash": signature, "jobs": {}}
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
