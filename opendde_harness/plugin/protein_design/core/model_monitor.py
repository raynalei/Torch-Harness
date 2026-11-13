"""Detached tracking and API cleanup for standalone Harness model requests."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from opendde_harness.plugin.protein_design.core.model_registry import ProteinModelRegistry
from opendde_harness.plugin.protein_design.initial_round.storage import write_json
from opendde_harness.utils.portable_lock import LockTimeoutError, file_lock

MODULE = "opendde_harness.plugin.protein_design.core.model_monitor"
TERMINAL = {"succeeded", "prepared", "failed", "cancelled"}


def _alive(launch):
    if launch.get("hostname") != socket.gethostname():
        return False
    try:
        pid = int(launch["pid"])
        os.kill(pid, 0)
        proc = Path(f"/proc/{pid}")
        if proc.exists():
            return (
                MODULE.encode() in (proc / "cmdline").read_bytes().split(b"\0")
                and ") Z " not in (proc / "stat").read_text()
            )
        return True
    except (KeyError, OSError, ValueError):
        return False


def cleanup_pending(record):
    if not record.get("api_allocation"):
        return False
    cleanup = record.get("api_cleanup", [])
    return not cleanup or any(item.get("state") not in {"cancel_requested", "not_in_queue"} for item in cleanup)


def start_monitor(registry, request_id):
    record, model = registry.read(request_id)
    root = registry.root.resolve()
    state_path = root / f"{request_id}.monitor.json"
    with file_lock(root / f"{request_id}.monitor-launch.lock", blocking=False):
        if state_path.is_file():
            launch = json.loads(state_path.read_text())
            if _alive(launch):
                progress = root / f"{request_id}.monitor-status.json"
                return {**launch, **(json.loads(progress.read_text()) if progress.is_file() else {})}
        if record.get("terminal_status") and not cleanup_pending(record):
            return {"state": "completed", "request_id": request_id}
        snapshot = root / f"{request_id}.monitor-config.json"
        write_json(snapshot, {record["model"]: model.model_dump(mode="json")})
        log_path = root / f"{request_id}.monitor.log"
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", MODULE, str(root), request_id],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        launch = {
            "state": "starting",
            "pid": process.pid,
            "hostname": socket.gethostname(),
            "request_id": request_id,
            "log_path": str(log_path),
        }
        write_json(state_path, launch)
        threading.Thread(target=process.wait, daemon=True).start()
        return launch


async def watch(registry, request_id):
    _, model = registry.read(request_id)
    state_path = registry.root / f"{request_id}.monitor-status.json"
    deadline = time.monotonic() + model.service.job_timeout
    while time.monotonic() < deadline:
        try:
            current = await registry.status(request_id)
            record, _ = registry.read(request_id)
            cleanup_failed = cleanup_pending(record)
            state = "completed" if current["state"] in TERMINAL and not cleanup_failed else "watching"
            write_json(state_path, {"state": state, "request_id": request_id, "task": current})
            if state == "completed":
                return
        except Exception as exc:
            # Keep remote allocations on uncertainty; a failed query does not prove completion.
            write_json(state_path, {"state": "retrying", "request_id": request_id, "error_type": type(exc).__name__})
        await asyncio.sleep(min(60, max(10, model.service.poll_interval)))
    write_json(
        state_path,
        {
            "state": "monitor_timeout",
            "request_id": request_id,
            "hint": "Remote state is unresolved; API retained. Query status/results to resume tracking.",
        },
    )


def run(root, request_id):
    ProteinModelRegistry.valid_id(request_id)
    try:
        with file_lock(root / f"{request_id}.monitor.lock", blocking=False):
            registrations = json.loads((root / f"{request_id}.monitor-config.json").read_text())
            asyncio.run(watch(ProteinModelRegistry(registrations, root), request_id))
    except LockTimeoutError:
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("request_id")
    args = parser.parse_args()
    run(args.root, args.request_id)
