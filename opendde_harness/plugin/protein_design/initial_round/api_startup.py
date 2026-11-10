"""Start an API allocation once, then connect to its assigned Slurm node."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

import httpx

from opendde_harness.utils.portable_lock import LockTimeoutError, file_lock

from .config import Service
from .storage import fingerprint, write_json


class ApiStartupError(RuntimeError):
    """Actionable local startup error, containing no remote HTTP body or credentials."""


class ApiStartupTimeoutError(ApiStartupError, TimeoutError):
    """An API allocation remains pending and can be reused on retry."""


async def slurm_command(config: Service, *args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), config.slurm_command_timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise ApiStartupError(f"{args[0]} failed; inspect Slurm permissions and API allocation logs")
    return stdout.decode().strip()


async def healthy(client: httpx.AsyncClient, path: str) -> bool:
    try:
        response = await client.get(path.lstrip("/"))
        response.raise_for_status()
        value = response.json()
        return (
            isinstance(value, dict)
            and value.get("ok") is not False
            and str(value.get("status", "ok")).lower() not in {"error", "failed", "unhealthy"}
        )
    except (httpx.HTTPError, ValueError):
        return False


async def ensure_api(config: Service, client: httpx.AsyncClient, *, start: bool = True) -> None:
    """Persist the allocation before submitting, so an uncertain sbatch is never retried."""
    if config.sbatch_script is None:
        return
    script = config.sbatch_script
    state_path = config.sbatch_state_file or script.parent / (
        ".harness-api-" + fingerprint({"script": str(script), "url": config.url})[:16] + ".json"
    )
    deadline = asyncio.get_running_loop().time() + config.startup_timeout
    while True:
        try:
            with file_lock(state_path.with_suffix(".lock"), blocking=False):
                await _ensure_locked(config, client, script, state_path, deadline, start=start)
                return
        except LockTimeoutError:
            if not start or asyncio.get_running_loop().time() >= deadline:
                raise ApiStartupError(
                    "another process is starting this API; retry after its startup finishes"
                ) from None
            await asyncio.sleep(config.startup_poll_interval)


async def _ensure_locked(config, client, script: Path, state_path: Path, deadline: float, *, start: bool):
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if not script.is_file():
        raise ValueError(f"API sbatch script does not exist: {script}")
    signature = fingerprint(
        {"script": str(script), "sha256": hashlib.sha256(script.read_bytes()).hexdigest(), "url": config.url}
    )
    if state is not None and state.get("signature") != signature:
        raise ValueError("API script or URL changed; reconcile the old allocation and use a new sbatch_state_file")
    if state is None:
        if not start:
            raise ApiStartupError("API has no saved sbatch allocation; --check does not submit API allocations")
        state = {"signature": signature, "state": "submission_uncertain"}
        write_json(state_path, state)
        # A timeout or lost stdout may still mean the scheduler accepted the job.
        output = await slurm_command(config, "sbatch", "--parsable", str(script))
        if not re.fullmatch(r"[0-9]+", output):
            raise ValueError("sbatch must return a numeric local-cluster job ID")
        state.update(job_id=output, state="submitted")
        write_json(state_path, state)
    job_id = str(state.get("job_id", ""))
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ApiStartupError(
            f"API submission uncertain; reconcile Slurm and record job_id in {state_path}; do not resubmit blindly"
        )
    while True:
        status = await slurm_command(config, "squeue", "--jobs", job_id, "--noheader", "--format=%T|%N")
        rows = status.splitlines()
        if len(rows) != 1 or "|" not in rows[0]:
            raise ApiStartupError(
                f"API allocation {job_id} is no longer queued; inspect Slurm logs before resetting {state_path}"
            )
        phase, nodes = rows[0].strip().split("|", 1)
        state["slurm_state"] = phase
        if phase == "RUNNING":
            if config.sbatch_resolve_host:
                hosts = (await slurm_command(config, "scontrol", "show", "hostnames", nodes)).splitlines()
                if len(hosts) != 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", hosts[0]):
                    raise ValueError("API sbatch allocation must resolve to exactly one hostname")
                client.base_url = httpx.URL(config.url.rstrip("/") + "/").copy_with(host=hosts[0])
            if await healthy(client, config.health_path):
                state.update(state="ready", resolved_url=str(client.base_url))
                write_json(state_path, state)
                return
        elif phase not in {"PENDING", "CONFIGURING"}:
            raise ApiStartupError(f"API allocation {job_id} is {phase}; inspect Slurm logs")
        write_json(state_path, state)
        if not start or asyncio.get_running_loop().time() >= deadline:
            raise ApiStartupTimeoutError(
                f"API allocation {job_id} is not healthy yet; retry to reuse it (not resubmit)"
            )
        await asyncio.sleep(config.startup_poll_interval)
