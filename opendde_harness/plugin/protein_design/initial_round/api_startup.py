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
        stdout, stderr = await asyncio.wait_for(process.communicate(), config.slurm_command_timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode:
        log = config.sbatch_state_file.with_suffix(".slurm.log") if config.sbatch_state_file else None
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("ab") as output:
                output.write((f"{args[0]} exit_code={process.returncode}\n").encode() + stdout + b"\n" + stderr + b"\n")
        raise ApiStartupError(f"{args[0]} failed (exit {process.returncode}); inspect {log or 'Slurm logs'}")
    return stdout.decode().strip()


async def allocation_status(config: Service, job_id: str, state: dict) -> tuple[str, str]:
    """An empty live queue is unknown until accounting confirms the allocation state."""
    try:
        status = await slurm_command(config, "squeue", "--jobs", job_id, "--noheader", "--format=%T|%N")
        rows = status.splitlines()
        if len(rows) == 1 and "|" in rows[0] and rows[0].split("|", 1)[0].strip():
            state.pop("query_error", None)
            state["query_failures"] = 0
            phase, nodes = rows[0].strip().split("|", 1)
            return phase.split()[0].rstrip("+"), nodes.strip()
        state["queue_observation"] = "not_visible"
    except (ApiStartupError, OSError, TimeoutError) as exc:
        state["query_error"] = str(exc)
        state["query_failures"] = state.get("query_failures", 0) + 1
    try:
        accounting = await slurm_command(
            config,
            "sacct",
            "--jobs",
            job_id,
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,State,ExitCode,NodeList",
        )
        for row in accounting.splitlines():
            fields = row.strip().split("|")
            # Ignore .batch/.extern and other steps: only the exact allocation counts.
            if len(fields) >= 4 and fields[0] == job_id and fields[1].strip():
                state["query_failures"] = 0
                state["exit_code"] = fields[2]
                return fields[1].split()[0].rstrip("+"), fields[3].strip()
    except (ApiStartupError, OSError, TimeoutError) as exc:
        state["accounting_error"] = str(exc)
    if state.get("query_failures", 0) >= 3:
        raise ApiStartupError(
            f"API allocation {job_id}: Slurm queries failed 3 times; "
            f"{state.get('query_error')}; accounting: {state.get('accounting_error', 'not yet visible')}"
        )
    return "UNKNOWN", ""


async def healthy(client: httpx.AsyncClient, path: str) -> bool:
    try:
        response = await client.get(path.lstrip("/"))
        response.raise_for_status()
        value = response.json()
        return (
            isinstance(value, dict)
            and value.get("backend_available") is not False
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
        allocation_name = "harness-api-" + fingerprint(str(state_path.resolve()))[:20]
        state = {"signature": signature, "state": "submission_uncertain", "allocation_name": allocation_name}
        write_json(state_path, state)
        # A timeout or lost stdout may still mean the scheduler accepted the job.
        output = await slurm_command(config, "sbatch", "--parsable", f"--job-name={allocation_name}", str(script))
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
        try:
            phase, nodes = await allocation_status(config, job_id, state)
        except ApiStartupError:
            write_json(state_path, state)
            raise
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
        elif phase in {
            "BOOT_FAIL",
            "CANCELLED",
            "COMPLETED",
            "DEADLINE",
            "FAILED",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "REVOKED",
            "TIMEOUT",
        }:
            state["state"] = "terminated"
            write_json(state_path, state)
            raise ApiStartupError(
                f"API allocation {job_id} is {phase} (exit_code={state.get('exit_code', 'unknown')}); "
                f"inspect its sbatch output log; state file: {state_path}"
            )
        write_json(state_path, state)
        if not start or asyncio.get_running_loop().time() >= deadline:
            raise ApiStartupTimeoutError(
                f"API allocation {job_id} is not healthy yet (Slurm state={phase}); "
                f"inspect {state_path}; retry within this startup attempt to reuse its ID (not resubmit)"
            )
        await asyncio.sleep(config.startup_poll_interval)


def allocation_record(config: Service):
    """Retain ownership separately from disposable HTTP connections."""
    path = config.sbatch_state_file
    if config.sbatch_script is None or path is None or not path.is_file():
        return None
    state = json.loads(path.read_text())
    if not state.get("job_id") or not state.get("allocation_name"):
        return None
    return {"state_file": str(path), **{k: state[k] for k in ("job_id", "allocation_name", "signature")}}


def allocation_was_released(record):
    if not record or not Path(record["state_file"]).is_file():
        return False
    return json.loads(Path(record["state_file"]).read_text()).get("cleanup", {}).get("state") in {
        "cancel_requested",
        "not_in_queue",
    }


async def release_allocation(config: Service, record: dict | None):
    """Cancel only an owned API allocation, after all inference/results are done."""
    if config.sbatch_script is None or not record:
        return {"state": "not_owned"}
    path = Path(record["state_file"])
    with file_lock(path.with_suffix(".lock"), blocking=False):
        state = json.loads(path.read_text())
        expected_name = "harness-api-" + fingerprint(str(path.resolve()))[:20]
        if any(state.get(k) != record.get(k) for k in ("job_id", "allocation_name", "signature")) or (
            record.get("allocation_name") != expected_name or not re.fullmatch(r"[0-9]+", str(record.get("job_id", "")))
        ):
            raise ApiStartupError("API allocation ownership does not match; no cancellation attempted")
        if state.get("cleanup", {}).get("state") in {"cancel_requested", "not_in_queue"}:
            return state["cleanup"]
        rows = (
            await slurm_command(config, "squeue", "--jobs", str(record["job_id"]), "--noheader", "--format=%T|%j")
        ).splitlines()
        if not rows:
            phase, _ = await allocation_status(config, str(record["job_id"]), state)
            terminal = {
                "COMPLETED",
                "CANCELLED",
                "FAILED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "PREEMPTED",
                "BOOT_FAIL",
                "DEADLINE",
                "REVOKED",
            }
            result = {
                "state": "not_in_queue" if phase in terminal else "cleanup_deferred",
                "job_id": record["job_id"],
                "slurm_state": phase,
            }
        else:
            if len(rows) != 1 or rows[0].strip().split("|")[-1] != expected_name:
                raise ApiStartupError("scheduler job name differs from owned allocation; no cancellation attempted")
            await slurm_command(config, "scancel", str(record["job_id"]))
            result = {"state": "cancel_requested", "job_id": record["job_id"]}
        state["cleanup"] = result
        write_json(path, state)
        return result
