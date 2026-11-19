"""Durable direct Slurm submission; never starts or contacts a Rosetta API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import shlex
import shutil
import time
from pathlib import Path
from uuid import uuid4

from .api_startup import slurm_command as command
from .services import RemoteJobFailedError, SubmissionUncertainError
from .storage import fingerprint, write_json

TERMINAL_FAILURES = {
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RosettaSlurmService:
    def __init__(self, config, checkpoint, files):
        if config.backend != "slurm":
            raise ValueError("direct Rosetta requires backend=slurm")
        self.config, self.checkpoint, self.files = config, checkpoint, files
        self.allocations = []

    async def close(self):
        pass

    async def release_allocations(self):
        return []  # There is no service allocation to cancel.

    def check(self):
        if not self.config.python.is_file():
            raise ValueError(f"PyRosetta interpreter does not exist: {self.config.python}")
        for executable in ("sbatch", "squeue", "sacct"):
            if shutil.which(executable) is None:
                raise ValueError(f"direct Rosetta requires Slurm command {executable} on the Harness host")

    async def scheduler_status(self, identifier):
        self.scheduler_error = None
        try:
            queue = await command(self.config, "squeue", "--noheader", "--jobs", identifier, "--format=%i|%T")
        except (RuntimeError, OSError, TimeoutError) as exc:
            self.scheduler_error = str(exc)
            queue = ""
        for row in queue.splitlines():
            fields = row.strip().split("|")
            if len(fields) >= 2 and fields[0] == identifier and fields[1] not in {"COMPLETED", *TERMINAL_FAILURES}:
                return fields[1], None
        try:
            accounting = await command(
                self.config,
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs",
                identifier,
                "--format=JobIDRaw,State,ExitCode",
            )
        except (RuntimeError, OSError, TimeoutError) as exc:
            self.scheduler_error = str(exc)
            return "UNKNOWN", None
        for row in accounting.splitlines():
            fields = row.strip().split("|")
            if len(fields) >= 3 and fields[0] == identifier:
                if not fields[1].strip():
                    continue
                exit_code = fields[2].strip()
                if not re.fullmatch(r"[0-9]+(?::[0-9]+)?", exit_code):
                    exit_code = None
                return fields[1].split()[0].rstrip("+"), exit_code
        return "UNKNOWN", None  # Accounting lag is not successful completion.

    def read_result(self, job):
        path = Path(job["result_path"])
        directory = path.parent.parent
        for name, expected in (
            ("worker.py", job["worker_sha256"]),
            ("input.pdb", job["input_sha256"]),
            ("request.json", job["metadata_sha256"]),
            ("job.sbatch", job["script_sha256"]),
        ):
            if digest(directory / name) != expected:
                raise ValueError("Rosetta job input/worker/script changed")
        result = json.loads(path.read_text())
        if result.get("request_hash") != job["request_hash"] or result.get("input_sha256") != job["input_sha256"]:
            raise ValueError("Rosetta result does not match the submitted request/input")
        records = result.get("metrics", [])
        if len(records) != 1 or any(
            type(records[0].get(key)) not in (int, float) or not math.isfinite(records[0][key])
            for key in ("dG_separated", "total_score")
        ):
            raise ValueError("missing/non-finite Rosetta interface metrics")
        expected = {self.files.remote(path.parent / name) for name in ("interface.sc", "residue_energies.out")}
        if set(result.get("files", [])) != expected:
            raise ValueError("Rosetta output manifest has unexpected paths")
        for remote in expected:
            if digest(self.files.local(remote)) != result.get("sha256", {}).get(remote):
                raise ValueError("Rosetta output changed or is incomplete")
        if job.get("result_sha256") and digest(path) != job["result_sha256"]:
            raise ValueError("cached Rosetta result changed")
        return result

    async def submit(self, key, payload, attach_id=None):
        if set(payload) - {"input_pdb", "chain_group_a", "chain_group_b", "pack_input", "pack_separated"}:
            raise ValueError("unsupported direct interface_analyze parameters")
        for flag in ("pack_input", "pack_separated"):
            if flag in payload and type(payload[flag]) is not bool:
                raise ValueError("Rosetta packing flags must be booleans")
        if any(
            not re.fullmatch(r"[A-Z]{1,10}", payload.get(key, "")) for key in ("chain_group_a", "chain_group_b")
        ) or set(payload["chain_group_a"]) & set(payload["chain_group_b"]):
            raise ValueError("Rosetta requires distinct uppercase PDB chain groups")
        source = self.files.local(payload["input_pdb"])
        signature = fingerprint({"backend": "rosetta_slurm", "payload": payload, "input_sha256": digest(source)})
        jobs = self.checkpoint.data["jobs"]
        job = jobs.get(key)
        if job and job.get("backend") != "rosetta_slurm":
            raise SubmissionUncertainError(f"{key}: existing API job must be reconciled before changing its backend")
        if job and job["request_hash"] != signature:
            raise ValueError(f"Rosetta request/input changed: {key}")
        if attach_id is not None:
            if not re.fullmatch(r"[1-9][0-9]*", attach_id) or job is None:
                raise ValueError("attach a numeric Slurm job ID to an existing uncertain Rosetta request")
            if job.get("id") and job["id"] != attach_id:
                raise ValueError("cannot replace an existing Rosetta Slurm job ID")
            job.update(id=attach_id, state="submitted")
            self.checkpoint.save()
        if job is not None:
            return job
        self.check()
        directory = self.checkpoint.path.parent / "rosetta-slurm" / fingerprint(key)[:16] / uuid4().hex
        directory.mkdir(parents=True)
        shutil.copy2(source, directory / "input.pdb")
        worker = Path(__file__).with_name("rosetta_worker.py")
        shutil.copy2(worker, directory / "worker.py")
        output = directory / "output"
        output.mkdir()
        metadata = {
            "request": payload,
            "request_hash": signature,
            "input_sha256": digest(source),
            "input_pdb": self.files.remote(directory / "input.pdb"),
            "output_dir": self.files.remote(output),
        }
        write_json(directory / "request.json", metadata)
        script = directory / "job.sbatch"
        script.write_text(
            "#!/bin/bash\nset -euo pipefail\nexec "
            + shlex.join(
                [
                    str(self.config.python),
                    self.files.remote(directory / "worker.py"),
                    "--request",
                    self.files.remote(directory / "request.json"),
                ]
            )
            + "\n"
        )
        job = jobs[key] = {
            "backend": "rosetta_slurm",
            "request_hash": signature,
            "input_sha256": digest(source),
            "state": "submission_uncertain",
            "submitted_at": time.time(),
            "result_path": str(output / "result.json"),
            "script": str(script),
            "log_path": str(directory / "slurm.out"),
            "worker_sha256": digest(worker),
            "metadata_sha256": digest(directory / "request.json"),
            "script_sha256": digest(script),
        }
        self.checkpoint.save()  # A crash after this point must never cause a duplicate sbatch.
        args = [
            "sbatch",
            "--parsable",
            f"--job-name=harness-rosetta-{directory.name[:16]}",
            f"--cpus-per-task={self.config.cpus}",
            f"--mem={self.config.memory}",
            f"--time={self.config.walltime}",
            f"--output={self.files.remote(directory / 'slurm.out')}",
            "--export=ALL",
        ]
        if self.config.partition:
            args.append(f"--partition={self.config.partition}")
        args.append(self.files.remote(script))
        try:
            submitted = (await command(self.config, *args)).strip().split(";", 1)[0]
            if not re.fullmatch(r"[1-9][0-9]*", submitted):
                raise ValueError("sbatch did not return a numeric job ID")
        except BaseException as exc:
            job["submission_error"] = type(exc).__name__
            self.checkpoint.save()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise SubmissionUncertainError(
                f"{key}: sbatch outcome uncertain; reconcile using --attach-job {key}=SLURM_JOB_ID"
            ) from exc
        job.update(id=submitted, state="submitted")
        self.checkpoint.save()
        return job

    async def run(self, key, route, id_field, payload, *, attach_id=None, status_base=None):
        if route != "interface_analyze":
            raise ValueError("direct PyRosetta workflow supports interface_analyze only")
        job = await self.submit(key, payload, attach_id)
        if job["state"] == "completed":
            return self.read_result(job)
        if job["state"] == "failed":
            raise RemoteJobFailedError(f"{key}: previous Rosetta failure; resume with retry_failed")
        if not job.get("id"):
            raise SubmissionUncertainError(f"{key}: reconcile uncertain sbatch using --attach-job {key}=SLURM_JOB_ID")
        deadline = time.monotonic() + self.config.job_timeout
        while time.monotonic() < deadline:
            state, exit_code = await self.scheduler_status(job["id"])
            job.update(remote_status=state, status={"state": state, "exit_code": exit_code})
            if self.scheduler_error:
                job["scheduler_error"] = self.scheduler_error
            self.checkpoint.save()
            if state in TERMINAL_FAILURES or (state == "COMPLETED" and exit_code not in (None, "0:0", "0")):
                job.update(state="failed", retry_safe=True)
                self.checkpoint.save()
                raise RemoteJobFailedError(f"{key}: Rosetta {state} ({exit_code}); see {job['log_path']}")
            if state == "COMPLETED" and exit_code in ("0:0", "0"):
                try:
                    result = self.read_result(job)
                except FileNotFoundError:
                    # Shared-filesystem visibility may lag completed accounting. Keep the job ID.
                    await asyncio.sleep(self.config.poll_interval)
                    continue
                except (ValueError, OSError, KeyError) as exc:
                    job.update(state="failed", retry_safe=True, error=str(exc))
                    self.checkpoint.save()
                    raise RemoteJobFailedError(f"{key}: invalid Rosetta output; see {job['log_path']}") from exc
                job.update(state="completed", results=result, result_sha256=digest(Path(job["result_path"])))
                self.checkpoint.save()
                return result
            await asyncio.sleep(self.config.poll_interval)
        raise TimeoutError(f"{key}: Rosetta job {job['id']} remains submitted; resume the same run_dir")
