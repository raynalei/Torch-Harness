"""TorchOptimization 2.0.13: receipts, scheduler completion and native artifacts."""

from __future__ import annotations

import asyncio
import re
from uuid import uuid4

from opendde_harness.plugin.protein_design.core.model_requests import validate_request

from . import api_startup
from .services import RemoteJobFailedError, SlurmService, SubmissionUncertainError
from .storage import fingerprint

JOB_ID = re.compile(r"[0-9]+(?:_[0-9]+)?")


def receipt_job_ids(receipts):
    """Read structured job-ID fields; never guess IDs from arbitrary log numbers."""
    found = set()

    def visit(value, job_field=False):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, key in {"job_id", "job_ids", "slurm_job_id", "slurm_job_ids", "jobs"})
        elif isinstance(value, list):
            for item in value:
                visit(item, job_field)
        elif job_field and isinstance(value, (str, int)) and not isinstance(value, bool):
            token = str(value).strip()
            if JOB_ID.fullmatch(token):
                found.add(token)

    visit(receipts)
    return sorted(found)


async def inspect_optimization(client, optimization_id, request):
    base = f"/v1/optimizations/{optimization_id}"
    metadata = await client.get(base)
    remote = str(metadata.get("state", "UNKNOWN")).upper()
    verdict = {
        "state": "unknown",
        "remote_state": remote,
        "optimization_verified": False,
        "completion_scope": "optimization",
        "scheduler_jobs": {},
        "metadata": metadata,
    }
    try:
        matching_request = validate_request("torchopt", metadata.get("request", {})) == request
    except ValueError:
        matching_request = False
    if metadata.get("optimization_id") != optimization_id or not matching_request:
        verdict["diagnostic"] = "status metadata does not match the submitted optimization request"
        return verdict
    if remote == "PREPARED" and request.get("prepare_only") and metadata.get("returncode") == 0:
        return {**verdict, "state": "prepared", "completion_scope": "preparation"}
    if remote == "SUBMISSION_FAILED":
        return {
            **verdict,
            "state": "submission_uncertain",
            "diagnostic": "launcher failed; some child jobs may still have been submitted; never POST again",
        }
    if remote == "FAILED":
        receipts = metadata.get("submissions", {})
        ids = set(receipt_job_ids(receipts))
        states = metadata.get("scheduler_jobs", {})
        terminal = {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
            "REVOKED",
        }
        if (
            ids
            and set(states) == ids
            and all(row.get("state") in terminal and row.get("exit_code") is not None for row in states.values())
        ):
            return {
                **verdict,
                "state": "failed",
                "retry_safe": True,
                "scheduler_jobs": states,
                "diagnostic": "All optimization child jobs are terminal; this attempt failed",
            }
        return {
            **verdict,
            "state": "optimization_failed",
            "diagnostic": "a child job failed; inspect all receipts before recovery",
        }
    if remote in {"PREPARING", "SUBMITTED", "RUNNING", "COLLECTING"}:
        return {**verdict, "state": "collecting" if remote == "COLLECTING" else "running"}
    if remote != "SUCCEEDED" or metadata.get("returncode") != 0:
        return {**verdict, "diagnostic": "waiting for API scheduler verification"}
    receipts = metadata.get("submissions", {})
    if not isinstance(receipts, dict) or not receipts or any(not receipt_job_ids(row) for row in receipts.values()):
        return {**verdict, "diagnostic": "no recognized scheduler receipts"}
    job_ids = set(receipt_job_ids(receipts))
    states = metadata.get("scheduler_jobs", {})
    if not isinstance(states, dict) or set(states) != job_ids or not job_ids:
        return {**verdict, "diagnostic": "scheduler evidence does not cover the receipts"}
    verdict["scheduler_jobs"] = states
    if any(
        not isinstance(row, dict) or row.get("state") != "COMPLETED" or row.get("exit_code") not in {"0", "0:0"}
        for row in states.values()
    ):
        return {**verdict, "diagnostic": "scheduler completion or exit code is unverified"}
    if not metadata.get("runs"):
        return {**verdict, "state": "collecting", "diagnostic": "waiting for native summaries"}
    artifacts = await client.get(base + "/results")
    if artifacts.get("optimization_id") != optimization_id:
        return {**verdict, "diagnostic": "results do not match optimization_id"}
    for field in ("results", "structures"):
        values = artifacts.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(v, str) or not v.startswith("/") for v in values)
        ):
            return {**verdict, "state": "collecting", "diagnostic": "waiting for final PDBs and result tables"}
    return {**verdict, "state": "succeeded", "optimization_verified": True, "artifacts": artifacts}


class TorchOptimizationService(SlurmService):
    async def run_optimization(self, key, payload, *, attach_id=None):
        jobs = self.checkpoint.data["jobs"]
        signature = fingerprint({"url": self.config.url, "route": "/v1/optimizations", "payload": payload})
        job = jobs.get(key)
        if attach_id and job is None:
            raise ValueError("attached optimization key must already exist in the checkpoint")
        if job is not None and job["request_hash"] != signature:
            raise ValueError(f"job request changed: {key}")
        if job and job["state"] == "completed":
            return job["results"]
        if job is None:
            await self.ensure_ready()
            job = jobs[key] = {
                "request_hash": signature,
                "state": "submission_uncertain",
                "api_url": str(self.client.base_url),
                "api_allocation": api_startup.allocation_record(self.config),
                "submission_key": uuid4().hex,
            }
            self.checkpoint.save()
            try:
                response = await self.post(
                    "/v1/optimizations", json=payload, headers={"Idempotency-Key": job["submission_key"]}
                )
                response.raise_for_status()
                identifier = response.json()["optimization_id"]
                if not re.fullmatch(r"[0-9a-f]{32}", identifier):
                    raise ValueError("invalid optimization_id")
            except Exception as exc:
                try:
                    recovered = await self.get(f"/v1/optimizations/by-key/{job['submission_key']}")
                    identifier = recovered["optimization_id"]
                    if (
                        not re.fullmatch(r"[0-9a-f]{32}", identifier)
                        or validate_request("torchopt", recovered["request"]) != payload
                    ):
                        raise ValueError("recovery response does not match request")
                except Exception:
                    raise SubmissionUncertainError(
                        f"{key}: optimization submission uncertain; reconcile existing task before resuming"
                    ) from exc
            job.update(id=identifier, state="submitted")
            self.checkpoint.save()
        elif job.get("api_url"):
            await self.restore_job_api(job)
        if attach_id:
            if not re.fullmatch(r"[0-9a-f]{32}", attach_id) or (job.get("id") and job["id"] != attach_id):
                raise ValueError("invalid or conflicting attached optimization ID")
            job.update(id=attach_id)
            self.checkpoint.save()
        if not job.get("id"):
            try:
                recovered = await self.get(f"/v1/optimizations/by-key/{job['submission_key']}")
                if validate_request("torchopt", recovered["request"]) == payload and re.fullmatch(
                    r"[0-9a-f]{32}", recovered["optimization_id"]
                ):
                    job.update(id=recovered["optimization_id"], state="submitted")
                    self.checkpoint.save()
            except (ValueError, KeyError):
                pass
        if not job.get("id"):
            raise SubmissionUncertainError(f"{key}: reconcile uncertain optimization using --attach-job {key}=ID")
        if job["state"] == "completed":
            return job["results"]
        deadline = asyncio.get_running_loop().time() + self.config.job_timeout
        while asyncio.get_running_loop().time() < deadline:
            verdict = await inspect_optimization(self, job["id"], payload)
            job.update(state="waiting", optimization_status=verdict)
            self.checkpoint.save()
            if verdict["state"] == "failed" and verdict.get("retry_safe"):
                job.update(state="failed", retry_safe=True)
                self.checkpoint.save()
                raise RemoteJobFailedError(f"{key}: optimization failed with all child jobs terminal; resume this run")
            if verdict["state"] == "succeeded":
                job.update(state="completed", results=verdict["artifacts"])
                self.checkpoint.save()
                return job["results"]
            if verdict["state"] in {"submission_uncertain", "optimization_failed"}:
                raise SubmissionUncertainError(f"{key}: {verdict['diagnostic']}; preserve submitted child jobs")
            await asyncio.sleep(self.config.poll_interval)
        raise TimeoutError(f"{key}: optimization still pending; resume the recorded optimization_id, do not resubmit")
