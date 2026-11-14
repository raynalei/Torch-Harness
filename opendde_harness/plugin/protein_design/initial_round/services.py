"""Adapters for the five supplied HTTP/Slurm protocols."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import httpx
from loguru import logger

from .api_startup import (
    allocation_record,
    allocation_status,
    allocation_was_released,
    api_path,
    ensure_api,
    release_allocation,
    remote_api_url,
    require_sbatch,
    validate_binding,
)
from .config import Service
from .storage import Checkpoint, fingerprint, select


class SubmissionUncertainError(RuntimeError):
    """A POST may have reached Slurm: never resubmit automatically."""


class RemoteJobFailedError(RuntimeError):
    pass


class ServiceConnection:
    """Bounded HTTP operations shared by foreground workflows and agent tools."""

    def __init__(self, config: Service, *, transport=None):
        self.config = config
        if config.sbatch_script is not None:
            # The configured path is a naming prefix, never a reusable allocation.
            prefix = config.sbatch_state_file or config.sbatch_script.parent / ".harness-api.json"
            self.config = config.model_copy(
                update={"sbatch_state_file": prefix.with_name(f"{prefix.stem}-{uuid4().hex}{prefix.suffix or '.json'}")}
            )
        self._api_ready = False
        self._bound_url = None
        self.allocations = []
        headers = {}
        if config.token_env:
            token = os.environ.get(config.token_env)
            if not token:
                raise ValueError(f"required credential variable is absent: {config.token_env}")
            headers["Authorization"] = f"Bearer {token}"
        self.client = httpx.AsyncClient(
            base_url=httpx.URL(config.url.rstrip("/") + "/").copy_with(host="unallocated-api.invalid"),
            timeout=config.request_timeout,
            headers=headers,
            transport=transport,
        )

    def attach_api(self, url: str, allocation: dict | None = None):
        """Reconnect to the endpoint recorded for an already submitted inference."""
        target = validate_binding(self.config, url, allocation)
        configured = httpx.URL(self.config.url.rstrip("/") + "/")
        if (
            target.scheme != configured.scheme
            or target.port != configured.port
            or target.path != configured.path
            or target.userinfo
            or target.query
            or target.fragment
        ):
            raise ValueError("recorded API endpoint does not match the configured protocol, port and path")
        self.client.base_url = target
        self.config = self.config.model_copy(update={"sbatch_state_file": Path(allocation["state_file"])})
        # The next request rechecks the recorded allocation through Slurm.
        self._bound_url = None
        self._api_ready = False

    async def close(self):
        await self.client.aclose()

    async def ensure_ready(self, *, start: bool = True):
        require_sbatch(self.config)
        if not self._api_ready:
            if start and self.config.sbatch_script is not None:
                logger.info("Starting this task's API allocation; state file: {}", self.config.sbatch_state_file)
            try:
                await ensure_api(self.config, self.client, start=start)
            finally:
                record = allocation_record(self.config)
                if record and record not in self.allocations:
                    self.allocations.append(record)
            self._api_ready = True
            self._bound_url = str(remote_api_url(self.client.base_url))

    def assert_bound(self):
        if not self._api_ready or self._bound_url != str(self.client.base_url):
            raise ValueError("API connection has no verified sbatch binding")
        remote_api_url(self.client.base_url)

    async def release_allocations(self):
        results = []
        for record in self.allocations:
            try:
                results.append(await release_allocation(self.config, record))
            except Exception as exc:
                results.append({"state": "cleanup_failed", "job_id": record["job_id"], "error": str(exc)})
        return results

    async def post(self, path: str, **kwargs):
        path = api_path(path)
        if kwargs.pop("follow_redirects", False):
            raise ValueError("API redirects cannot override the allocated node")
        await self.ensure_ready()
        self.assert_bound()
        return await self.client.post(path, follow_redirects=False, **kwargs)

    async def get(self, path: str, *, start_service: bool = True):
        path = api_path(path)
        await self.ensure_ready(start=start_service)
        self.assert_bound()
        response = await self.client.get(path, follow_redirects=False)
        response.raise_for_status()
        return response.json()


class SlurmService(ServiceConnection):
    def __init__(self, config: Service, checkpoint: Checkpoint, *, transport=None):
        super().__init__(config, transport=transport)
        self.checkpoint = checkpoint
        self._registration = config
        self._transport = transport
        self.allocations.extend(checkpoint.data.get("api_allocations", {}).get(config.url, []))

    async def ensure_ready(self, *, start: bool = True):
        try:
            await super().ensure_ready(start=start)
        finally:
            if self.allocations:
                self.checkpoint.data.setdefault("api_allocations", {})[self.config.url] = self.allocations
                self.checkpoint.save()

    async def restore_job_api(self, job):
        """Rehost persisted API metadata after its owned allocation has ended; never re-POST inference."""
        allocation = job.get("api_allocation")
        old_url = job.get("api_url")
        if not allocation or not old_url:
            raise ValueError("job has no saved API allocation; reconcile it before recovery")
        ended = allocation_was_released(allocation)
        if not ended:
            self.attach_api(old_url, allocation)
            state = json.loads(Path(allocation["state_file"]).read_text())
            phase, _ = await allocation_status(self.config, str(allocation["job_id"]), state)
            ended = phase in {
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
        if not ended:
            return
        owned = self.allocations[:]
        await self.client.aclose()
        ServiceConnection.__init__(self, self._registration, transport=self._transport)
        self.allocations.extend(owned)
        await self.ensure_ready()
        job.setdefault("api_history", []).append({"api_url": old_url, "api_allocation": allocation})
        job.update(api_url=str(self.client.base_url), api_allocation=allocation_record(self.config))
        self.checkpoint.save()

    async def reconcile(self, route: str, job: dict, payload: dict) -> bool:
        """Only trust services that echo the persisted key and exact request hash."""
        key = job.get("submission_key")
        if not key or route != "/v1/designs":
            return False
        try:
            value = await self.get(f"{route.lstrip('/')}/{key}")
        except (httpx.HTTPError, ValueError):
            return False
        if not isinstance(value, dict) or (
            value.get("submission_key") != key
            or value.get("design_id") != key
            or value.get("request_hash") != fingerprint(payload)
        ):
            return False
        job.update(id=key, state="submitted", submission=value)
        self.checkpoint.save()
        return True

    async def run(
        self,
        key: str,
        route: str,
        id_field: str,
        payload: dict,
        *,
        attach_id: str | None = None,
        status_base: str | None = None,
    ):
        signature = fingerprint({"url": self.config.url, "route": route, "payload": payload})
        jobs = self.checkpoint.data["jobs"]
        job = jobs.get(key)
        if job and job.get("api_allocation") and job["api_allocation"] not in self.allocations:
            self.allocations.append(job["api_allocation"])
        if attach_id:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", attach_id):
                raise ValueError("invalid attached job ID")
            if job is None:
                raise ValueError(f"--attach-job key must exist in checkpoint: {key}")
            if job.get("id") and job["id"] != attach_id:
                raise ValueError(f"cannot replace existing job ID for {key}")
        if job and job["request_hash"] != signature:
            raise ValueError(f"job request changed: {key}")
        if job and job["state"] == "completed":
            return job["results"]
        if job is None:
            await self.ensure_ready()
            job = jobs[key] = {
                "request_hash": signature,
                "state": "submission_uncertain",
                "submitted_at": time.time(),
                "api_url": str(self.client.base_url),
                "submission_key": uuid4().hex,
                "api_allocation": allocation_record(self.config),
            }
            self.checkpoint.save()
            # Checkpoint before POST: a crash in this window needs manual reconciliation.
            try:
                response = await self.post(route, json=payload, headers={"Idempotency-Key": job["submission_key"]})
                response.raise_for_status()
                job_id = response.json()[id_field]
                if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
                    raise ValueError("invalid task ID in submission response")
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                job["submission_error"] = detail
                self.checkpoint.save()
                if not await self.reconcile(route, job, payload):
                    raise SubmissionUncertainError(
                        f"{key}: submission response uncertain ({detail}); no automatic resubmission. "
                        f"Inspect server tasks and resume with --attach-job {key}=ID "
                        "using the API task ID, not the API allocation's Slurm job ID"
                    ) from exc
            else:
                job.update(id=job_id, state="submitted", submission=response.json())
                self.checkpoint.save()
        elif self.config.sbatch_script is not None and job.get("api_url"):
            await self.restore_job_api(job)
        if attach_id:
            job.update(id=attach_id)
            if job["state"] == "submission_uncertain":
                job["state"] = "submitted"
            self.checkpoint.save()
        if not job.get("id"):
            await self.reconcile(route, job, payload)
        if not job.get("id"):
            raise SubmissionUncertainError(f"{key}: reconcile uncertain submission using --attach-job {key}=ID")
        if job["state"] == "completed":
            return job["results"]
        if job["state"] == "failed":
            raise RemoteJobFailedError(
                f"{key}: persisted failed job {job['id']}; resume this run with retry_failed enabled"
            )
        if self.config.sbatch_script is not None and not job.get("api_url"):
            raise ValueError("older job has no recorded API endpoint; reconcile it before continuing polling")
        status_route = f"{status_base or route}/{quote(job['id'], safe='')}"
        deadline = asyncio.get_running_loop().time() + self.config.job_timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                status = await self.get(status_route)
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in {429, 500, 502, 503, 504}:
                    raise
                await asyncio.sleep(self.config.poll_interval)
                continue
            states = str(select(status, self.config.status_field, "UNKNOWN")).upper().split()
            state = states[0].rstrip("+") if states else "UNKNOWN"
            job.update(state="waiting", remote_status=state, status=status)
            self.checkpoint.save()
            if state == "SUBMISSION_UNCERTAIN":
                raise SubmissionUncertainError(
                    f"{key}: API task {job['id']} exists, but its Slurm submission is uncertain: "
                    f"{status.get('submission_error', 'inspect server metadata and scheduler')}; "
                    "reconcile the existing server task before resuming; do not POST again"
                )
            if state in {
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
            }:
                job.update(state="failed", retry_safe=True)
                self.checkpoint.save()
                raise RemoteJobFailedError(f"{key}: {state}; see {status_route}/logs on the service")
            if state in {"COMPLETED", "SUCCEEDED", "SUCCESS"}:
                exit_code = select(status, self.config.exit_code_field)
                if exit_code is None and self.config.require_exit_code:
                    raise RemoteJobFailedError(f"{key}: completed without exit_code; check the response profile")
                if exit_code is not None and str(exit_code) not in {"0", "0:0"}:
                    job.update(state="failed", retry_safe=True)
                    self.checkpoint.save()
                    raise RemoteJobFailedError(f"{key}: nonzero exit code {exit_code}")
                results = await self.get(status_route + "/results")
                job.update(state="completed", results=results)
                self.checkpoint.save()
                return results
            # UNKNOWN/accounting lag is neither success nor failure.
            await asyncio.sleep(self.config.poll_interval)
        raise TimeoutError(
            f"{key}: waiting timed out; remote job {job['id']} remains submitted; resume to keep polling"
        )
