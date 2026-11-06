"""Configured scientific models and durable, non-blocking external job control."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field

from opendde_harness.plugin.protein_design.core.contracts import ContractModel
from opendde_harness.plugin.protein_design.core.model_requests import (
    REQUEST_TYPES,
    ROSETTA_OPERATIONS,
    AF3ScoreRequest,
    validate_request,
)
from opendde_harness.plugin.protein_design.initial_round.config import Service
from opendde_harness.plugin.protein_design.initial_round.services import ServiceConnection
from opendde_harness.plugin.protein_design.initial_round.storage import fingerprint, select, write_json
from opendde_harness.utils.portable_lock import file_lock


class ModelRegistration(ContractModel):
    kind: Literal["ppiflow", "torchcraft", "af3score", "torchfold", "rosetta"]
    service: Service
    max_samples: int = Field(default=20, ge=1)
    allowed_input_roots: list[str] = Field(default_factory=list)


class ProteinModelRegistry:
    def __init__(self, registrations: dict, root: Path, *, transport=None):
        self.models = {}
        for name, value in registrations.items():
            self.valid_id(name)
            self.models[name] = ModelRegistration.model_validate(value)
        self.root = root
        self.transport = transport

    @staticmethod
    def valid_id(value: str):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
            raise ValueError("model/request IDs must contain only letters, digits, underscores and hyphens")

    def model(self, name: str) -> ModelRegistration:
        if name not in self.models:
            raise ValueError(f"model {name!r} is not registered; call protein_model_list")
        return self.models[name]

    def catalogue(self):
        return {
            "models": [
                {
                    "model": name,
                    "kind": model.kind,
                    "configured": True,
                    "inference_verified": False,
                    "max_samples": model.max_samples,
                    "operations": list(ROSETTA_OPERATIONS) if model.kind == "rosetta" else [],
                    "request_schema": REQUEST_TYPES[model.kind].model_json_schema(),
                    "cancel_supported": model.kind == "af3score",
                    "notes": (
                        "Pipeline-only AF3Score jobs produce prepared input artifacts, not metrics. Prepared mode returns raw outputs."
                        if model.kind == "af3score"
                        else "Returns a remote job immediately; use status/results tools later."
                    ),
                }
                for name, model in sorted(self.models.items())
            ]
        }

    def client(self, model: ModelRegistration):
        return ServiceConnection(model.service, transport=self.transport)

    async def catalogue_with_health(self):
        catalogue = self.catalogue()

        async def probe(item):
            model = self.model(item["model"])
            client = None
            try:
                client = self.client(model)
                health = await client.get(model.service.health_path)
                item["reachable"] = True
                item["healthy"] = (
                    isinstance(health, dict)
                    and health.get("ok") is not False
                    and str(health.get("status", "ok")).lower() not in {"error", "failed", "unhealthy"}
                )
            except Exception:
                item.update(reachable=False, healthy=False)
            finally:
                if client is not None:
                    await client.close()

        await asyncio.gather(*(probe(item) for item in catalogue["models"]))
        return catalogue

    @staticmethod
    def registration_identity(model):
        # Response profiles and rotated credentials can be corrected without
        # pointing an old job ID at a different model server.
        return fingerprint({"kind": model.kind, "url": model.service.url.rstrip("/")})

    def paths(self, request_id):
        self.valid_id(request_id)
        return self.root / f"{request_id}.json", self.root / f"{request_id}.lock"

    def read(self, request_id):
        path, _ = self.paths(request_id)
        if not path.is_file():
            raise ValueError(f"unknown request_id {request_id!r}")
        record = json.loads(path.read_text())
        model = self.model(record["model"])
        if record["registration_hash"] != self.registration_identity(model):
            raise ValueError(
                "model registration changed since submission; restore the original registration before querying this job"
            )
        return record, model

    @staticmethod
    def routes(kind, operation):
        if kind in {"ppiflow", "torchcraft"}:
            return "/v1/designs", "/v1/designs", "design_id"
        if kind == "torchfold":
            return "/v1/folds", "/v1/folds", "fold_id"
        if kind == "rosetta":
            return f"/v1/tools/{operation}", "/v1/tasks", "task_id"
        return "/v1/tasks", "/v1/tasks", "task_id"

    @staticmethod
    def validate_paths(model, payload):
        for key, value in payload.items():
            if key not in {
                "input_pdb_dir",
                "json_dir",
                "h5_dir",
                "af3score_output_dir",
                "antigen_pdb",
                "framework_dir",
                "input_pdb",
                "complex_pdb",
                "pdb_path",
                "target_full_seq",
                "framework_cif_path",
                "json_path",
            }:
                continue
            path = PurePosixPath(value)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{key} must be an absolute cluster path without '..'")
            if model.allowed_input_roots and not any(
                path.is_relative_to(PurePosixPath(root)) for root in model.allowed_input_roots
            ):
                raise ValueError(f"{key} is outside the registered input roots")

    @staticmethod
    def public_record(record):
        return {
            key: record[key]
            for key in (
                "request_id",
                "model",
                "kind",
                "state",
                "remote_id",
                "slurm_job_id",
                "metrics_expected",
                "cancel_state",
            )
            if key in record
        }

    async def submit(self, name: str, request_id: str, request: dict, operation: str | None = None):
        model = self.model(name)
        payload = validate_request(model.kind, request, operation)
        self.validate_paths(model, payload)
        for field in (
            "num_samples",
            "samples_per_target",
            "num_designs",
            "num_diffusion_samples",
            "nstruct",
            "replicates",
        ):
            if payload.get(field, 1) > model.max_samples:
                raise ValueError(f"{field} exceeds configured max_samples={model.max_samples}")
        route, status_base, id_field = self.routes(model.kind, operation)
        signature = fingerprint({"model": name, "operation": operation, "payload": payload})
        path, lock = self.paths(request_id)
        with file_lock(lock, blocking=False):
            if path.exists():
                record, _ = self.read(request_id)
                if record["request_hash"] != signature:
                    raise ValueError("request_id already belongs to a different model/request; use a distinct ID")
                # Including uncertain submissions: never POST again on retry.
                return self.public_record(record)
            record = {
                "request_id": request_id,
                "model": name,
                "kind": model.kind,
                "registration_hash": self.registration_identity(model),
                "request_hash": signature,
                "request": payload,
                "operation": operation,
                "status_base": status_base,
                "state": "submission_uncertain",
                "metrics_expected": AF3ScoreRequest.model_validate(payload).produces_metrics
                if model.kind == "af3score"
                else False,
            }
            client = self.client(model)
            try:
                write_json(path, record)
                response = await client.client.post(route.lstrip("/"), json=payload)
                response.raise_for_status()
                submitted = response.json()
                remote_id = submitted[id_field]
                self.valid_id(remote_id)
                record.update(state="submitted", remote_id=remote_id, slurm_job_id=submitted.get("slurm_job_id"))
                write_json(path, record)
            except Exception:
                # Do not expose headers, tokens or server-provided arbitrary text.
                return {
                    **self.public_record(record),
                    "error": "Submission outcome is uncertain. Inspect the service/Slurm before taking further action; do not submit again with a new request_id.",
                }
            finally:
                await client.close()
            return self.public_record(record)

    async def refresh(self, request_id):
        record, model = self.read(request_id)
        if not record.get("remote_id"):
            return {
                **self.public_record(record),
                "error": "No remote ID was received; reconcile this request with the service operator before resubmitting.",
            }
        client = self.client(model)
        try:
            status = await client.get(f"{record['status_base']}/{record['remote_id']}")
        finally:
            await client.close()
        raw = str(select(status, model.service.status_field, "UNKNOWN")).upper().split()
        raw_state = raw[0].rstrip("+") if raw else "UNKNOWN"
        exit_code = select(status, model.service.exit_code_field)
        if raw_state in {"COMPLETED", "SUCCEEDED", "SUCCESS"}:
            state = (
                "succeeded"
                if str(exit_code) in {"0", "0:0"} or (exit_code is None and not model.service.require_exit_code)
                else "unverified"
                if exit_code is None
                else "failed"
            )
        elif raw_state in {"CANCELLED", "CANCELED"}:
            state = "cancelled"
        elif raw_state in {
            "FAILED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
            "REVOKED",
        }:
            state = "failed"
        elif raw_state in {"PENDING", "QUEUED", "CONFIGURING"}:
            state = "queued"
        elif raw_state in {"RUNNING", "COMPLETING"}:
            state = "running"
        else:
            state = "unknown"
        return {**self.public_record(record), "state": state, "remote_state": raw_state, "exit_code": exit_code}

    async def status(self, request_id):
        path, lock = self.paths(request_id)
        with file_lock(lock, blocking=False):
            current = await self.refresh(request_id)
            record, _ = self.read(request_id)
            record["state"] = current["state"]
            write_json(path, record)
            return current

    async def results(self, request_id):
        path, lock = self.paths(request_id)
        with file_lock(lock, blocking=False):
            current = await self.refresh(request_id)
            if current["state"] != "succeeded":
                return {**current, "results_available": False}
            record, model = self.read(request_id)
            client = self.client(model)
            try:
                base = f"{record['status_base']}/{record['remote_id']}"
                results = await client.get(base + "/results")
                metrics = await client.get(base + "/metrics") if record["metrics_expected"] else None
            finally:
                await client.close()
            # Raw artifacts stay on disk and are untrusted data, not prompt instructions.
            artifact = self.root / f"{request_id}.results.json"
            write_json(artifact, {"results": results, "metrics": metrics})
            record.update(state="succeeded", results_path=str(artifact))
            write_json(path, record)
            return {
                **current,
                "results_available": True,
                "results_path": str(artifact),
                "results": results,
                "metrics": metrics,
                "interpretation": "AF3Score pipeline-only/prepared tasks do not provide per-PDB metrics automatically."
                if model.kind == "af3score" and not record["metrics_expected"]
                else "Model outputs are predictions, not experimental validation.",
            }

    async def cancel(self, request_id):
        path, lock = self.paths(request_id)
        with file_lock(lock, blocking=False):
            record, model = self.read(request_id)
            if model.kind != "af3score":
                raise ValueError("only the AF3Score README documents a cancellation endpoint")
            if not record.get("remote_id"):
                raise ValueError("cannot cancel an uncertain submission without a verified remote ID")
            if record.get("cancel_state"):
                return self.public_record(record)
            record["cancel_state"] = "uncertain"
            write_json(path, record)
            client = self.client(model)
            try:
                response = await client.client.post(f"v1/tasks/{record['remote_id']}/cancel")
                response.raise_for_status()
                record["cancel_state"] = "requested"
                write_json(path, record)
            finally:
                await client.close()
            return self.public_record(record)
