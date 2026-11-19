"""Configured scientific models and durable, non-blocking external job control."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

import httpx
from pydantic import Field, model_validator

from opendde_harness.plugin.protein_design.core.contracts import ContractModel
from opendde_harness.plugin.protein_design.core.model_requests import (
    REQUEST_TYPES,
    ROSETTA_OPERATIONS,
    AF3ScoreRequest,
    validate_request,
    validate_request_defaults,
)
from opendde_harness.plugin.protein_design.initial_round.api_startup import allocation_record, allocation_was_released
from opendde_harness.plugin.protein_design.initial_round.config import Service
from opendde_harness.plugin.protein_design.initial_round.services import ServiceConnection
from opendde_harness.plugin.protein_design.initial_round.storage import fingerprint, response_state, select, write_json
from opendde_harness.plugin.protein_design.initial_round.torchopt import inspect_optimization
from opendde_harness.utils.portable_lock import file_lock


class ModelRegistration(ContractModel):
    kind: Literal["ppiflow", "torchcraft", "af3score", "torchscore", "torchopt", "torchfold", "rosetta"]
    service: Service
    max_samples: int = Field(default=20, ge=1)
    allowed_input_roots: list[str] = Field(default_factory=list)
    request_defaults: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def defaults_contract(self):
        validate_request_defaults(self.kind, self.request_defaults)
        return self


class ProteinModelRegistry:
    def __init__(
        self,
        registrations: dict,
        root: Path,
        *,
        transport=None,
        config_path: Path | None = None,
        auto_monitor: bool = False,
        workflow_settings: dict | None = None,
    ):
        self.models = {}
        for name, value in registrations.items():
            self.valid_id(name)
            self.models[name] = ModelRegistration.model_validate(value)
        self.root = root
        self.transport = transport
        self.config_path = config_path
        self.auto_monitor = auto_monitor
        self.workflow_settings = workflow_settings or {}

    @staticmethod
    def valid_id(value: str):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
            raise ValueError("model/request IDs must contain only letters, digits, underscores and hyphens")

    def model(self, name: str) -> ModelRegistration:
        if name not in self.models:
            raise ValueError(f"model {name!r} is not registered; call protein_model_list")
        return self.models[name]

    def request_schema(self, name):
        model = self.model(name)
        schema = REQUEST_TYPES[model.kind].model_json_schema()
        schema["required"] = [key for key in schema.get("required", []) if key not in model.request_defaults]
        for key, value in model.request_defaults.items():
            schema["properties"][key]["default"] = value
        return schema

    def catalogue(self, *, limit=20, offset=0):
        records = self.recent_requests(limit=None)
        return {
            "jobs_dir": str(self.root.resolve()),
            "workflow_defaults_configured": bool(self.workflow_settings.get("workflow_defaults")),
            "deployment_config_source": self.workflow_settings.get("deployment_config"),
            "design_profiles": sorted(self.workflow_settings.get("design_profiles", {})),
            "compact_task_tool": "protein_design_start",
            "known_job_directories": [str(root) for root in self.known_directories()],
            "recent_requests": records[offset : offset + limit],
            "total_requests": len(records),
            "offset": offset,
            "active_config_path": str(self.config_path) if self.config_path else None,
            "available_model_kinds": sorted(REQUEST_TYPES),
            "unconfigured_model_kinds": sorted(set(REQUEST_TYPES) - {model.kind for model in self.models.values()}),
            "configuration_hint": "An empty models list means tool code is installed but no API services are configured. Register sbatch-managed services in this active config using protein-design models register, then restart Harness. Every scientific request requires sbatch_script; direct/local HTTP is disabled.",
            "models": [
                {
                    "model": name,
                    "kind": model.kind,
                    "configured": True,
                    "execution": "sbatch",
                    "sbatch_configured": bool(model.service.sbatch_script),
                    "inference_verified": False,
                    "max_samples": model.max_samples,
                    "service": model.service.model_dump(mode="json", exclude_none=True, exclude={"environment"}),
                    "environment_keys": sorted(model.service.environment),
                    "request_defaults": model.request_defaults,
                    "operations": list(ROSETTA_OPERATIONS) if model.kind == "rosetta" else [],
                    "request_schema": self.request_schema(name),
                    "cancel_supported": model.kind == "af3score",
                    "notes": (
                        "AF3Score prepared mode returns raw outputs without automatic per-PDB metrics."
                        if model.kind == "af3score"
                        else "TorchScore scoring is independent of AF3Score; supports pdb/metrics, seed=10, one sample."
                        if model.kind == "torchscore"
                        else "Torch Optimization runs /v1/optimizations jobs; SUBMITTED is not GPU completion."
                        if model.kind == "torchopt"
                        else "Returns a remote job immediately; use status/results tools later."
                    ),
                }
                for name, model in sorted(self.models.items())
            ],
        }

    def client(self, model: ModelRegistration, record: dict | None = None):
        client = ServiceConnection(model.service, transport=self.transport)
        if record is not None:
            if not record.get("api_url"):
                raise ValueError("request has no saved sbatch API endpoint; reconcile it before querying")
            allocation = record.get("api_allocation")
            if not allocation_was_released(allocation):
                client.attach_api(record["api_url"], allocation)
            if allocation:
                client.allocations.append(allocation)
        return client

    async def catalogue_with_health(self, *, limit=20, offset=0):
        # Discovery must never probe a placeholder/local port or queue an allocation.
        catalogue = self.catalogue(limit=limit, offset=offset)
        for item in catalogue["models"]:
            item.update(
                reachable=None,
                healthy=None,
                startup="deferred_until_submission",
                sbatch_configured=bool(self.model(item["model"]).service.sbatch_script),
            )
        return catalogue

    def directory_index(self):
        return self.config_path.parent / "protein_model_job_directories.json" if self.config_path else None

    def remember_directory(self):
        path = self.directory_index()
        if path is None:
            return
        with file_lock(path.with_suffix(".lock")):
            directories = json.loads(path.read_text()) if path.is_file() else []
            write_json(path, sorted(set(directories) | {str(self.root.resolve())}))

    def known_directories(self):
        path = self.directory_index()
        directories = json.loads(path.read_text()) if path and path.is_file() else []
        return sorted({self.root.resolve(), *(Path(item) for item in directories)}, key=str)

    def for_request(self, request_id):
        self.valid_id(request_id)
        matches = [root for root in self.known_directories() if (root / f"{request_id}.json").is_file()]
        if len(matches) > 1:
            raise ValueError(
                "request_id exists in multiple jobs_dir entries; select the original model_jobs_dir from protein_model_list"
            )
        if not matches or matches[0] == self.root.resolve():
            return self
        return ProteinModelRegistry(
            {name: item.model_dump(mode="json") for name, item in self.models.items()},
            matches[0],
            transport=self.transport,
            config_path=self.config_path,
            auto_monitor=self.auto_monitor,
        )

    def recent_requests(self, limit=20):
        records = []
        for root in self.known_directories():
            for path in root.glob("*.json"):
                try:
                    value = json.loads(path.read_text())
                    if value.get("request_id") != path.stem or "registration_hash" not in value:
                        continue
                    records.append(
                        (
                            path.stat().st_mtime,
                            {**self.public_record(value), "request_record": str(path.resolve()), "jobs_dir": str(root)},
                        )
                    )
                except (OSError, ValueError, AttributeError):
                    continue
        return [value for _, value in sorted(records, key=lambda item: item[0], reverse=True)[:limit]]

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
            raise ValueError(
                f"unknown local request_id {request_id!r} in {self.root.resolve()}; "
                "use protein_model_list to find recent_requests and jobs_dir. "
                "An API remote ID or an initial-round worker job ID is not a standalone model request_id."
            )
        record = json.loads(path.read_text())
        record.setdefault("request_record", str(path.resolve()))
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
        if kind == "torchopt":
            return "/v1/optimizations", "/v1/optimizations", "optimization_id"
        return "/v1/tasks", "/v1/tasks", "task_id"

    @staticmethod
    def validate_paths(model, payload):
        for index, value in enumerate(payload.get("pdb_paths", [])):
            if not value.endswith(".pdb"):
                raise ValueError("TorchOptimization inputs must be PDB files")
            ProteinModelRegistry.validate_paths(model, {"input_pdb": value})
        for key, value in payload.items():
            if key not in {
                "config",
                "model_weights",
                "output_dir",
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
                "source",
                "export",
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
                "submission_diagnostic",
                "request_record",
                "cancel_state",
            )
            if key in record
        }

    async def submit(self, name: str, request_id: str, request: dict, operation: str | None = None):
        model = self.model(name)
        payload = validate_request(model.kind, {**model.request_defaults, **request}, operation)
        self.validate_paths(model, payload)
        for field in (
            "num_samples",
            "samples_per_target",
            "num_designs",
            "num_diffusion_samples",
            "nstruct",
            "replicates",
            "full_jobs",
            "full_gpus",
            "mcmc_proposals",
        ):
            if payload.get(field, 1) > model.max_samples:
                raise ValueError(f"{field} exceeds configured max_samples={model.max_samples}")
        if len(payload.get("pdb_paths", [])) > model.max_samples:
            raise ValueError("pdb_paths exceeds configured max_samples")
        self.remember_directory()
        route, status_base, id_field = self.routes(model.kind, operation)
        signature = fingerprint({"model": name, "operation": operation, "payload": payload})
        path, lock = self.paths(request_id)
        with file_lock(lock, blocking=False):
            startup_attempts = []
            if path.exists():
                record, _ = self.read(request_id)
                if record["request_hash"] != signature:
                    raise ValueError("request_id already belongs to a different model/request; use a distinct ID")
                safe_startup = record["state"] == "configuration_failed" or (
                    record["state"] == "api_start_failed" and record.get("api_start_retry_safe") is True
                )
                if not safe_startup:
                    # Including uncertain submissions: never POST again on retry.
                    return self.public_record(record)
                startup_attempts = record.get("startup_attempts", []) + [
                    {k: record[k] for k in ("state", "api_allocation", "api_cleanup", "error_type") if k in record}
                ]
            record = {
                "request_id": request_id,
                "startup_attempts": startup_attempts,
                "request_record": str(path.resolve()),
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
                else model.kind == "torchscore",
                **({"submission_key": uuid4().hex} if model.kind == "torchopt" else {}),
            }
            record["state"] = "starting_api"
            write_json(path, record)
            try:
                client = self.client(model)
            except Exception as exc:
                record.update(state="configuration_failed", error_type=type(exc).__name__)
                write_json(path, record)
                raise
            try:
                await client.ensure_ready()
            except BaseException:
                record["state"] = "api_start_failed"
                state_file = client.config.sbatch_state_file
                record["api_start_retry_safe"] = state_file is None or not state_file.is_file()
                if client.allocations:
                    record.update(api_allocation=client.allocations[-1], api_cleanup=await client.release_allocations())
                    record["api_start_retry_safe"] = bool(record["api_cleanup"]) and all(
                        item.get("state") in {"cancel_requested", "not_in_queue"} for item in record["api_cleanup"]
                    )
                write_json(path, record)
                await client.close()
                raise
            submission_stage = "record_allocation"
            try:
                record["state"] = "submission_uncertain"
                record["api_url"] = str(client.client.base_url)
                record["api_allocation"] = allocation_record(client.config)
                write_json(path, record)
                submission_stage = "post"
                response = await client.post(
                    route.lstrip("/"),
                    json=payload,
                    **({"headers": {"Idempotency-Key": record["submission_key"]}} if model.kind == "torchopt" else {}),
                )
                response.raise_for_status()
                submission_stage = "decode_response"
                submitted = response.json()
                submission_stage = "read_remote_id"
                remote_id = submitted[id_field]
                submission_stage = "validate_remote_id"
                self.valid_id(remote_id)
                record.update(state="submitted", remote_id=remote_id, slurm_job_id=submitted.get("slurm_job_id"))
                write_json(path, record)
            except Exception as exc:
                # Keep protocol evidence, never arbitrary response bodies or exception text.
                diagnostic = {
                    "method": "POST",
                    "endpoint": route,
                    "stage": submission_stage,
                    "error_type": type(exc).__name__,
                    "expected_id_field": id_field,
                }
                if isinstance(exc, httpx.HTTPStatusError):
                    diagnostic["http_status"] = exc.response.status_code
                if model.kind == "torchopt":
                    try:
                        recovered = await client.get(f"/v1/optimizations/by-key/{record['submission_key']}")
                        if validate_request("torchopt", recovered["request"]) != payload:
                            raise ValueError("optimization recovery does not match request")
                        identifier = recovered["optimization_id"]
                        if not re.fullmatch(r"[0-9a-f]{32}", identifier):
                            raise ValueError("invalid recovered optimization ID")
                        record.update(state="submitted", remote_id=identifier)
                        write_json(path, record)
                        return self.public_record(record)
                    except Exception:
                        pass
                record["submission_diagnostic"] = diagnostic
                write_json(path, record)
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
        if record.get("terminal_status"):
            return {
                **record["terminal_status"],
                **({"api_cleanup": record["api_cleanup"]} if "api_cleanup" in record else {}),
            }
        if not record.get("remote_id"):
            if model.kind == "torchopt" and record.get("submission_key"):
                client = self.client(model, record)
                try:
                    recovered = await client.get(f"/v1/optimizations/by-key/{record['submission_key']}")
                    if validate_request("torchopt", recovered["request"]) == record["request"] and re.fullmatch(
                        r"[0-9a-f]{32}", recovered["optimization_id"]
                    ):
                        record.update(remote_id=recovered["optimization_id"], state="submitted")
                        write_json(self.paths(request_id)[0], record)
                except (httpx.HTTPError, ValueError, KeyError):
                    pass
                finally:
                    await client.close()
        if not record.get("remote_id"):
            return {
                **self.public_record(record),
                "error": "No remote ID was received; reconcile this request with the service operator before resubmitting.",
            }
        if model.kind == "torchopt":
            client = self.client(model, record)
            try:
                verdict = await inspect_optimization(client, record["remote_id"], record["request"])
            finally:
                await client.close()
            return {
                **self.public_record(record),
                **{k: v for k, v in verdict.items() if k not in {"metadata", "artifacts"}},
            }
        client = self.client(model, record)
        try:
            status = await client.get(f"{record['status_base']}/{record['remote_id']}")
        finally:
            await client.close()
        raw_state = response_state(status, model.service.status_field)
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
        current = {**self.public_record(record), "state": state, "remote_state": raw_state, "exit_code": exit_code}
        return current

    async def status(self, request_id):
        path, lock = self.paths(request_id)
        with file_lock(lock, blocking=False):
            current = await self.refresh(request_id)
            record, model = self.read(request_id)
            record["state"] = current["state"]
            if current["state"] in {"succeeded", "prepared"}:
                # Cache results before releasing this request's private API allocation.
                await self._collect(path, record, model, current)
                return {**current, **({"api_cleanup": record["api_cleanup"]} if "api_cleanup" in record else {})}
            if current["state"] in {"failed", "cancelled"}:
                record["terminal_status"] = current
                await self._release(record, model)
                current.update({"api_cleanup": record["api_cleanup"]} if "api_cleanup" in record else {})
            write_json(path, record)
            return current

    async def _release(self, record, model):
        if not record.get("api_allocation"):
            return
        client = ServiceConnection(model.service, transport=self.transport)
        client.allocations.append(record["api_allocation"])
        try:
            record["api_cleanup"] = await client.release_allocations()
        finally:
            await client.close()

    async def _collect(self, path, record, model, current):
        artifact = self.root / f"{record['request_id']}.results.json"
        if record.get("results_path") and artifact.is_file():
            cached = json.loads(artifact.read_text())
            results, metrics = cached["results"], cached["metrics"]
        else:
            client = self.client(model, record)
            try:
                base = f"{record['status_base']}/{record['remote_id']}"
                results = await client.get(base + "/results")
                metrics = await client.get(base + "/metrics") if record["metrics_expected"] else None
            finally:
                await client.close()
            write_json(artifact, {"results": results, "metrics": metrics})
        record.update(state=current["state"], results_path=str(artifact), terminal_status=current)
        # Persist evidence before cancellation, including when the worker exits here.
        write_json(path, record)
        await self._release(record, model)
        write_json(path, record)
        return {
            **current,
            **({"api_cleanup": record["api_cleanup"]} if "api_cleanup" in record else {}),
            "results_available": True,
            "results_path": str(artifact),
            "results": results,
            "metrics": metrics,
            "interpretation": "Native optimization artifacts; scheduler completion verified. Re-evaluate structures independently."
            if model.kind == "torchopt" and current["state"] == "succeeded"
            else "Preparation only; no optimization jobs were submitted."
            if model.kind == "torchopt"
            else "AF3Score prepared tasks do not provide per-PDB metrics automatically."
            if model.kind == "af3score" and not record["metrics_expected"]
            else "Model outputs are predictions, not experimental validation.",
        }

    async def results(self, request_id):
        current = await self.status(request_id)
        path, lock = self.paths(request_id)
        with file_lock(lock, blocking=False):
            record, model = self.read(request_id)
            if current["state"] not in {"succeeded", "prepared"}:
                return {**current, "results_available": False}
            return await self._collect(path, record, model, current)

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
            client = self.client(model, record)
            try:
                await client.ensure_ready()
                record["cancel_state"] = "uncertain"
                write_json(path, record)
                response = await client.post(f"v1/tasks/{record['remote_id']}/cancel")
                response.raise_for_status()
                record["cancel_state"] = "requested"
                write_json(path, record)
            finally:
                await client.close()
            return self.public_record(record)
