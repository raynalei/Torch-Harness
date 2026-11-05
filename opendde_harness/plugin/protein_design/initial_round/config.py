"""Strict, separate configuration for initial rounds; no LLM or local GPU required."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ConfigDict, Field, model_validator

from opendde_harness.plugin.protein_design.core.contracts import ContractModel


class RoundModel(ContractModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PathMapping(RoundModel):
    local: Path
    remote: str

    @model_validator(mode="after")
    def absolute(self):
        if not self.local.is_absolute() or not Path(self.remote).is_absolute():
            raise ValueError("path mappings must be absolute")
        return self


class Service(RoundModel):
    url: str
    health_path: str = Field(default="/health", pattern=r"^/[A-Za-z0-9_./-]*$")
    token_env: str | None = None
    request_timeout: float = Field(default=30, gt=0)
    poll_interval: float = Field(default=10, gt=0)
    job_timeout: float = Field(default=86400, gt=0)
    status_field: str = "status"
    exit_code_field: str = "exit_code"
    require_exit_code: bool = True
    files_field: str = "files"
    metrics_field: str = "metrics"
    structure_field: str | None = None
    metric_fields: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_url(self):
        from urllib.parse import urlsplit

        parts = urlsplit(self.url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            raise ValueError("service URL must be HTTP(S), without embedded credentials")
        if parts.query or parts.fragment:
            raise ValueError("service URL must not contain query or fragment")
        return self


class Chains(RoundModel):
    target: list[str] = Field(min_length=1)
    binder: list[str] = Field(min_length=1)
    # Generated chain ID -> TorchFold input/output chain ID.
    refold_map: dict[str, str]

    @model_validator(mode="after")
    def complete_map(self):
        chains = self.target + self.binder
        if len(set(chains)) != len(chains) or any(not c for c in chains):
            raise ValueError("target/binder chain IDs must be distinct and nonempty")
        if set(self.refold_map) != set(chains) or len(set(self.refold_map.values())) != len(chains):
            raise ValueError("refold_map must map every target/binder chain exactly once")
        return self


class Generator(RoundModel):
    service: Service
    request: dict[str, Any]
    chains: Chains
    fold_template: Path
    # Optional explicit key in /results holding a candidate manifest.
    candidates_field: str = "candidates"
    structure_glob: str = "*.pdb"
    per_source_refold_limit: int = Field(default=5, ge=1, le=1000)


class Evaluation(RoundModel):
    af3score: Service
    torchfold: Service
    rosetta: Service
    af3score_options: dict[str, Any] = Field(default_factory=dict)
    torchfold_options: dict[str, Any] = Field(default_factory=lambda: {"run_data_pipeline": False})
    rosetta_options: dict[str, Any] = Field(default_factory=lambda: {"pack_input": False, "pack_separated": True})
    min_af3score_iptm: float = Field(default=0.5, ge=0, le=1)
    min_refold_iptm: float = Field(default=0.8, ge=0, le=1)
    min_dockq: float = Field(default=0.49, ge=0, le=1)
    dockq_python: Path
    dockq_timeout: float = Field(default=300, gt=0)
    # Explicit scalar field in TorchFold confidence JSON, selected by Service.metric_fields.
    fold_confidence_glob: str = "*summary_confidences*.json"
    rosetta_scores_glob: str = "*.sc"

    @model_validator(mode="after")
    def fixed_scoring_protocol(self):
        if self.af3score_options.get("num_samples", 1) != 1 or self.af3score_options.get("mode", "pdb") != "pdb":
            raise ValueError("initial round AF3Score uses mode=pdb and num_samples=1")
        if (
            "run_data_pipeline" in self.torchfold_options
            and type(self.torchfold_options["run_data_pipeline"]) is not bool
        ):
            raise ValueError("run_data_pipeline must be a boolean")
        return self


class InitialRoundConfig(RoundModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    work_dir: Path
    path_mappings: list[PathMapping] = Field(min_length=1)
    generators: dict[Literal["ppiflow", "torchcraft"], Generator]
    evaluation: Evaluation

    @model_validator(mode="after")
    def check_workflow(self):
        if set(self.generators) != {"ppiflow", "torchcraft"}:
            raise ValueError("initial round requires both ppiflow and torchcraft")
        pair = (self.generators["ppiflow"].request.get("mode"), self.generators["torchcraft"].request.get("mode"))
        if pair not in {("nanobody", "vhh"), ("binder", "mini_binder")}:
            raise ValueError("pair nanobody with vhh, or binder with mini_binder")
        if not self.work_dir.is_absolute():
            raise ValueError("work_dir must be absolute and on the shared filesystem")
        modes = {"ppiflow": {"nanobody", "binder"}, "torchcraft": {"vhh", "mini_binder"}}
        for name, generator in self.generators.items():
            if generator.request.get("mode") not in modes[name]:
                raise ValueError(f"{name} initial round supports {sorted(modes[name])}")
            required = {
                "nanobody": ("antigen_pdb", "antigen_chain", "framework_dir", "heavy_chain"),
                "binder": ("input_pdb", "target_chain", "binder_chain", "samples_min_length", "samples_max_length"),
                "vhh": ("pdb_path", "hotspot_indices", "target_full_seq", "framework_cif_path"),
                "mini_binder": ("pdb_path", "hotspot_indices", "target_full_seq", "design_length"),
            }[generator.request["mode"]]
            if any(generator.request.get(key) in (None, "") for key in required):
                raise ValueError(f"{name} {generator.request['mode']} requires {', '.join(required)}")
            if len(generator.chains.binder) != 1:
                raise ValueError("initial round VHH/minibinder modes require exactly one binder chain")
        if not math.isfinite(self.evaluation.dockq_timeout):
            raise ValueError("dockq_timeout must be finite")
        return self

    @classmethod
    def load(cls, path: Path):
        return cls.model_validate(yaml.safe_load(path.read_text()))
