"""Strict, separate configuration for initial rounds; no LLM or local GPU required."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, ConfigDict, Field, model_serializer, model_validator

from opendde_harness.plugin.protein_design.core.contracts import ContractModel
from opendde_harness.plugin.protein_design.core.model_requests import (
    AF3ScoreControls,
    TorchScoreControls,
    validate_request,
)


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
    sbatch_script: Path | None = None
    sbatch_state_file: Path | None = None
    sbatch_resolve_host: Literal[True] = True
    startup_timeout: float = Field(default=1800, gt=0)
    startup_poll_interval: float = Field(default=5, gt=0)
    slurm_command_timeout: float = Field(default=30, gt=0)
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

        for path in (self.sbatch_script, self.sbatch_state_file):
            if path is not None and not path.is_absolute():
                raise ValueError("sbatch paths must be absolute paths on the Harness host")
        if self.sbatch_state_file is not None and self.sbatch_script is None:
            raise ValueError("sbatch_state_file requires sbatch_script")
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
    service: Service | None = None
    request: dict[str, Any]
    chains: Chains
    fold_template: Path | None = None
    fold_seeds: list[int] = Field(default_factory=lambda: [1], min_length=1)
    # Optional explicit key in /results holding a candidate manifest.
    candidates_field: str = "candidates"
    structure_glob: str = "*.pdb"
    per_source_refold_limit: int | None = Field(default=None, ge=1, le=1000)
    deduplicate_sequences: bool = True


class Evaluation(RoundModel):
    scoring_backend: Literal["af3score", "torchscore"] = "af3score"
    af3score: Service | None = None
    torchscore: Service | None = None
    torchfold: Service | None = None
    rosetta: Service | None = None
    torchopt: Service | None = None
    torchopt_options: dict[str, Any] = Field(default_factory=dict)
    af3score_options: dict[str, Any] = Field(default_factory=dict)
    torchscore_options: dict[str, Any] = Field(default_factory=dict)
    torchfold_options: dict[str, Any] = Field(default_factory=lambda: {"run_data_pipeline": False})
    rosetta_options: dict[str, Any] = Field(default_factory=lambda: {"pack_input": False, "pack_separated": True})
    min_af3score_iptm: float = Field(default=0.5, ge=0, le=1)
    min_torchscore_iptm: float = Field(default=0.5, ge=0, le=1)
    min_refold_iptm: float = Field(default=0.8, ge=0, le=1)
    min_dockq: float = Field(default=0.49, ge=0, le=1)
    dockq_python: Path
    dockq_timeout: float = Field(default=300, gt=0)
    # Explicit scalar field in TorchFold confidence JSON, selected by Service.metric_fields.
    fold_confidence_glob: str = "*summary_confidences*.json"
    rosetta_scores_glob: str = "*.sc"

    @property
    def score_options(self):
        return getattr(self, f"{self.scoring_backend}_options")

    @property
    def score_metric(self):
        return f"{self.scoring_backend}_iptm"

    @property
    def score_threshold(self):
        return getattr(self, f"min_{self.scoring_backend}_iptm")

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        value = handler(self)
        if self.torchopt is None and not self.torchopt_options:
            value.pop("torchopt", None)
            value.pop("torchopt_options", None)
        # Preserve the historical config hash for unchanged AF3Score runs.
        if (
            self.scoring_backend == "af3score"
            and self.torchscore is None
            and not self.torchscore_options
            and self.min_torchscore_iptm == 0.5
        ):
            for key in ("scoring_backend", "torchscore", "torchscore_options", "min_torchscore_iptm"):
                value.pop(key, None)
        return value

    @model_validator(mode="after")
    def fixed_scoring_protocol(self):
        if getattr(self, self.scoring_backend) is None:
            raise ValueError(f"evaluation.{self.scoring_backend} service is required for the selected scoring_backend")
        if self.score_options.get("num_samples", 1) != 1 or self.score_options.get("mode", "pdb") != "pdb":
            raise ValueError(f"initial round {self.scoring_backend} uses mode=pdb and num_samples=1")
        schema = AF3ScoreControls if self.scoring_backend == "af3score" else TorchScoreControls
        controls = schema.model_validate({k: v for k, v in self.score_options.items() if k != "mode"})
        if not controls.run_inference:
            raise ValueError(
                f"initial round requires {'AF3Score' if self.scoring_backend == 'af3score' else 'TorchScore'} run_inference=true to produce selection metrics; use protein_model_submit for pipeline-only tasks"
            )
        if (
            "run_data_pipeline" in self.torchfold_options
            and type(self.torchfold_options["run_data_pipeline"]) is not bool
        ):
            raise ValueError("run_data_pipeline must be a boolean")
        return self


class AutonomousRound(RoundModel):
    """Bounded follow-up experiments; thresholds are prototype defaults."""

    max_rounds: int = Field(default=3, ge=1, le=20)
    max_samples: int = Field(default=100, ge=1, le=10000)
    batch_size: int = Field(default=5, ge=1, le=1000)
    max_batch_size: int = Field(default=20, ge=1, le=1000)
    patience: int = Field(default=2, ge=1)
    rescue_limit: int = Field(default=3, ge=0, le=100)
    min_score: float = Field(default=0.8, ge=0, le=1, validation_alias=AliasChoices("min_score", "min_af3score"))
    min_dockq: float = Field(default=0.49, ge=0, le=1)
    contact_distance: float = Field(default=5.0, gt=0)
    framework_rmsd: float = Field(default=2.0, gt=0)
    region_rmsd: float = Field(default=3.0, gt=0)
    decision_policy: Literal["agent", "rules"] = "agent"
    decision_timeout: float = Field(default=120, gt=0)


class InitialRoundConfig(RoundModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    work_dir: Path
    path_mappings: list[PathMapping] = Field(min_length=1)
    generators: dict[Literal["ppiflow", "torchcraft"], Generator]
    evaluation: Evaluation
    autonomous: AutonomousRound | None = None

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        value = handler(self)
        if self.autonomous is None:
            value.pop("autonomous", None)
        return value

    @model_validator(mode="after")
    def check_workflow(self):
        if self.evaluation.torchfold is None:
            raise ValueError("torchfold service must be configured or uniquely registered")
        if any(g.service is None for g in self.generators.values()):
            raise ValueError("generator services must be configured or uniquely registered")
        if self.autonomous is not None:
            if self.evaluation.rosetta is None:
                raise ValueError("autonomous rounds require a scoring backend and Rosetta for the ranking formula")
            for source, generator in self.generators.items():
                validate_request(source, {"name": self.name, **generator.request})
            initial_samples = sum(
                int(g.request.get("samples_per_target" if source == "ppiflow" else "num_designs", 1))
                for source, g in self.generators.items()
            )
            if initial_samples > self.autonomous.max_samples:
                raise ValueError("initial generator requests exceed autonomous.max_samples")
        if not self.generators or "ppiflow" not in self.generators:
            raise ValueError("initial round requires ppiflow")
        if "torchcraft" in self.generators:
            pair = (self.generators["ppiflow"].request.get("mode"), self.generators["torchcraft"].request.get("mode"))
            if pair not in {("nanobody", "vhh"), ("binder", "mini_binder")}:
                raise ValueError("pair nanobody with vhh, or binder with mini_binder")
        if not self.work_dir.is_absolute():
            raise ValueError("work_dir must be absolute and on the shared filesystem")
        modes = {"ppiflow": {"nanobody", "binder", "antibody"}, "torchcraft": {"vhh", "mini_binder"}}
        for name, generator in self.generators.items():
            if any(type(seed) is not int or seed < 0 for seed in generator.fold_seeds):
                raise ValueError("fold_seeds must contain nonnegative integers")
            if generator.request.get("mode") not in modes[name]:
                raise ValueError(f"{name} initial round supports {sorted(modes[name])}")
            required = {
                "antibody": (
                    "antigen_pdb",
                    "antigen_chain",
                    "framework_dir",
                    "heavy_chain",
                    "light_chain",
                    "cdr_length",
                ),
                "nanobody": ("antigen_pdb", "antigen_chain", "framework_dir", "heavy_chain"),
                "binder": ("input_pdb", "target_chain", "binder_chain", "samples_min_length", "samples_max_length"),
                "vhh": ("pdb_path", "hotspot_indices", "target_full_seq", "framework_cif_path"),
                "mini_binder": ("pdb_path", "hotspot_indices", "target_full_seq", "design_length"),
            }[generator.request["mode"]]
            if any(generator.request.get(key) in (None, "") for key in required):
                raise ValueError(f"{name} {generator.request['mode']} requires {', '.join(required)}")
            if generator.request["mode"] == "antibody":
                # Use the same request contract as the public model tools.
                validate_request("ppiflow", generator.request)
                heavy = generator.request["heavy_chain"]
                light = generator.request["light_chain"]
                if heavy == light or set(generator.chains.binder) != {heavy, light}:
                    raise ValueError("antibody binder chains must match distinct heavy_chain and light_chain")
                if generator.request["antigen_chain"] not in generator.chains.target:
                    raise ValueError("antibody antigen_chain must be included in target chains")
            elif len(generator.chains.binder) != 1:
                raise ValueError("initial round VHH/minibinder modes require exactly one binder chain")
        if not math.isfinite(self.evaluation.dockq_timeout):
            raise ValueError("dockq_timeout must be finite")
        return self

    @classmethod
    def load(cls, path: Path):
        from opendde_harness.plugin.protein_design.initial_round.automation import prepare_run

        return prepare_run(path)
