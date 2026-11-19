"""Validated requests for registered protein-model HTTP services."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictBool, TypeAdapter, model_validator

from opendde_harness.plugin.protein_design.core.contracts import ContractModel


class AF3ScoreControls(ContractModel):
    run_data_pipeline: StrictBool = False
    run_inference: StrictBool = True
    init_guess: StrictBool = True
    num_samples: int = Field(default=1, ge=1, le=20)
    num_workers: int = Field(default=4, ge=1)
    seed: int = Field(default=10, ge=0)
    write_fold_input_json_file: StrictBool = False


class AF3ScoreRequest(AF3ScoreControls):
    mode: Literal["pdb", "prepared", "metrics"] = "pdb"
    input_pdb_dir: str | None = None
    json_dir: str | None = None
    h5_dir: str | None = None
    af3score_output_dir: str | None = None

    @model_validator(mode="after")
    def task_mode(self):
        required = {
            "pdb": {"input_pdb_dir"},
            "prepared": {"json_dir", "h5_dir"},
            "metrics": {"input_pdb_dir", "af3score_output_dir"},
        }[self.mode]
        paths = {"input_pdb_dir", "json_dir", "h5_dir", "af3score_output_dir"}
        if any(not getattr(self, field) for field in required):
            raise ValueError(f"AF3Score {self.mode} requires {', '.join(sorted(required))}")
        if any(getattr(self, field) is not None for field in paths - required):
            raise ValueError("AF3Score input paths do not match the selected mode")
        # Every run writes the TorchFold fold input JSON so a refold stage can
        # reuse it; (true, true) therefore yields both AF3Score metrics and fold JSON.
        self.write_fold_input_json_file = True
        if self.mode == "metrics":
            # Only metrics extraction may skip both the pipeline and inference.
            self.run_data_pipeline = False
            self.run_inference = False
        elif self.run_inference is not True:
            # pdb/prepared are regular inference runs: the model must actually run.
            raise ValueError(
                "run_inference=true is required for pdb/prepared mode; "
                "run_data_pipeline=false with run_inference=false is only valid in metrics mode"
            )
        return self

    @property
    def produces_metrics(self) -> bool:
        return bool(
            (self.mode == "pdb" and self.run_inference) or (self.mode == "metrics" and self.af3score_output_dir)
        )


class TorchScoreControls(ContractModel):
    run_data_pipeline: StrictBool = False
    run_inference: StrictBool = True
    init_guess: StrictBool = True
    num_samples: Literal[1] = 1
    num_workers: int = Field(default=4, ge=1, le=64)
    seed: Literal[10] = 10
    write_fold_input_json_file: StrictBool = False

    @model_validator(mode="after")
    def supported_controls(self):
        if self.run_data_pipeline or not self.init_guess or self.write_fold_input_json_file:
            raise ValueError(
                "TorchScore requires run_data_pipeline=false, init_guess=true and write_fold_input_json_file=false"
            )
        return self


class TorchScoreRequest(TorchScoreControls):
    mode: Literal["pdb", "metrics"] = "pdb"
    input_pdb_dir: str
    af3score_output_dir: str | None = Field(
        default=None, description="Compatibility API field: existing TorchScore raw outputs, not AF3Score outputs."
    )

    @model_validator(mode="after")
    def scoring_mode(self):
        if self.mode == "pdb" and (not self.run_inference or self.af3score_output_dir is not None):
            raise ValueError("TorchScore pdb mode requires inference and no existing output directory")
        if self.mode == "metrics":
            if not self.af3score_output_dir:
                raise ValueError("TorchScore metrics mode requires af3score_output_dir containing TorchScore outputs")
            self.run_inference = False
        return self


class TorchOptimizationRequest(ContractModel):
    name: str = Field(default="target", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    pdb_paths: list[str] = Field(min_length=1)
    binder_type: Literal["mini-binder", "vhh"] = "mini-binder"
    method: Literal["mcmc", "ddg", "mcmc-ddg"] = "mcmc-ddg"
    binder_chain: str = Field(default="A", min_length=1, max_length=1)
    target_chain: str = Field(default="C", min_length=1, max_length=1)
    interface_cutoff: float = Field(default=5, gt=0)
    mcmc_rounds: int = Field(default=5, ge=1)
    mcmc_proposals: int = Field(default=5, ge=1)
    ddg_rounds: int = Field(default=5, ge=1)
    rescue_iptm_cutoff: float = Field(default=0.8, ge=0, le=1)
    rescue_bottom_percent: float = Field(default=25, ge=0, le=100)
    partition: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_,.-]+$")
    full_jobs: int = Field(default=1, ge=1)
    full_gpus: int = Field(default=1, ge=1)
    prepare_only: StrictBool = False

    @model_validator(mode="after")
    def distinct_chains(self):
        if self.binder_chain == self.target_chain:
            raise ValueError("binder_chain and target_chain must differ")
        return self


# Match the deployed PPIFlow DesignRequest model.
PPIFLOW_DEFAULT_SAMPLES = 5
_PPIFLOW_COMMON_FIELDS = {"mode", "name", "config", "model_weights", "output_dir", "samples_per_target"}
_PPIFLOW_ANTIGEN_FIELDS = {"antigen_pdb", "antigen_chain", "framework_dir", "heavy_chain"}
_PPIFLOW_PARTIAL_FIELDS = {
    "complex_pdb",
    "antigen_chain",
    "heavy_chain",
    "fixed_positions",
    "cdr_position",
    "start_t",
}
_PPIFLOW_REQUIRED_FIELDS = {
    "antibody": _PPIFLOW_ANTIGEN_FIELDS | {"light_chain", "cdr_length"},
    "nanobody": _PPIFLOW_ANTIGEN_FIELDS,
    "binder": {"input_pdb", "target_chain", "binder_chain", "samples_min_length", "samples_max_length"},
    "partial_antibody": _PPIFLOW_PARTIAL_FIELDS | {"light_chain"},
    "partial_nanobody": _PPIFLOW_PARTIAL_FIELDS,
}
_PPIFLOW_ALLOWED_FIELDS = {
    mode: required
    | _PPIFLOW_COMMON_FIELDS
    | {"specified_hotspots"}
    | ({"num_timesteps"} if mode != "binder" else set())
    | ({"cdr_length"} if mode == "nanobody" else set())
    for mode, required in _PPIFLOW_REQUIRED_FIELDS.items()
}


def ppiflow_defaults_for_mode(defaults: dict, mode: str | None) -> dict:
    allowed = _PPIFLOW_ALLOWED_FIELDS.get(mode)
    if allowed is None:
        return defaults
    return {key: value for key, value in defaults.items() if key in allowed}


class PPIFlowRequest(ContractModel):
    mode: Literal["antibody", "nanobody", "binder", "partial_antibody", "partial_nanobody"]
    name: str = Field(default="target", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    samples_per_target: int = Field(default=PPIFLOW_DEFAULT_SAMPLES, ge=1, le=1000)
    num_timesteps: int | None = Field(default=None, ge=1, le=2000)
    config: str | None = None
    model_weights: str | None = None
    output_dir: str | None = None
    antigen_pdb: str | None = None
    antigen_chain: str | None = None
    framework_dir: str | None = None
    heavy_chain: str | None = None
    light_chain: str | None = None
    cdr_length: str | None = None
    specified_hotspots: str | None = None
    input_pdb: str | None = None
    target_chain: str | None = None
    binder_chain: str | None = None
    samples_min_length: int | None = Field(default=None, ge=1)
    samples_max_length: int | None = Field(default=None, ge=1)
    complex_pdb: str | None = None
    fixed_positions: str | None = None
    cdr_position: str | None = None
    start_t: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def required_inputs(self):
        fields = self.model_dump(exclude_none=True).keys()
        unsupported = fields - _PPIFLOW_ALLOWED_FIELDS[self.mode]
        missing = _PPIFLOW_REQUIRED_FIELDS[self.mode] - fields
        if unsupported or missing:
            raise ValueError(
                f"PPIFlow {self.mode} unsupported fields: {sorted(unsupported)}; missing fields: {sorted(missing)}"
            )
        if self.samples_min_length and self.samples_max_length and self.samples_min_length > self.samples_max_length:
            raise ValueError("samples_min_length exceeds samples_max_length")
        return self


class TorchCraftRequest(ContractModel):
    mode: Literal["mini_binder", "vhh", "monomer_unconditional"] = Field(
        description="mini_binder/vhh require target inputs; monomer_unconditional requires json_path and design_length, with no target/hotspot/framework inputs."
    )
    name: str
    pdb_path: str | None = None
    hotspot_indices: str | None = None
    target_full_seq: str | None = Field(
        default=None,
        description="Optional absolute remote CSV path (target,real_sequence); target matches PDB filename stem. Omit to use the native PDB-sequence fallback.",
    )
    json_path: str | None = None
    design_length: int | None = Field(default=None, ge=1)
    framework_cif_path: str | None = None
    num_designs: int = Field(default=1, ge=1)
    seed: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def mode_inputs(self):
        if self.mode == "monomer_unconditional":
            if not self.json_path or self.design_length is None:
                raise ValueError("TorchCraft monomer_unconditional requires json_path and design_length")
            if any(
                value is not None
                for value in (self.pdb_path, self.hotspot_indices, self.target_full_seq, self.framework_cif_path)
            ):
                raise ValueError("TorchCraft monomer_unconditional uses json_path, not target/framework inputs")
            return self
        if not all((self.pdb_path, self.hotspot_indices)):
            raise ValueError(
                f"TorchCraft {self.mode} requires pdb_path and hotspot_indices; target_full_seq CSV is optional"
            )
        if self.json_path is not None:
            raise ValueError("json_path is currently documented for TorchCraft monomer_unconditional only")
        if self.mode == "vhh" and not self.framework_cif_path:
            raise ValueError("TorchCraft vhh requires framework_cif_path")
        if self.mode == "mini_binder" and self.design_length is None:
            raise ValueError("TorchCraft mini_binder requires design_length")
        return self


class TorchFoldRequest(ContractModel):
    name: str | None = None
    json_path: str
    run_data_pipeline: StrictBool = False
    num_diffusion_samples: int = Field(default=2, ge=1)


class RosettaRequest(ContractModel):
    input_pdb: str
    nstruct: int | None = Field(default=None, ge=1)
    protocol: str | None = None
    coordinate_stdev: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    chain_group_a: str | None = None
    chain_group_b: str | None = None
    pack_input: StrictBool | None = None
    pack_separated: StrictBool | None = None
    interface_radius: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    backbone: StrictBool | None = None
    repeats: int | None = Field(default=None, ge=1)
    mutation: str | None = None
    replicates: int | None = Field(default=None, ge=1)
    backrub_trials: int | None = Field(default=None, ge=1)


REQUEST_TYPES = {
    "af3score": AF3ScoreRequest,
    "torchscore": TorchScoreRequest,
    "torchopt": TorchOptimizationRequest,
    "ppiflow": PPIFlowRequest,
    "torchcraft": TorchCraftRequest,
    "torchfold": TorchFoldRequest,
    "rosetta": RosettaRequest,
}
ROSETTA_OPERATIONS = (
    "structure_relax",
    "constrained_relax",
    "interface_analyze",
    "structure_score",
    "interface_local_relax",
    "mutation_binding_ddg",
)


def validate_request(kind: str, request: dict, operation: str | None = None, *, omit_defaults=False) -> dict:
    value = REQUEST_TYPES[kind].model_validate(request)
    if kind == "rosetta":
        if operation not in ROSETTA_OPERATIONS:
            raise ValueError(f"Rosetta operation must be one of {ROSETTA_OPERATIONS}")
        if operation in {"interface_analyze", "interface_local_relax", "mutation_binding_ddg"}:
            if not value.chain_group_a or not value.chain_group_b:
                raise ValueError("Rosetta interface tools require both chain groups")
        if operation == "mutation_binding_ddg" and not value.mutation:
            raise ValueError("mutation_binding_ddg requires mutation")
    elif operation is not None:
        raise ValueError("operation is only used for Rosetta")
    return value.model_dump(exclude_none=True, exclude_unset=omit_defaults)


TARGET_REQUEST_FIELDS = frozenset(
    {
        "antigen_pdb",
        "input_pdb",
        "complex_pdb",
        "pdb_path",
        "pdb_paths",
        "json_path",
        "input_pdb_dir",
        "af3score_output_dir",
        "specified_hotspots",
        "hotspot_indices",
        "target_full_seq",
        "name",
    }
)


TASK_DESIGN_FIELDS = frozenset(
    {"framework_dir", "framework_cif_path", "fixed_positions", "cdr_position", "mask_residues"}
)


def validate_request_defaults(kind, defaults):
    """Validate provided fields without manufacturing missing scientific inputs."""
    fields = REQUEST_TYPES[kind].model_fields
    if set(defaults) - set(fields):
        raise ValueError(f"unknown request default fields for {kind}")
    if set(defaults) & (TARGET_REQUEST_FIELDS | TASK_DESIGN_FIELDS):
        raise ValueError(
            "target-specific and framework/residue inputs must be supplied by each task, not request_defaults"
        )
    for key, value in defaults.items():
        field = fields[key]
        annotation = Annotated[field.annotation, *field.metadata] if field.metadata else field.annotation
        TypeAdapter(annotation).validate_python(value)
    return defaults
