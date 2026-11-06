"""Validated requests for registered protein-model HTTP services."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, StrictBool, model_validator

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
        if self.mode != "metrics" and not (self.run_data_pipeline or self.run_inference):
            raise ValueError("run_data_pipeline and run_inference cannot both be false outside metrics mode")
        if self.mode == "metrics":
            self.run_data_pipeline = False
            self.run_inference = False
        if self.mode != "metrics" and self.run_data_pipeline and not self.run_inference:
            self.write_fold_input_json_file = True
        return self

    @property
    def produces_metrics(self) -> bool:
        return self.mode == "metrics" or (self.mode == "pdb" and self.run_inference)


class PPIFlowRequest(ContractModel):
    mode: Literal["antibody", "nanobody", "binder", "partial_antibody", "partial_nanobody"]
    name: str
    samples_per_target: int = Field(default=1, ge=1, le=1000)
    num_timesteps: int | None = Field(default=None, ge=1)
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
    fixed_positions: Any | None = None
    cdr_position: Any | None = None
    start_t: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def required_inputs(self):
        required = {
            "antibody": ("antigen_pdb", "antigen_chain", "framework_dir", "heavy_chain", "light_chain", "cdr_length"),
            "nanobody": ("antigen_pdb", "antigen_chain", "framework_dir", "heavy_chain"),
            "binder": ("input_pdb", "target_chain", "binder_chain", "samples_min_length", "samples_max_length"),
            "partial_antibody": (
                "complex_pdb",
                "antigen_chain",
                "heavy_chain",
                "light_chain",
                "fixed_positions",
                "cdr_position",
                "start_t",
            ),
            "partial_nanobody": (
                "complex_pdb",
                "antigen_chain",
                "heavy_chain",
                "fixed_positions",
                "cdr_position",
                "start_t",
            ),
        }[self.mode]
        if any(getattr(self, key) is None or getattr(self, key) == "" for key in required):
            raise ValueError(f"PPIFlow {self.mode} requires {', '.join(required)}")
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
        default=None, description="Absolute remote path to the target sequence CSV, not a sequence string."
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
        if not all((self.pdb_path, self.hotspot_indices, self.target_full_seq)):
            raise ValueError(f"TorchCraft {self.mode} requires pdb_path, hotspot_indices and target_full_seq CSV path")
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


def validate_request(kind: str, request: dict, operation: str | None = None) -> dict:
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
    return value.model_dump(exclude_none=True)
