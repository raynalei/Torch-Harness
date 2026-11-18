"""Default Torch-Harness workflow and evidence tools; no compute-service client."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from opendde_harness.agent.tools.base import Tool, ToolResult
from opendde_harness.plugin.context import PluginContext

from ..initial_round.automation import list_rounds, prepare_run, resume_round, round_status, start_round
from ..initial_round.config import InitialRoundConfig
from ..initial_round.task_inputs import DesignTask, prepare_design_task
from ..scientific_agent.backends.analysis import analyze_evolution, analyze_structure, apply_mutations, rank_candidates
from ..scientific_agent.backends.developability import assess_developability
from ..scientific_agent.backends.public_lookup import lookup_public


def prepare_workflow(value, *, registrations=None, pdb_path=None, name=None, plugin_config=None):
    if isinstance(value, (str, Path)):
        value = yaml.safe_load(Path(value).read_text())
    if not isinstance(value, dict):
        raise ValueError("workflow configuration must be an object")
    value = dict(value)
    value["autonomous"] = dict(value.get("autonomous") or {})
    if value["autonomous"].get("decision_policy", "planner") == "planner":
        value["autonomous"].setdefault("planner_layout", "two_stage")
    return prepare_run(value, registrations=registrations, pdb_path=pdb_path, name=name, plugin_config=plugin_config)


class EvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["developability", "structure", "epitope", "evolution", "mutations", "rank", "lookup"]
    source: Literal["pdb", "uniprot", "pubmed"] | None = None
    identifier: str | None = None
    sequences: dict[str, str] = Field(default_factory=dict)
    structure_path: Path | None = None
    binder_chains: list[str] = Field(default_factory=list)
    target_chains: list[str] = Field(default_factory=list)
    hotspots: list[dict] = Field(default_factory=list)
    candidates: list[dict] = Field(default_factory=list)
    mutations: list[dict] = Field(default_factory=list)
    fixed_positions: dict[str, list[int]] = Field(default_factory=dict)


def evidence(request: EvidenceRequest):
    if request.action == "lookup":
        if request.source is None or request.identifier is None:
            raise ValueError("source and identifier are required")
        return lookup_public(request.source, request.identifier)
    if request.action == "rank":
        return rank_candidates(request.candidates)
    if request.action == "developability":
        return assess_developability(
            request.sequences, structure_path=request.structure_path, binder_chains=request.binder_chains
        )
    if request.action in {"structure", "epitope"}:
        if request.structure_path is None:
            raise ValueError("structure_path is required")
        return analyze_structure(
            request.structure_path, request.binder_chains, request.target_chains, hotspots=request.hotspots
        )
    if request.action == "evolution":
        return analyze_evolution(request.candidates)
    return apply_mutations(request.sequences, request.mutations, request.fixed_positions)


class AnalysisTool(Tool):
    name = "protein_design_analyze"
    description = "Run native structure/epitope/lineage/mutation evidence or developability (chemical PTM motifs, PSH/PNC/PSC descriptors and EpiMatrix availability). No OpenDDE compute service. Missing licensed immunogenicity remains uncomputed; use explicit chain IDs and PDB residue IDs for hotspots."

    @property
    def parameters(self):
        return EvidenceRequest.model_json_schema()

    async def execute(self, **kwargs):
        try:
            result = await asyncio.to_thread(evidence, EvidenceRequest.model_validate(kwargs))
            return ToolResult(json.dumps(result, allow_nan=False), "Native scientific evidence", retryable=False)
        except Exception as exc:
            return ToolResult(json.dumps({"error": str(exc)}), "Evidence computation failed", retryable=False)


class WorkflowTool(Tool):
    def __init__(self, config, action):
        self.config, self.action = config, action
        self.external_effects = action in {"start", "resume", "regions"}

    @property
    def name(self):
        return "protein_design_" + self.action

    @property
    def description(self):
        return {
            "regions": "Supply user-confirmed CDR ranges for the exact pending candidate question and resume the saved workflow.",
            "context": "Read configured Torch-Harness model kinds, native skill capabilities and default workflow profile. No OpenDDE compute service or Docker setup is used.",
            "start": "Start the authorized two-stage Qwen-planned Torch-Harness workflow (first evaluation: TorchScore and Rosetta only; second: full evaluation), with compact task inputs and persistent deployment defaults, or inline run_config/workflow_config. Choose task.generators from the user prompt: PPIFlow or TorchCraft; neither is mandatory. No per-task YAML or repeated deployment paths needed. Qwen selects analysis/continue/stop or optimizations using scientific context and mutation_selector; staged rules remain fallback. Uses registered PPIFlow/TorchCraft, TorchScore, TorchFold, Rosetta and compatible TorchOptimization; no initial-round option or compute URL is required. Preserve inputs and checkpoint identity on recovery.",
            "status": "Read a workflow's saved status by run_dir; workers poll remote jobs automatically.",
            "resume": "Resume the same authorized workflow by run_dir, retaining completed scientific work and remote IDs.",
            "list": "Discover indexed Torch-Harness workflows after restarting the session.",
            "candidates": "Read measured candidate JSON from a workflow run_dir; missing metrics remain unknown.",
        }[self.action]

    @property
    def parameters(self):
        if self.action == "regions":
            return {
                "type": "object",
                "properties": {
                    "run_dir": {"type": "string"},
                    "question_id": {"type": "string"},
                    "cdr_position": {"type": "string"},
                    "fixed_positions": {"type": "string"},
                    "user_confirmed": {"type": "boolean", "const": True},
                },
                "required": ["run_dir", "question_id", "cdr_position", "user_confirmed"],
                "additionalProperties": False,
            }
        if self.action in {"context", "list"}:
            return {"type": "object", "properties": {}, "additionalProperties": False}
        if self.action != "start":
            return {
                "type": "object",
                "properties": {"run_dir": {"type": "string"}},
                "required": ["run_dir"],
                "additionalProperties": False,
            }
        schema = InitialRoundConfig.model_json_schema()
        definitions = schema.pop("$defs", {})
        definitions["AutonomousRound"]["properties"]["planner_layout"]["default"] = "two_stage"
        return {
            "type": "object",
            "$defs": definitions,
            "properties": {
                "task": DesignTask.model_json_schema(),
                "run_config": schema,
                "config_path": {"type": "string"},
                "name": {"type": "string"},
                "pdb_path": {"type": "string"},
                "user_confirmed": {"type": "boolean", "const": True},
            },
            "required": ["user_confirmed"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs):
        try:
            if self.action == "regions":
                if kwargs.get("user_confirmed") is not True:
                    raise ValueError("the user must supply/confirm the CDR answer for this pending question")
                from ..initial_round.region_input import supply_regions

                await asyncio.to_thread(
                    supply_regions,
                    Path(kwargs["run_dir"]),
                    kwargs["question_id"],
                    kwargs["cdr_position"],
                    kwargs.get("fixed_positions"),
                )
                result = await asyncio.to_thread(resume_round, Path(kwargs["run_dir"]))
            elif self.action == "context":
                result = {
                    "backend": "registered_models_and_native_analysis",
                    "compute_service_required": False,
                    "default_decision_policy": "planner",
                    "default_planner_layout": "two_stage",
                    "mutation_selector_tool": "protein_design_mutation_selector",
                    "models": [
                        {"alias": key, "kind": value.get("kind")}
                        for key, value in self.config.get("model_services", {}).items()
                    ],
                    "workflow_config": self.config.get("workflow_config"),
                    "workflow_defaults_configured": bool(self.config.get("workflow_defaults")),
                    "design_profiles": sorted(self.config.get("design_profiles", {})),
                    "native_actions": [
                        "developability",
                        "structure",
                        "epitope",
                        "evolution",
                        "mutations",
                        "rank",
                        "lookup",
                    ],
                    "unavailable": {
                        "EpiMatrix": "licensed adapter not configured",
                        "inverse_folding": "no corresponding registered model",
                        "ESM2": "no corresponding registered model",
                    },
                }
            elif self.action == "start":
                if kwargs.get("user_confirmed") is not True:
                    raise ValueError("authorize the scientific inputs and model execution before starting")
                if kwargs.get("task") is not None:
                    if any(kwargs.get(key) is not None for key in ("run_config", "config_path", "pdb_path")):
                        raise ValueError("task is mutually exclusive with run_config, config_path and pdb_path")
                    task = dict(kwargs["task"])
                    if kwargs.get("name"):
                        task["name"] = kwargs["name"]
                    config = await asyncio.to_thread(prepare_design_task, task, self.config, planner_layout="two_stage")
                    result = await asyncio.to_thread(start_round, config)
                    return ToolResult(json.dumps(result, allow_nan=False), "Workflow started", retryable=False)
                if kwargs.get("run_config") is not None and kwargs.get("config_path"):
                    raise ValueError("supply run_config or config_path, not both")
                value = (
                    kwargs.get("run_config")
                    or kwargs.get("config_path")
                    or self.config.get("workflow_config")
                    or self.config.get("initial_round_config")
                )
                if value is None:
                    raise ValueError("assemble run_config from known inputs or select a workflow_config profile")
                config = prepare_workflow(
                    value,
                    registrations=self.config.get("model_services"),
                    plugin_config=self.config,
                    pdb_path=Path(kwargs["pdb_path"]) if kwargs.get("pdb_path") else None,
                    name=kwargs.get("name"),
                )
                result = await asyncio.to_thread(start_round, config)
            elif self.action == "list":
                result = await asyncio.to_thread(list_rounds)
            elif self.action == "candidates":
                result = json.loads((Path(kwargs["run_dir"]) / "candidates.json").read_text())
            else:
                function = resume_round if self.action == "resume" else round_status
                result = await asyncio.to_thread(function, Path(kwargs["run_dir"]))
            return ToolResult(
                json.dumps(result, ensure_ascii=False, allow_nan=False),
                "Torch-Harness workflow " + self.action,
                retryable=False,
            )
        except Exception as exc:
            return ToolResult(json.dumps({"error": str(exc)}), "Workflow operation failed", retryable=False)


def make_context_tool(ctx: PluginContext):
    return WorkflowTool(ctx.config, "context")


def make_start_tool(ctx: PluginContext):
    return WorkflowTool(ctx.config, "start")


def make_status_tool(ctx: PluginContext):
    return WorkflowTool(ctx.config, "status")


def make_resume_tool(ctx: PluginContext):
    return WorkflowTool(ctx.config, "resume")


def make_list_tool(ctx: PluginContext):
    return WorkflowTool(ctx.config, "list")


def make_candidates_tool(ctx: PluginContext):
    return WorkflowTool(ctx.config, "candidates")


def make_analysis_tool(ctx: PluginContext):
    return AnalysisTool()


def make_regions_tool(ctx: PluginContext):
    return WorkflowTool(ctx.config, "regions")
