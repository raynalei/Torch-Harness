"""Conversation launch/status tools for an autonomous, detached initial round."""

import asyncio
import json
from pathlib import Path

from opendde_harness.agent.tools.base import Tool, ToolResult
from opendde_harness.plugin.context import PluginContext
from opendde_harness.plugin.protein_design.initial_round.automation import (
    list_rounds,
    prepare_run,
    resume_round,
    round_status,
    start_round,
)
from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig
from opendde_harness.plugin.protein_design.initial_round.task_inputs import DesignTask, prepare_design_task


class InitialRoundTool(Tool):
    def __init__(self, config: dict, action: str):
        self.config = config
        self.action = action
        self.external_effects = action in {"start", "resume", "regions"}

    @property
    def name(self):
        return f"protein_initial_round_{self.action}"

    @property
    def description(self):
        if self.action == "regions":
            return "Resolve a pending CDR identification question with the user-provided cdr_position and optional fixed_positions for the exact candidate PDB, then resume. First show pending_regions and ask the user; do not invent CDRs or treat prior compute authorization as a region answer. Python validates numbering and computes fixed positions as the complement."
        if self.action == "list":
            return "Discover existing initial-round/autonomous workflows after TUI/session restart. Reads the persistent instance run index; optional work_dir discovers pre-index runs on shared storage. Returns run_dir and saved status without submitting inference. Call before claiming there is no active task or creating a replacement task."
        if self.action == "resume":
            return "Resume an authorized workflow from its existing run_dir and immutable snapshot. Reuse completed generation/scoring/folding/results; retry confirmed failed steps only, preserving attempt history. Never choose a new run name to recover. Running or uncertain jobs retain their remote IDs; failed infrastructure does not imply scientific failure."
        if self.action == "status":
            return "Read saved automatic initial-round progress once. If pending_regions is present, show its question, candidate PDB and reason to the user and request their CDR ranges. Otherwise the worker already polls Slurm; do not use repeated LLM calls to poll."
        return "Start the authorized checkpointed workflow. Autonomous rounds use TorchScore ipTM/pTM: initial ipTM<0.5 terminates that candidate; both scores>0.8 go directly to TorchFold/DockQ. Other initial candidates use Rosetta ResResE logs and the supplied interface-energy algorithm to fix binder residues with energy<-5 for Partial Flow. Rescore Partial Flow output: both>0.8 go to TorchFold/DockQ, otherwise run registered TorchOptimization. Rescore optimization output: both>0.8 go to TorchFold/DockQ, otherwise terminate. Hotspot contacts are diagnostic. Qwen chooses only executable menu entries within the round/sample budget. Missing scores/logs are failures, never assumed passes. Do not invent residue energies or bypass Harness. Prefer compact task inputs with persistent deployment settings and server-side model defaults; never load a historical YAML or copy optional model parameters unless explicitly requested; no repeated source/weights/framework/deployment configuration or YAML is required. Advanced callers can supply run_config or config_path. Return run_dir; use status/list/resume after disconnects."

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
        if self.action == "list":
            return {
                "type": "object",
                "properties": {
                    "work_dir": {
                        "type": "string",
                        "description": "Optional original parent directory containing run folders; discovers older workflows.",
                    }
                },
                "additionalProperties": False,
            }
        if self.action in {"status", "resume"}:
            return {
                "type": "object",
                "properties": {"run_dir": {"type": "string"}},
                "required": ["run_dir"],
                "additionalProperties": False,
            }
        run_schema = InitialRoundConfig.model_json_schema()
        definitions = run_schema.pop("$defs", {})
        return {
            "type": "object",
            "$defs": definitions,
            "properties": {
                "task": {
                    **DesignTask.model_json_schema(),
                    "description": "New target task using persistent workflow/design/model defaults. Supply only target PDB/chain, design type, explicit hotspots and total initial_samples; optional framework/fixed-point overrides. Choose generators explicitly from the user prompt: [ppiflow] or [torchcraft]; neither is mandatory. Paired VH/VL requires PPIFlow. No YAML or deployment paths needed.",
                },
                "pdb_path": {"type": "string", "description": "Absolute local mounted shared-filesystem PDB path."},
                "config_path": {
                    "type": "string",
                    "description": "Explicitly requested pipeline YAML only; historical initial_round_config is not loaded automatically.",
                },
                "run_config": {
                    **run_schema,
                    "description": "Complete structured workflow assembled by the agent from existing inputs; no YAML file required. Mutually exclusive with config_path.",
                },
                "name": {"type": "string", "description": "Unique experiment name; reuse on retry."},
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
                from opendde_harness.plugin.protein_design.initial_round.region_input import supply_regions

                await asyncio.to_thread(
                    supply_regions,
                    Path(kwargs["run_dir"]),
                    kwargs["question_id"],
                    kwargs["cdr_position"],
                    kwargs.get("fixed_positions"),
                )
                result = await asyncio.to_thread(resume_round, Path(kwargs["run_dir"]))
            elif self.action == "list":
                work_dir = kwargs.get("work_dir")
                if not work_dir:
                    profile = self.config.get("initial_round_config")
                    if profile and Path(profile).expanduser().is_file():
                        import yaml

                        value = yaml.safe_load(Path(profile).expanduser().read_text())
                        work_dir = value.get("work_dir") if isinstance(value, dict) else None
                result = await asyncio.to_thread(list_rounds, Path(work_dir) if work_dir else None)
            elif self.action == "resume":
                result = await asyncio.to_thread(resume_round, Path(kwargs["run_dir"]))
            elif self.action == "status":
                result = await asyncio.to_thread(round_status, Path(kwargs["run_dir"]))
            else:
                if kwargs.get("user_confirmed") is not True:
                    raise ValueError("user authorization is required for the complete pipeline compute scope")
                task = kwargs.get("task")
                if task is not None:
                    if any(kwargs.get(key) is not None for key in ("run_config", "config_path", "pdb_path")):
                        raise ValueError("task is mutually exclusive with run_config, config_path and pdb_path")
                    config = await asyncio.to_thread(
                        prepare_design_task,
                        {**task, **({"name": kwargs["name"]} if kwargs.get("name") else {})},
                        self.config,
                    )
                    result = await asyncio.to_thread(start_round, config)
                    return ToolResult(
                        json.dumps(result, ensure_ascii=False), "Automatic workflow started", retryable=False
                    )
                profile = kwargs.get("config_path")
                inline = kwargs.get("run_config")
                if inline is not None and kwargs.get("config_path"):
                    raise ValueError("use run_config or config_path, not both")
                if inline is None and not profile:
                    raise ValueError(
                        "supply task inputs using persistent deployment settings, or an explicit run_config/config_path; historical initial_round_config is never loaded automatically"
                    )
                config = prepare_run(
                    inline if inline is not None else Path(profile),
                    pdb_path=Path(kwargs["pdb_path"]) if kwargs.get("pdb_path") else None,
                    name=kwargs.get("name"),
                    registrations=self.config.get("model_services"),
                    plugin_config=self.config,
                )
                result = await asyncio.to_thread(start_round, config)
            return ToolResult(
                json.dumps(result, ensure_ascii=False),
                result["pending_regions"]["question"]
                if result.get("pending_regions")
                else f"Automatic initial round {self.action}",
                retryable=False,
            )
        except Exception as exc:
            return ToolResult(
                json.dumps({"error": str(exc)}, ensure_ascii=False), "Initial round operation failed", retryable=False
            )


def make_initial_round_start_tool(ctx: PluginContext):
    return InitialRoundTool(ctx.config, "start")


def make_initial_round_status_tool(ctx: PluginContext):
    return InitialRoundTool(ctx.config, "status")


def make_initial_round_resume_tool(ctx: PluginContext):
    return InitialRoundTool(ctx.config, "resume")


def make_initial_round_list_tool(ctx: PluginContext):
    return InitialRoundTool(ctx.config, "list")


def make_initial_round_regions_tool(ctx: PluginContext):
    return InitialRoundTool(ctx.config, "regions")
