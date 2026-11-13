"""Conversation launch/status tools for an autonomous, detached initial round."""

import asyncio
import json
from pathlib import Path

from opendde_harness.agent.tools.base import Tool, ToolResult
from opendde_harness.plugin.context import PluginContext
from opendde_harness.plugin.protein_design.initial_round.automation import prepare_run, round_status, start_round
from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig


class InitialRoundTool(Tool):
    def __init__(self, config: dict, action: str):
        self.config = config
        self.action = action
        self.external_effects = action == "start"

    @property
    def name(self):
        return f"protein_initial_round_{self.action}"

    @property
    def description(self):
        if self.action == "status":
            return "Read saved automatic initial-round progress once. The background Python worker already polls Slurm via GET and advances stages; do not use repeated LLM calls to poll."
        return "Start an authorized PPIFlow -> AF3Score/TorchScore -> TorchFold -> DockQ workflow from natural-language instructions. Build run_config from known registered services, tool parameters and Python-derived data; the user does not need to write YAML. Alternatively reuse config_path or the configured profile. Never invent scientific inputs, mounted paths or service addresses. An autonomous configuration authorizes its bounded follow-up budget; no per-round confirmation is needed within that scope. Return run_dir immediately; the worker polls status and releases its private API allocations after completion. Reuse the exact run configuration/name on retries."

    @property
    def parameters(self):
        if self.action == "status":
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
                "pdb_path": {"type": "string", "description": "Absolute local mounted shared-filesystem PDB path."},
                "config_path": {
                    "type": "string",
                    "description": "Pipeline YAML; optional when plugin initial_round_config is configured.",
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
            if self.action == "status":
                result = await asyncio.to_thread(round_status, Path(kwargs["run_dir"]))
            else:
                if kwargs.get("user_confirmed") is not True:
                    raise ValueError("user authorization is required for the complete pipeline compute scope")
                profile = kwargs.get("config_path") or self.config.get("initial_round_config")
                inline = kwargs.get("run_config")
                if inline is not None and kwargs.get("config_path"):
                    raise ValueError("use run_config or config_path, not both")
                if inline is None and not profile:
                    raise ValueError(
                        "assemble run_config from known tool inputs and registered services, or reuse a configured profile; no handwritten YAML is required"
                    )
                config = prepare_run(
                    inline if inline is not None else Path(profile),
                    pdb_path=Path(kwargs["pdb_path"]) if kwargs.get("pdb_path") else None,
                    name=kwargs.get("name"),
                    registrations=self.config.get("model_services"),
                )
                result = await asyncio.to_thread(start_round, config)
            return ToolResult(
                json.dumps(result, ensure_ascii=False), f"Automatic initial round {self.action}", retryable=False
            )
        except Exception as exc:
            return ToolResult(
                json.dumps({"error": str(exc)}, ensure_ascii=False), "Initial round operation failed", retryable=False
            )


def make_initial_round_start_tool(ctx: PluginContext):
    return InitialRoundTool(ctx.config, "start")


def make_initial_round_status_tool(ctx: PluginContext):
    return InitialRoundTool(ctx.config, "status")
