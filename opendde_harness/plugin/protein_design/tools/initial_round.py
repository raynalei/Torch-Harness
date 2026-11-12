"""Conversation launch/status tools for an autonomous, detached initial round."""

import asyncio
import json
from pathlib import Path

from opendde_harness.agent.tools.base import Tool, ToolResult
from opendde_harness.plugin.context import PluginContext
from opendde_harness.plugin.protein_design.initial_round.automation import prepare_run, round_status, start_round


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
        return "Start an authorized autonomous PPIFlow (nanobody, binder or paired-chain antibody) -> selected AF3Score/TorchScore threshold -> TorchFold refold -> DockQ pipeline in a background worker. Use a configured pipeline profile plus the mounted input PDB; no per-stage LLM decisions are needed. Return run_dir immediately. The worker polls remote GET status and continues automatically after successful exit codes. Reuse the same profile/PDB/run name on retries. Framework, chain/hotspot policy and service paths must be prepared in the profile."

    @property
    def parameters(self):
        if self.action == "status":
            return {
                "type": "object",
                "properties": {"run_dir": {"type": "string"}},
                "required": ["run_dir"],
                "additionalProperties": False,
            }
        return {
            "type": "object",
            "properties": {
                "pdb_path": {"type": "string", "description": "Absolute local mounted shared-filesystem PDB path."},
                "config_path": {
                    "type": "string",
                    "description": "Pipeline YAML; optional when plugin initial_round_config is configured.",
                },
                "name": {"type": "string", "description": "Unique experiment name; reuse on retry."},
                "user_confirmed": {"type": "boolean", "const": True},
            },
            "required": ["pdb_path", "user_confirmed"],
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
                if not profile:
                    raise ValueError("set plugin initial_round_config or supply config_path for the prepared pipeline")
                config = prepare_run(Path(profile), pdb_path=Path(kwargs["pdb_path"]), name=kwargs.get("name"))
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
