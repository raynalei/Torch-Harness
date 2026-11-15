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


class InitialRoundTool(Tool):
    def __init__(self, config: dict, action: str):
        self.config = config
        self.action = action
        self.external_effects = action in {"start", "resume", "approve"}

    @property
    def name(self):
        return f"protein_initial_round_{self.action}"

    @property
    def description(self):
        if self.action == "approve":
            return "Approve exactly one pending Partial Flow/TorchOptimization action and resume. First show the pending request and basis and ask the user; user_confirmed=true requires an explicit answer approving this specific action. Prior workflow/budget authorization is insufficient. Never call automatically after status or resume."
        if self.action == "list":
            return "Discover existing initial-round/autonomous workflows after TUI/session restart. Reads the persistent instance run index; optional work_dir discovers pre-index runs on shared storage. Returns run_dir and saved status without submitting inference. Call before claiming there is no active task or creating a replacement task."
        if self.action == "resume":
            return "Resume an authorized workflow from its existing run_dir and immutable snapshot. Reuse completed generation/scoring/folding/results; retry confirmed failed steps only, preserving attempt history. Never choose a new run name to recover. Running or uncertain jobs retain their remote IDs; failed infrastructure does not imply scientific failure."
        if self.action == "status":
            return "Read saved automatic initial-round progress once. The background Python worker already polls Slurm via GET and advances stages; do not use repeated LLM calls to poll."
        return "Start an authorized PPIFlow -> AF3Score/TorchScore -> TorchFold -> DockQ workflow from natural-language instructions. Build run_config from known registered services, tool parameters and Python-derived data; the user does not need to write YAML. Alternatively reuse config_path or the configured profile. Never invent scientific inputs, mounted paths or service addresses. When autonomous is present, this same worker runs expand/regenerate, Partial Flow for encodable local changes, and TorchOptimization for eligible low-confidence VHH/minibinder candidates with a configured torchopt service. These actions are not the legacy protein_design_start loop. Zero accepted candidates does not by itself stop autonomous feedback; bounded rescue candidates are refolded for diagnostics and legal regeneration/repair actions remain available. Autonomous hit classification uses its selected scorer threshold, DockQ threshold and hotspot evidence; evaluation.min_refold_iptm is not an additional autonomous hit gate. Without autonomous, this tool runs only the single initial round. Generation/expansion stays within the authorized budget; every Partial Flow or TorchOptimization action pauses for explicit user approval. On awaiting_approval, show pending_approval.question, request and basis to the user; never approve it yourself. Return run_dir immediately; the worker polls status and releases its private API allocations after completion. Reuse the exact run configuration/name on retries."

    @property
    def parameters(self):
        if self.action == "approve":
            return {
                "type": "object",
                "properties": {
                    "run_dir": {"type": "string"},
                    "approval_id": {"type": "string"},
                    "user_confirmed": {"type": "boolean", "const": True},
                },
                "required": ["run_dir", "approval_id", "user_confirmed"],
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
            if self.action == "approve":
                if kwargs.get("user_confirmed") is not True:
                    raise ValueError("explicit user approval of this pending refinement action is required")
                from opendde_harness.plugin.protein_design.initial_round.approval import approve_action

                await asyncio.to_thread(approve_action, Path(kwargs["run_dir"]), kwargs["approval_id"])
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


def make_initial_round_resume_tool(ctx: PluginContext):
    return InitialRoundTool(ctx.config, "resume")


def make_initial_round_list_tool(ctx: PluginContext):
    return InitialRoundTool(ctx.config, "list")


def make_initial_round_approve_tool(ctx: PluginContext):
    return InitialRoundTool(ctx.config, "approve")
