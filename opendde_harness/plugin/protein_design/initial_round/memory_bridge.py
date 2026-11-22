"""Optional native DesignMemory bridge for the autonomous Initial Round.

The bridge never decides which actions are legal, changes the scorer, or alters
checkpoint recovery. An unavailable memory service is treated as no evidence.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def open_design_memory():
    """Use the same configured MemoryBackend as OpenDDE's protein-design worker."""
    backend = None
    memory = None
    try:
        from opendde_harness.cli._plugin_stack import build_plugin_registry, maybe_build_memory_backend
        from opendde_harness.config.opendde_harness import load_opendde_harness_config
        from opendde_harness.plugin.protein_design.core.memory import DesignMemory

        cfg = load_opendde_harness_config()
        if cfg.memory.backend not in (None, "off"):
            backend = maybe_build_memory_backend(
                cfg.workspace_path, cfg, registry=build_plugin_registry(cfg),
            )
            if backend is not None:
                await backend.start()
                memory = DesignMemory(
                    backend,
                    agent_id=str(getattr(backend, "agent_id", None) or cfg.memory.agent_id),
                    app_id="protein-design-initial-round",
                )
    except Exception as exc:
        logger.warning("initial-round memory unavailable; continuing without recall: %s", exc)
        if backend is not None:
            try:
                await backend.stop()
            except Exception:
                pass
        backend = None
    try:
        yield memory
    finally:
        if backend is not None:
            try:
                await backend.stop()
            except Exception as exc:
                logger.warning("initial-round memory shutdown failed: %s", exc)


async def recall_for_decision(memory, runner, options):
    """Return small, untrusted historical hints; never return legal actions."""
    if memory is None:
        return []
    target = runner.config.name
    actions = sorted({option["action"] for option in options})
    query = "Previously completed protein-design decisions for actions " + ", ".join(actions)
    try:
        hits = await memory.retrieve(target, query, top_k=3)
        return [str(hit)[:400] for hit in hits[:3]]
    except Exception as exc:
        logger.warning("initial-round memory recall failed: %s", exc)
        return []


async def record_completed(memory, runner, decision):
    """Persist an executed decision only; failures never block the design loop."""
    if memory is None or decision.get("state") != "completed":
        return
    candidates = runner.checkpoint.data.get("candidates", [])
    parent = next((c for c in candidates if c.get("candidate_id") == decision.get("parent_id")), None)
    children = [
        c for c in candidates
        if c.get("parent_id") == decision.get("parent_id")
        and c.get("round_index") == decision.get("round")
    ]
    metric_keys = {
        "torchscore_iptm", "torchscore_ptm", "design_refold_dockq",
        "torchfold_iptm", "rosetta_interface_dg", "ranking_score",
    }
    def metrics(candidate):
        return {
            key: value for key, value in (candidate or {}).get("metrics", {}).items()
            if key in metric_keys and type(value) in (int, float)
        }
    outcome = {
        "identity": {"target": runner.config.name, "objective_key": "initial-round"},
        "action": {"primary_skill": decision["action"]},
        "selected_skill_id": decision["action"],
        "task_intent": "Evaluate protein design action on target " + runner.config.name,
        "lesson": {
            "summary": "Completed action with observed metrics, not a validated causal improvement.",
            "verdict": "completed",
        },
        "memory_schema": "initial-round-decision-v1",
        "parent_metrics": metrics(parent),
        "child_metrics": [metrics(child) for child in children[:3]],
        "child_statuses": [child.get("status") for child in children[:10]],
    }
    try:
        await memory.record(
            str(runner.root.resolve()) + ":round-" + str(decision["round"]),
            decision["round"], runner.config.name, outcome,
            ["Action " + decision["action"] + " completed; outcomes are observational."],
        )
    except Exception as exc:
        logger.warning("initial-round memory recording failed: %s", exc)
