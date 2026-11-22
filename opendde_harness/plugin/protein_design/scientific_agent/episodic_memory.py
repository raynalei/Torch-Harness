"""Durable, bounded, provenance-aware protein-design decision episodes.

Episodes live on shared work_dir storage (usually /lustre), not chat memory.
A decision is written only after its execution is marked completed. Results
from earlier runs inform the next decision but never authorize model actions.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from opendde_harness.utils.portable_lock import file_lock

logger = logging.getLogger(__name__)


def store_path(runner) -> Path:
    configured = runner.config.autonomous.episodic_memory_path
    return Path(configured) if configured else runner.config.work_dir / "episodic_memory.jsonl"


def _numbers(metrics):
    return {
        key: float(value)
        for key, value in (metrics or {}).items()
        if key in {"torchscore_iptm", "torchscore_ptm", "rosetta_interface_dg",
                   "ranking_score", "torchfold_iptm", "design_refold_dockq"}
        and type(value) in (int, float) and math.isfinite(value)
    }


def _episode(runner, decision):
    parent = next(
        (c for c in runner.checkpoint.data.get("candidates", [])
         if c["candidate_id"] == decision.get("parent_id")), None
    )
    if parent is None:
        return None
    children = [
        c for c in runner.checkpoint.data["candidates"]
        if c.get("parent_id") == parent["candidate_id"]
        and c.get("round_index") == decision.get("round")
    ]
    return {
        "episode_id": f"{runner.root.resolve()}:{decision.get('round')}:{decision.get('parent_id')}",
        "run": str(runner.root.resolve()),
        "round": decision.get("round"),
        "mode": runner.config.generators[parent["source"]].scientific_mode,
        "source": parent["source"],
        "stage": decision.get("scientific_decision_stage"),
        "action": decision.get("action"),
        "policy": decision.get("policy"),
        "fallback": bool(decision.get("fallback")),
        "parent_metrics": _numbers(parent.get("metrics")),
        "parent_diagnosis": (parent.get("diagnostics") or {}).get("diagnosis"),
        "outcome": {
            "state": decision.get("state"),
            "children": len(children),
            "accepted": sum(c.get("status") == "accepted" for c in children),
            "child_metrics": [_numbers(c.get("metrics")) for c in children[:3]],
        },
    }


def record_decision(runner, decision) -> None:
    if decision.get("state") != "completed" or runner.config.autonomous is None:
        return
    try:
        entry = _episode(runner, decision)
        if entry is None:
            return
        path = store_path(runner)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        logger.warning("episodic memory episode unavailable (continuing workflow): %s", exc)
        return
    # Memory is advisory. Failed storage must never turn a successful scientific
    # action into a failed/retried job; keep the checkpoint as source of truth.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(path.with_suffix(path.suffix + ".lock")):
            existing = set()
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        existing.add(json.loads(line)["episode_id"])
                    except (ValueError, KeyError, TypeError):
                        continue
            if entry["episode_id"] in existing:
                return
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, ValueError) as exc:
        logger.warning("episodic memory write unavailable (continuing workflow): %s", exc)


def retrieve(runner, legal_options, *, limit=4):
    """Similarity-ranked cross-run evidence; excludes current run and failures.

    Action overlap is required to prevent unrelated experiences from dominating.
    Stored rows are evidence only; the router's legal menu is authoritative.
    """
    path = store_path(runner)
    if not path.is_file():
        return []
    modes = {
        runner.config.generators[c["source"]].scientific_mode
        for c in runner.checkpoint.data.get("candidates", [])
        if c.get("source") in runner.config.generators
    }
    legal = {o["action"] for o in legal_options}
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-5000:]
    except (OSError, UnicodeError) as exc:
        logger.warning("episodic memory read unavailable (continuing workflow): %s", exc)
        return []
    for line in lines:
        try:
            entry = json.loads(line)
        except (ValueError, TypeError):
            continue
        if (not isinstance(entry, dict)
                or entry.get("run") == str(runner.root.resolve())
                or entry.get("action") not in legal
                or entry.get("mode") not in modes
                or not isinstance(entry.get("outcome"), dict)
                or entry["outcome"].get("state") != "completed"):
            continue
        rows.append(entry)
    rows.sort(key=lambda e: (
        -int(e.get("outcome", {}).get("accepted", 0) > 0),
        -int(e.get("stage") in {o.get("scientific_decision_stage") for o in legal_options}),
        str(e.get("episode_id")),
    ))
    # Never send arbitrary stored strings, paths, sequences or unbounded text.
    return [
        {
            "mode": e["mode"],
            "action": e["action"],
            "stage": e.get("stage"),
            "parent_metrics": _numbers(e.get("parent_metrics")),
            "parent_diagnosis": str(e.get("parent_diagnosis") or "")[:80],
            "outcome": e["outcome"],
        }
        for e in rows[:limit]
    ]
