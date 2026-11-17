"""Extract Candidate JSON, enrich with executable evidence, compress for Qwen.

This module does not run models, infer missing measurements, or change routing.
Full candidate records and requests remain in the workflow checkpoint.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from ..initial_round.policy import candidate_stage, rule_actions

REFERENCES = Path(__file__).resolve().parents[1] / "skills" / "protein-design" / "references"
GUIDES = ("binder_failure_diagnosis.md", "generation_strategy.md", "continue_or_stop.md")
STRUCTURE_METRICS = (
    "plddt",
    "ptm",
    "iptm",
    "pae",
    "mean_interface_plddt",
    "mean_interface_pae",
    "torchfold_iptm",
    "design_refold_dockq",
)
INTERFACE_METRICS = (
    "rosetta_interface_score",
    "rosetta_interface_dg",
    "ipsae",
    "total_binder_contacts",
    "cdr_contact_fraction",
    "framework_contact_fraction",
    "coverage_ratio",
)
DEVELOPABILITY_FIELDS = (
    "available",
    "reason",
    "ptm_risks",
    "surface_patches",
    "immunogenicity",
    "pi",
    "pI",
    "net_charge",
    "aggregation_score",
    "solubility_score",
    "tap",
    "TAP",
    "hydrophobicity",
    "liability_count",
    "liabilities",
    "cdr_liabilities",
    "passed",
    "risk",
    "flags",
)


class ContextBudgetError(ValueError):
    """A legal option and its essential evidence cannot fit the prompt budget."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def token_cost(value):
    """Conservative UTF-8 byte upper bound for the byte-tokenized Qwen payload.

    No vocabulary download, model-dependent tokenizer or character/4 estimate is
    needed. This bounds the user context; fixed system/output tokens are separate.
    """
    return len(_json(value).encode("utf-8"))


def _small(value, depth=0):
    if depth > 4:
        return None
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value if len(value) <= 240 else value[:240] + " [truncated]"
    if isinstance(value, Mapping):
        return {str(k): _small(v, depth + 1) for k, v in list(value.items())[:12]}
    if isinstance(value, (list, tuple)):
        return [_small(v, depth + 1) for v in value[:12]]
    return None


def _fields(value, names):
    return {name: _small(value[name]) for name in names if name in value} if isinstance(value, Mapping) else {}


def _fixed_summary(value):
    if not isinstance(value, Mapping):
        return None
    return {
        chain: {"count": len(positions), "positions": list(positions)[:12], "omitted": max(0, len(positions) - 12)}
        for chain, positions in value.items()
    }


def extract(candidate, *, score_metric="torchscore_iptm"):
    """Project both autonomous candidate dicts and serialized core Candidate JSON."""
    metrics = candidate.get("metrics") or {}
    metadata = candidate.get("metadata") or {}
    diagnostics = candidate.get("diagnostics") or {}
    reported = {**(metadata.get("fold") or {}), **metrics}
    ptm_metric = score_metric.removesuffix("_iptm") + "_ptm"
    confidence = {
        key: metrics[key]
        if type(metrics.get(key)) in (int, float) and math.isfinite(metrics[key]) and 0 <= metrics[key] <= 1
        else None
        for key in (score_metric, ptm_metric)
    }
    developability = candidate.get("developability") or metadata.get("developability_evidence") or {}
    developability = {**_fields(metrics, DEVELOPABILITY_FIELDS), **_fields(developability, DEVELOPABILITY_FIELDS)}
    # Missing measurements stay unknown, rather than inheriting a passing score.
    if not developability:
        developability = {"available": False}
    fixed = candidate.get("interface_fixed_positions")
    energies = candidate.get("interface_residue_energies") or {}
    favorable = [
        (str(chain), str(residue), energy)
        for chain, values in energies.items()
        for residue, energy in values.items()
        if type(energy) in (float, int) and math.isfinite(energy)
    ]
    favorable.sort(key=lambda row: (row[2], row[0], row[1]))
    return {
        "candidate_id": candidate["candidate_id"],
        "source": candidate.get("source"),
        "parent_id": candidate.get("parent_id") or metadata.get("parent_id"),
        "stage": candidate_stage(candidate),
        "status": candidate.get("status"),
        "lineage_constraints": _fields(
            candidate.get("generation_request", {}), ("mode", "fixed_positions", "cdr_position")
        ),
        "metrics": confidence,
        "structure": {
            "basis": "refold" if candidate.get("refold_structure") else "generated",
            "path": _small(
                candidate.get("refold_structure")
                or candidate.get("generated_structure")
                or candidate.get("structure_path")
            ),
            "metrics": _fields(reported, STRUCTURE_METRICS),
            "units": _fields(candidate.get("metric_units") or metadata.get("metric_units") or {}, STRUCTURE_METRICS),
        },
        "interface": {
            "metrics": _fields(reported, INTERFACE_METRICS),
            "energy_basis": candidate.get("rosetta_structure"),
            "fixed_positions": _fixed_summary(fixed),
            "lowest_residue_energies": favorable[:6],
        },
        "hotspot": _fields(
            diagnostics or metadata.get("gate_evidence") or metadata.get("hotspot_gate") or {},
            (
                "epitope_hit",
                "hotspot_structure",
                "hotspots",
                "notes",
                "gate_passed",
                "contacted_hotspots",
                "missed_hotspots",
                "distance_cutoff_a",
                "contact_map_method",
            ),
        ),
        "developability": developability,
        "diagnostics": _fields(diagnostics, ("diagnosis", "stage", "unavailable_actions")),
    }


def enrich(evidence, candidate, legal_options, *, score_metric, thresholds):
    """Separate rule recommendations from actions actually executable for this parent."""
    recommendation = rule_actions(
        candidate.get("metrics", {}),
        score_metric=score_metric,
        min_iptm=thresholds.get("min_score", 0.5),
        min_ptm=thresholds.get("min_ptm", 0.8),
        min_dockq=thresholds.get("min_dockq", 0.49),
        stage=candidate_stage(candidate),
        min_refold_iptm=thresholds.get("min_refold_score", 0.8),
        fixed_positions=candidate.get("interface_fixed_positions"),
    )
    if "partial_flow" not in recommendation:
        evidence = {
            **evidence,
            "interface": {
                key: value for key, value in evidence["interface"].items() if key != "lowest_residue_energies"
            },
        }
    return {
        **evidence,
        "rule_recommendation": recommendation,
        "available_actions": sorted(
            {o["action"] for o in legal_options if o.get("parent_id") == candidate["candidate_id"]}
        ),
    }


def decision_guidance(actions):
    """Read trusted, packaged skill references; never load paths from candidates."""
    result = {}
    for name in GUIDES:
        text = (REFERENCES / name).read_text(encoding="utf-8")
        sections = text.split("\n## ")
        text = sections[0]
        if name == "generation_strategy.md":
            wanted = {"partial_flow": "Partial redesign", "torchopt": "TorchOptimization"}
            text += "".join(
                "\n## " + section
                for section in sections[1:]
                if any(section.startswith(wanted[action] + "\n") for action in actions if action in wanted)
            )
        result[name] = text
    return result


def compress(context, records, options, *, max_tokens):
    """Bound the complete serialized context, preserving original option IDs.

    Paginating the menu changes prompt size only. Unshown parents remain in the
    checkpoint and can be selected on a subsequent decision; no experiment cap.
    """
    if max_tokens < 1:
        raise ContextBudgetError("decision context budget must be positive")
    base = {
        **context,
        "candidates": [],
        "options": [],
        "compression": {
            "omitted_options": len(options),
            "omitted_candidates": len(records),
            "budget_tokens": max_tokens,
            "omitted_history": max(0, context.get("history_total", 0) - len(context.get("history", []))),
        },
    }
    # Optional detail is removed before dropping candidate/option identities.
    for field in ("history", "guidance"):
        if token_cost(base) > max_tokens:
            base[field] = {} if field == "guidance" else []
            if field == "history":
                base["compression"]["omitted_history"] = context.get("history_total", 0)
    if token_cost(base) > max_tokens:
        raise ContextBudgetError("task constraints exceed the decision context budget")
    by_id = {c["candidate_id"]: c for c in records}
    for option in options:
        trial = {
            **base,
            "options": [*base["options"], option],
            "candidates": list(base["candidates"]),
            "compression": dict(base["compression"]),
        }
        parent = by_id.get(option.get("parent_id"))
        if parent is None:
            raise ContextBudgetError("an executable option has no candidate evidence")
        if parent and not any(c["candidate_id"] == parent["candidate_id"] for c in trial["candidates"]):
            trial["candidates"].append(parent)
        trial["compression"].update(
            omitted_options=len(options) - len(trial["options"]),
            omitted_candidates=len(records) - len(trial["candidates"]),
        )
        if token_cost(trial) <= max_tokens:
            base = trial
    if options and not base["options"]:
        # Retry with compact scientific evidence and without optional guide/history
        # text. Essential confidence, rule and actual capabilities stay together.
        reduced = [
            {
                key: c[key]
                for key in (
                    "candidate_id",
                    "stage",
                    "status",
                    "metrics",
                    "lineage_constraints",
                    "rule_recommendation",
                    "available_actions",
                )
            }
            for c in records
        ]
        if context.get("guidance") or context.get("history") or records != reduced:
            return compress(
                {**context, "guidance": {}, "history": [], "evidence_detail": "minimal"},
                reduced,
                options,
                max_tokens=max_tokens,
            )
        raise ContextBudgetError("no executable option fits the decision context budget")
    return base


def build_context(
    candidates,
    legal_options,
    *,
    round_index,
    score_metric="torchscore_iptm",
    thresholds=None,
    constraints=None,
    history=(),
    unavailable=None,
    max_tokens=4096,
):
    """Extract → enrich → compress. Inputs and executable requests are never mutated."""
    thresholds = thresholds or {}
    projected = [
        enrich(
            extract(c, score_metric=score_metric), c, legal_options, score_metric=score_metric, thresholds=thresholds
        )
        for c in candidates
    ]
    options = [
        {
            "option_id": index,
            "parent_id": option.get("parent_id"),
            **_fields(option, ("action", "source", "region_method")),
        }
        for index, option in enumerate(legal_options)
    ]
    recent = []
    for row in history[-8:]:
        item = _fields(row, ("round", "action", "source", "parent_id", "state"))
        item["results"] = [
            {
                "candidate_id": c["candidate_id"],
                "status": c.get("status"),
                "metrics": _fields(
                    c.get("metrics", {}),
                    (score_metric, score_metric.removesuffix("_iptm") + "_ptm", "design_refold_dockq"),
                ),
            }
            for c in candidates
            if c.get("parent_id") == row.get("parent_id") and c.get("round_index") == row.get("round")
        ][:4]
        recent.append(item)
    context = {
        "round": round_index,
        "scoring_backend": score_metric.removesuffix("_iptm"),
        "thresholds": _small(thresholds),
        "constraints": _small(constraints or {}),
        "history": recent,
        "history_total": len(history),
        "history_summary": dict(Counter(row.get("action", "unknown") for row in history)),
        "guidance": decision_guidance({o["action"] for o in legal_options}),
        "unavailable": _small(unavailable or {}),
    }
    return compress(context, projected, options, max_tokens=max_tokens)


def context_for_runner(runner, options, index):
    config = runner.config
    constraints = {
        source: {
            "mode": generator.request.get("mode"),
            "target_chains": generator.chains.target,
            "binder_chains": generator.chains.binder,
            **_fields(generator.request, ("specified_hotspots", "fixed_positions", "cdr_position")),
        }
        for source, generator in config.generators.items()
    }
    return build_context(
        runner.checkpoint.data["candidates"],
        options,
        round_index=index,
        score_metric=config.evaluation.score_metric,
        thresholds={
            key: getattr(config.autonomous, key)
            for key in (
                "min_score",
                "min_refold_score",
                "min_ptm",
                "min_dockq",
                "interface_energy_threshold",
                "interface_distance",
                "contact_distance",
            )
        },
        constraints=constraints,
        history=runner.checkpoint.data["autonomous"]["decisions"],
        unavailable={} if config.evaluation.torchopt else {"torchopt": "no registered optimization service"},
        max_tokens=config.autonomous.decision_context_tokens,
    )
