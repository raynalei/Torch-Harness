"""Small deterministic predicates used by the diagram and the executable router."""

from __future__ import annotations


def choose_scorer(explicit, configured, registered):
    if explicit is not None:
        return explicit
    return "torchscore" if (configured or registered) == {"torchscore"} else "af3score"


def hotspot_hit(contacts, cutoff=5.0):
    if not contacts:
        return None
    return any(row["distance"] < cutoff for row in contacts)


def repair_allowed(candidate, repaired):
    return candidate["candidate_id"] not in repaired


def torchopt_supported(chains, mode, registered):
    return (
        bool(registered)
        and mode in {"nanobody", "vhh", "binder", "mini_binder"}
        and (
            len(chains.binder) == len(chains.target) == 1
            and all(len(chain) == 1 for chain in chains.binder + chains.target)
        )
    )


def stop_reason(round_index, used_samples, stale_rounds, options):
    if round_index > options.max_rounds:
        return "max_rounds"
    if used_samples >= options.max_samples:
        return "sample_budget"
    if stale_rounds >= options.patience:
        return "no_improvement"
    return None


def score_passes(metrics, metric, iptm_threshold, ptm_threshold=0.8, *, require_ptm=False):
    import math

    ptm_metric = metric.removesuffix("_iptm") + "_ptm"
    needs_ptm = require_ptm or metric == "torchscore_iptm"
    fields = [metric, *([ptm_metric] if needs_ptm else [])]
    if any(type(metrics.get(key)) not in (int, float) or not math.isfinite(metrics[key]) for key in fields):
        return False
    return metrics[metric] > iptm_threshold and (not needs_ptm or metrics[ptm_metric] > ptm_threshold)


def rule_actions(metrics, hotspot, score_metric="torchscore_iptm", min_iptm=0.5, min_ptm=0.8, min_dockq=0.49):
    """Ordered decision table. Unknown required evidence never passes a row."""
    import math

    def finite(key):
        value = metrics.get(key)
        return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1

    if hotspot is False:
        return ["regenerate"]
    if hotspot is not True or not finite(score_metric):
        return []
    if metrics[score_metric] <= min_iptm:
        return ["regenerate"]
    ptm_metric = score_metric.removesuffix("_iptm") + "_ptm"
    if not finite(ptm_metric):
        return []
    if metrics[ptm_metric] <= min_ptm:
        return ["torchopt"]
    if not finite("design_refold_dockq"):
        return []
    if metrics["design_refold_dockq"] <= min_dockq:
        return ["partial_flow"]
    return ["partial_flow", "expand", "stop"]
