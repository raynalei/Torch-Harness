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


def flexibility(loop_fraction, local_rmsd, motion_threshold=2.0):
    if loop_fraction is None or local_rmsd is None:
        return "unknown"
    if loop_fraction >= 0.5 and local_rmsd > motion_threshold:
        return "flexible"
    if loop_fraction < 0.3 and local_rmsd < motion_threshold:
        return "rigid"
    return "unknown"


def local_region(core_rmsd, region_rmsds, core_count, total_count, core_limit=2.0, region_limit=3.0):
    changed = [name for name, value in region_rmsds.items() if value > region_limit]
    if core_count >= 3 and core_count >= total_count / 2 and core_rmsd < core_limit and len(changed) == 1:
        return changed[0]
    return None


def repair_allowed(candidate, repaired):
    return candidate["candidate_id"] not in repaired and not candidate.get("parent_id")


def torchopt_supported(chains, mode, registered):
    return (
        bool(registered)
        and mode in {"nanobody", "vhh", "binder", "mini_binder"}
        and (
            len(chains.binder) == len(chains.target) == 1
            and all(len(chain) == 1 for chain in chains.binder + chains.target)
        )
    )


def generation_order(available, best_flexibility, used):
    preferred = "torchcraft" if best_flexibility == "flexible" else "ppiflow"
    known = best_flexibility in {"rigid", "flexible"}
    return sorted(
        available,
        key=lambda source: (
            (source != preferred, used.count(source)) if known else (used.count(source), source != preferred)
        ),
    )


def stop_reason(round_index, used_samples, stale_rounds, options):
    if round_index > options.max_rounds:
        return "max_rounds"
    if used_samples >= options.max_samples:
        return "sample_budget"
    if stale_rounds >= options.patience:
        return "no_improvement"
    return None
