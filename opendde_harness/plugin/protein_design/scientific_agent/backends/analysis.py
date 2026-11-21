"""Native contacts, lineage and fixed-position mutation validation."""

from __future__ import annotations

import numpy as np

from ...initial_round.artifacts import chain_sequences, structure
from .developability import validate_sequences


def analyze_structure(structure_path, binder_chains, target_chains, *, hotspots=(), cutoff=5.0):
    if (
        not binder_chains
        or not target_chains
        or set(binder_chains) & set(target_chains)
        or not np.isfinite(cutoff)
        or cutoff <= 0
    ):
        raise ValueError("provide disjoint nonempty chain groups and a positive contact cutoff")
    atoms = structure(structure_path)
    atoms = atoms[~np.isin(atoms.element, ["H", "D"])]
    if not set(binder_chains + target_chains) <= set(atoms.chain_id.tolist()) or not np.isfinite(atoms.coord).all():
        raise ValueError("missing chain groups or non-finite coordinates")
    binder = atoms[np.isin(atoms.chain_id, binder_chains)]
    target = atoms[np.isin(atoms.chain_id, target_chains)]
    contacts = {}
    # Block distances to avoid allocating a complete all-atom contact matrix.
    for start in range(0, len(target), 256):
        distances = np.min(
            np.linalg.norm(target.coord[start : start + 256, None] - binder.coord[None, :], axis=-1), axis=1
        )
        for atom, distance in zip(target[start : start + 256], distances):
            key = (str(atom.chain_id), int(atom.res_id), str(atom.ins_code))
            contacts[key] = min(float(distance), contacts.get(key, float("inf")))
    rows = [
        {"chain": chain, "residue_id": number, "insertion_code": ins, "distance": distance}
        for (chain, number, ins), distance in sorted(contacts.items())
        if distance < cutoff
    ]
    observed = []
    for item in hotspots:
        chain, number = item["chain"], item["residue_id"]
        key = (chain, number, item.get("insertion_code", ""))
        if key not in contacts:
            raise ValueError("hotspot is not mapped to an explicit target PDB residue")
        observed.append({**item, "distance": contacts[key], "contact": contacts[key] < cutoff})
    return {
        "available": True,
        "method": "minimum_heavy_atom_distance",
        "distance_cutoff_angstrom": cutoff,
        "chain_sequences": chain_sequences(structure_path),
        "epitope_residues": rows,
        "hotspots": observed,
        "epitope_hit": any(row["contact"] for row in observed) if observed else None,
        "confidence": {
            "available": False,
            "reason": "read TorchScore/TorchFold metrics separately; PDB B factors are not assumed pLDDT",
        },
    }


def analyze_evolution(candidates):
    ids = [c["candidate_id"] for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate identities")
    parents = {c["candidate_id"]: c.get("parent_id") or c.get("metadata", {}).get("parent_id") for c in candidates}
    for identity in ids:
        seen, cursor = set(), identity
        while cursor in parents and parents[cursor] is not None:
            if cursor in seen:
                raise ValueError("cyclic candidate lineage")
            seen.add(cursor)
            cursor = parents[cursor]
    return {
        "available": True,
        "method": "recorded_parent_ids",
        "nodes": ids,
        "edges": [{"parent_id": parent, "candidate_id": identity} for identity, parent in parents.items() if parent],
        "missing_parents": sorted({p for p in parents.values() if p and p not in parents}),
    }


def apply_mutations(sequences, mutations, fixed_positions=None):
    result = {chain: list(sequence) for chain, sequence in validate_sequences(sequences).items()}
    fixed_positions = fixed_positions or {}
    seen = set()
    for mutation in mutations:
        chain, position, amino_acid = mutation["chain"], mutation["position"], mutation["amino_acid"]
        if chain not in result or type(position) is not int or not 1 <= position <= len(result[chain]):
            raise ValueError("mutation position outside the supplied chain sequence")
        if position in fixed_positions.get(chain, []) or (chain, position) in seen:
            raise ValueError("mutation changes a fixed position or repeats a position")
        validate_sequences({chain: amino_acid})
        if len(amino_acid) != 1 or result[chain][position - 1] == amino_acid:
            raise ValueError("expected a non-no-op single-residue substitution")
        result[chain][position - 1] = amino_acid
        seen.add((chain, position))
    return {
        "available": True,
        "method": "validated_substitutions",
        "sequences": {c: "".join(s) for c, s in result.items()},
        "requires_rescoring": True,
    }


def rank_candidates(candidates):
    """Report complete measured confidence tuples without manufacturing missing ranks."""
    import math

    fields = ("torchscore_iptm", "torchscore_ptm", "design_refold_dockq")
    ranked, unranked = [], []
    for candidate in candidates:
        metrics = candidate.get("metrics", {})
        if not all(
            type(metrics.get(k)) in (int, float) and math.isfinite(metrics[k]) and 0 <= metrics[k] <= 1 for k in fields
        ):
            unranked.append(candidate["candidate_id"])
            continue
        ranked.append(
            {
                "candidate_id": candidate["candidate_id"],
                "status": candidate.get("status"),
                "metrics": {k: metrics[k] for k in fields},
            }
        )
    ranked.sort(key=lambda c: (-c["metrics"]["torchscore_iptm"], c["candidate_id"]))
    return {
        "available": True,
        "method": "measured_iptm_desc",
        "ranking": ranked,
        "unranked_missing_metrics": unranked,
        "changes_acceptance": False,
    }
