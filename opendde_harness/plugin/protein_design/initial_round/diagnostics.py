"""Hotspot evidence and staged confidence routing; contacts are diagnostic."""

from __future__ import annotations

import re

import numpy as np

from .artifacts import chain_sequences, structure
from .policy import candidate_stage, hotspot_hit, rule_actions


def _ca(atoms, chain):
    selected = atoms[(atoms.chain_id == chain) & (atoms.atom_name == "CA")]
    keys = list(zip(selected.res_id.tolist(), selected.ins_code.tolist()))
    if len(keys) != len(set(keys)):
        raise ValueError("ambiguous C-alpha residue mapping")
    return selected


def classify(metrics, evidence, options, score_metric="torchscore_iptm", *, stage="initial", fixed_positions=None):
    actions = rule_actions(
        metrics,
        evidence.get("epitope_hit"),
        score_metric,
        options.min_score,
        options.min_ptm,
        options.min_dockq,
        stage=stage,
        min_refold_iptm=options.min_refold_score,
        fixed_positions=fixed_positions,
    )
    if actions == ["terminate"]:
        return "low_iptm" if stage == "initial" and metrics.get(score_metric, 1) < options.min_score else "terminated"
    return {
        ("torchopt",): "needs_optimization",
        ("partial_flow",): "interface_fixed_points",
        ("score_interface",): "needs_interface_score",
        ("refold",): "needs_refold",
        ("stop",): "hit",
    }.get(tuple(actions), "evaluation_incomplete")


def analyze(runner, candidate):
    options = runner.config.autonomous
    chains = runner.config.generators[candidate["source"]].chains
    refolded = bool(candidate.get("refold_structure"))

    def observed_chain(chain):
        return chains.refold_map[chain] if refolded else chain

    path = candidate["refold_structure"] if refolded else candidate["generated_structure"]
    folded = structure(runner.files.local(path))
    if not np.isfinite(folded.coord).all():
        raise ValueError("non-finite structure coordinates")
    evidence = {"epitope_hit": None, "hotspot_structure": "refold" if refolded else "generated", "notes": []}
    try:
        if "ppiflow" in runner.config.generators:
            ppiflow = runner.config.generators["ppiflow"].request
            hotspots = ppiflow.get("specified_hotspots", "")
            input_path = ppiflow.get("antigen_pdb") or ppiflow.get("input_pdb")
        else:
            import json

            provenance_path = runner.root / "inputs" / "target-provenance.json"
            provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
            hotspots = ",".join(provenance.get("normalized_hotspots", []))
            input_path = (
                runner.config.generators["torchcraft"].request["pdb_path"]
                if "torchcraft" in runner.config.generators
                else runner.files.remote(runner.root / "inputs" / "target.pdb")
            )
        if not hotspots:
            raise ValueError("no explicit PDB hotspot provenance; TorchCraft hotspot index semantics are not assumed")
        input_atoms = structure(runner.files.local(input_path))
        input_sequences = chain_sequences(runner.files.local(input_path))
        contacts = []
        binder_atoms = folded[
            np.isin(folded.chain_id, [observed_chain(c) for c in chains.binder]) & (folded.element != "H")
        ]
        if not len(binder_atoms):
            raise ValueError("missing binder heavy atoms")
        for token in hotspots.split(","):
            match = re.fullmatch(r"([A-Za-z]+):?(-?\d+)([A-Za-z]?)", token.strip())
            if not match:
                raise ValueError("hotspots require explicit chain and PDB residue IDs, e.g. C:42")
            chain, number, insertion = match.groups()
            source_ca = _ca(input_atoms, chain)
            matches = np.flatnonzero((source_ca.res_id == int(number)) & (source_ca.ins_code == insertion))
            if len(matches) != 1:
                raise ValueError("hotspot is not uniquely mapped in the input structure")
            input_sequence = input_sequences[chain]
            if len(source_ca) != len(input_sequence):
                raise ValueError("input target has incomplete C-alpha coordinates")
            target_matches = []
            for target in chains.target:
                sequence = candidate["chains"][target]
                offset = input_sequence.find(sequence)
                if offset >= 0 and input_sequence.find(sequence, offset + 1) < 0:
                    index = int(matches[0]) - offset
                    if 0 <= index < len(sequence):
                        target_matches.append((target, index))
            if len(target_matches) != 1:
                raise ValueError("hotspot cannot be mapped uniquely through target cropping/chain renaming")
            target, index = target_matches[0]
            ca = _ca(folded, observed_chain(target))
            if len(ca) != len(candidate["chains"][target]):
                raise ValueError("incomplete target coordinates")
            residue = ca[index]
            atoms = folded[
                (folded.chain_id == residue.chain_id)
                & (folded.res_id == residue.res_id)
                & (folded.ins_code == residue.ins_code)
                & (folded.element != "H")
            ]
            distance = float(np.linalg.norm(atoms.coord[:, None] - binder_atoms.coord[None, :], axis=-1).min())
            contacts.append(
                {"hotspot": token.strip(), "distance": distance, "contact": distance < options.contact_distance}
            )
        evidence.update(hotspots=contacts, epitope_hit=hotspot_hit(contacts, options.contact_distance))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        evidence["notes"].append(str(exc))
    stage = candidate_stage(candidate)
    fixed = candidate.get("interface_fixed_positions")
    evidence["diagnosis"] = classify(
        candidate["metrics"],
        evidence,
        options,
        runner.config.evaluation.score_metric,
        stage=stage,
        fixed_positions=fixed,
    )
    evidence["allowed_actions"] = rule_actions(
        candidate["metrics"],
        evidence["epitope_hit"],
        runner.config.evaluation.score_metric,
        options.min_score,
        options.min_ptm,
        options.min_dockq,
        stage=stage,
        min_refold_iptm=options.min_refold_score,
        fixed_positions=fixed,
    )
    evidence["stage"] = stage
    return evidence
