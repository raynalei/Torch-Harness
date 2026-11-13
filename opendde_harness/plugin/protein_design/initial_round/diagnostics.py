"""Local geometry evidence for bounded initial-round decisions, without model calls."""

from __future__ import annotations

import math
import re

import numpy as np
from pydantic import Field, model_validator

from .artifacts import chain_sequences, structure
from .config import RoundModel


class DesignRegion(RoundModel):
    chain: str
    positions: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def sequence_positions(self):
        if any(p < 1 for p in self.positions) or len(set(self.positions)) != len(self.positions):
            raise ValueError("design region positions must be unique, one-based sequence positions")
        return self


def _ca(atoms, chain):
    selected = atoms[(atoms.chain_id == chain) & (atoms.atom_name == "CA")]
    keys = list(zip(selected.res_id.tolist(), selected.ins_code.tolist()))
    if len(keys) != len(set(keys)):
        raise ValueError("ambiguous C-alpha residue mapping")
    return selected


def _fit(reference, mobile):
    if len(reference) < 3 or np.linalg.matrix_rank(reference - reference.mean(axis=0)) < 2:
        raise ValueError("insufficient non-collinear target atoms for alignment")
    a, b = reference.mean(axis=0), mobile.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - b).T @ (reference - a))
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    return rotation, a - b @ rotation


def _rmsd(displacements):
    return float(np.sqrt(np.mean(np.square(displacements))))


def classify(metrics, evidence, options, score_metric="af3score_iptm"):
    required = (score_metric, "design_refold_dockq", "rosetta_interface_dg")
    if any(
        isinstance(metrics.get(k), bool)
        or not isinstance(metrics.get(k), (int, float))
        or not math.isfinite(metrics[k])
        for k in required
    ):
        return "evaluation_incomplete"
    if evidence.get("epitope_hit") is False:
        return "off_epitope"
    if evidence.get("epitope_hit") is not True:
        return "uncertain"
    if metrics[score_metric] > options.min_score and metrics["design_refold_dockq"] > options.min_dockq:
        return "hit"
    if metrics["design_refold_dockq"] <= options.min_dockq:
        if evidence.get("local_region"):
            return "local_change"
        return "pose_change"
    return "low_confidence"


def analyze(runner, candidate):
    """Missing/ambiguous mappings yield unknown evidence, never invented regions."""
    options = runner.config.autonomous
    generator = runner.config.generators[candidate["source"]]
    chains = generator.chains
    original = structure(runner.files.local(candidate["generated_structure"]))
    folded = structure(runner.files.local(candidate["refold_structure"]))
    if not np.isfinite(original.coord).all() or not np.isfinite(folded.coord).all():
        raise ValueError("non-finite structure coordinates")
    evidence = {"epitope_hit": None, "local_region": None, "flexibility": "unknown", "notes": []}
    mapped_hotspots = []
    try:
        ppiflow = runner.config.generators["ppiflow"].request
        hotspots = ppiflow.get("specified_hotspots", "")
        if not hotspots:
            raise ValueError("no explicit PPIFlow hotspots; TorchCraft hotspot index semantics are not assumed")
        input_path = ppiflow.get("antigen_pdb") or ppiflow.get("input_pdb")
        input_atoms = structure(runner.files.local(input_path))
        input_sequences = chain_sequences(runner.files.local(input_path))
        contacts = []
        binder_atoms = folded[
            np.isin(folded.chain_id, [chains.refold_map[c] for c in chains.binder]) & (folded.element != "H")
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
            mapped_hotspots.append((target, index))
            ca = _ca(folded, chains.refold_map[target])
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
        evidence.update(hotspots=contacts, epitope_hit=any(c["contact"] for c in contacts))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        evidence["notes"].append(str(exc))
    try:
        reference, mobile = [], []
        for chain in chains.target:
            a, b = _ca(original, chain), _ca(folded, chains.refold_map[chain])
            if len(a) != len(candidate["chains"][chain]) or len(a) != len(b):
                raise ValueError("incomplete target C-alpha coordinates")
            reference.extend(a.coord)
            mobile.extend(b.coord)
        reference, mobile = np.asarray(reference), np.asarray(mobile)
        core = np.ones(len(reference), dtype=bool)
        for _ in range(3):
            rotation, translation = _fit(reference[core], mobile[core])
            deviation = np.linalg.norm(reference - (mobile @ rotation + translation), axis=1)
            keep = deviation <= options.framework_rmsd
            if np.array_equal(core, keep):
                break
            core = keep
        rotation, translation = _fit(reference[core], mobile[core])
        if core.mean() < 0.5:
            raise ValueError("no majority stable target core")
        evidence["target_core_fraction"] = float(core.mean())
        if mapped_hotspots:
            import biotite.structure as struc

            local_motion, loop_flags = [], []
            for chain in chains.target:
                indices = [index for target, index in mapped_hotspots if target == chain]
                if not indices:
                    continue
                a, b = _ca(original, chain), _ca(folded, chains.refold_map[chain])
                neighborhood = (np.linalg.norm(a.coord[:, None] - a.coord[indices][None, :], axis=-1) < 10).any(axis=1)
                sse = struc.annotate_sse(original[original.chain_id == chain])
                if len(sse) != len(a) or np.any(sse[neighborhood] == ""):
                    continue
                loop_flags.extend((sse[neighborhood] == "c").tolist())
                local_motion.extend(np.linalg.norm(a.coord - (b.coord @ rotation + translation), axis=1)[neighborhood])
            if loop_flags:
                fraction, motion = float(np.mean(loop_flags)), _rmsd(local_motion)
                evidence["flexibility_evidence"] = {
                    "loop_fraction": fraction,
                    "local_rmsd": motion,
                    "interpretation": "structure-pair routing hint, not measured dynamics",
                }
                if fraction >= 0.5 and motion > options.framework_rmsd:
                    evidence["flexibility"] = "flexible"
                elif fraction < 0.3 and motion < options.framework_rmsd:
                    evidence["flexibility"] = "rigid"
        from .regions import derive_regions

        deviations, framework, region_rmsd = {}, [], {}
        for chain in chains.binder:
            a, b = _ca(original, chain), _ca(folded, chains.refold_map[chain])
            if len(a) != len(candidate["chains"][chain]) or len(a) != len(b):
                raise ValueError("incomplete binder C-alpha coordinates")
            deviations[chain] = np.linalg.norm(a.coord - (b.coord @ rotation + translation), axis=1)
        request = candidate.get("generation_request", generator.request)
        regions, origin = derive_regions(original, chains, request, deviations, options.region_rmsd)
        evidence["region_source"] = origin
        evidence["repair_regions"] = {name: region.model_dump() for name, region in regions.items()}
        for chain in chains.binder:
            mutable = set()
            for name, region in regions.items():
                if region.chain == chain:
                    positions = np.asarray(region.positions) - 1
                    mutable.update(positions.tolist())
                    region_rmsd[name] = _rmsd(deviations[chain][positions])
            framework.extend(d for index, d in enumerate(deviations[chain]) if index not in mutable)
        evidence["binder_rmsd"] = _rmsd(np.concatenate(list(deviations.values())))
        if regions and len(framework) >= 3 and len(framework) >= sum(map(len, deviations.values())) / 2:
            evidence.update(framework_rmsd=_rmsd(framework), region_rmsd=region_rmsd)
            changed = [name for name, value in region_rmsd.items() if value > options.region_rmsd]
            if evidence["framework_rmsd"] < options.framework_rmsd and len(changed) == 1:
                evidence["local_region"] = changed[0]
        else:
            evidence["notes"].append("no localized mutable loop with a majority stable binder core")
        if evidence["flexibility"] == "unknown":
            evidence["notes"].append("flexibility evidence insufficient; use bounded generator trials")
    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        evidence["local_region"] = None
        evidence["notes"].append(str(exc))
    evidence["diagnosis"] = classify(candidate["metrics"], evidence, options, runner.config.evaluation.score_metric)
    return evidence
