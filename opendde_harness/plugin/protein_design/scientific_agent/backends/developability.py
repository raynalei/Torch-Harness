"""Chemical-liability motifs, explicit surface patch descriptors and EpiMatrix adapter.

These descriptors are not official TAP calibration or clinical risk categories.
No score is inferred from an LLM opinion, an unavailable structure or missing API.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Protocol

import numpy as np

from ...initial_round.artifacts import structure

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
# Black & Mould normalized hydrophobicities; Tien et al. maximum residue ASA (A^2).
HYDROPHOBICITY = dict(
    zip(
        "ACDEFGHIKLMNPQRSTVWY",
        (
            0.616,
            0.680,
            0.028,
            0.043,
            1.000,
            0.501,
            0.165,
            0.943,
            0.283,
            0.943,
            0.738,
            0.236,
            0.711,
            0.251,
            0.000,
            0.359,
            0.450,
            0.825,
            0.878,
            0.880,
        ),
    )
)
MAX_ASA = dict(
    zip(
        "ARNDCQEGHILKMFPSTWYV",
        (129, 274, 195, 193, 167, 225, 223, 104, 224, 197, 201, 236, 224, 240, 159, 155, 172, 285, 263, 174),
    )
)
MOTIFS = {
    "n_glycosylation_sequon": r"N[^P][ST]",
    "deamidation_motif": r"N[GST]",
    "aspartate_isomerization_motif": r"D[GST]",
    "methionine_oxidation_site": r"M",
    "tryptophan_oxidation_site": r"W",
    "acid_cleavage_motif": r"DP",
    "cysteine_site": r"C",
}


class EpiMatrixAdapter(Protocol):
    """Trusted, operator-installed licensed adapter; no guessed vendor protocol."""

    def predict(self, sequences: Mapping[str, str]) -> dict: ...


def validate_sequences(sequences):
    if not isinstance(sequences, Mapping) or not sequences:
        raise ValueError("provide a nonempty chain -> amino-acid sequence mapping")
    for chain, sequence in sequences.items():
        if (
            not isinstance(chain, str)
            or not chain
            or not isinstance(sequence, str)
            or not sequence
            or set(sequence) - AA
        ):
            raise ValueError("sequences must contain explicit chain IDs and canonical uppercase amino acids")
    return dict(sequences)


def chemical_modification_risks(sequences):
    sequences = validate_sequences(sequences)
    hits = []
    for chain, sequence in sequences.items():
        for name, pattern in MOTIFS.items():
            for match in re.finditer("(?=(" + pattern + "))", sequence):
                hits.append({"chain": chain, "position": match.start() + 1, "motif": match[1], "risk": name})
    return {
        "available": True,
        "method": "sequence_motif_screen_v1",
        "position_semantics": "one-based chain sequence",
        "sites": hits,
        "count": len(hits),
        "interpretation": "Potential sequence liabilities, not PTM occupancy or reaction rates. Cysteine sites are not assumed unpaired; glycosylation sequons are not proof of modification.",
    }


def surface_patches(structure_path, binder_chains, *, distance_cutoff=7.5, exposure_cutoff=0.075):
    """SASA-selected sidechain pair descriptors, reported with their exact definition.

    On isolated binder chains, exposed residue pairs within the cutoff contribute
    H_i*H_j/d_ij^2 to PSH; like-charge D/E and K/R pairs contribute 1/d_ij^2
    to PNC and PSC. Histidine protonation is not assumed. No official TAP label.
    """
    if not binder_chains or len(set(binder_chains)) != len(binder_chains):
        raise ValueError("provide distinct binder chain IDs")
    if not math.isfinite(distance_cutoff) or distance_cutoff <= 0 or not 0 <= exposure_cutoff <= 1:
        raise ValueError("invalid surface-patch distance/exposure parameters")
    import biotite.structure as struc
    from biotite.sequence import ProteinSequence

    atoms = structure(structure_path)
    atoms = atoms[np.isin(atoms.chain_id, binder_chains) & ~np.isin(atoms.element, ["H", "D"])]
    if set(atoms.chain_id.tolist()) != set(binder_chains):
        raise ValueError("binder chains are missing from the structure")
    if not np.isfinite(atoms.coord).all():
        raise ValueError("non-finite structure coordinates")
    sasa = struc.sasa(atoms, point_number=1000)
    if not np.isfinite(sasa).all():
        raise ValueError("SASA could not be computed for every binder atom")
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    exposed, sequences = [], {chain: "" for chain in binder_chains}
    for start, end in zip(starts[:-1], starts[1:]):
        residue = atoms[start:end]
        if len(set(residue.atom_name.tolist())) != len(residue):
            raise ValueError("ambiguous/alternate atom positions require normalization")
        aa = ProteinSequence.convert_letter_3to1(str(residue.res_name[0]))
        chain = str(residue.chain_id[0])
        sequences[chain] += aa
        rsa = float(np.sum(sasa[start:end])) / MAX_ASA[aa]
        if rsa < exposure_cutoff:
            continue
        side = residue[~np.isin(residue.atom_name, ["N", "CA", "C", "O", "OXT"])]
        if not len(side):
            side = residue[residue.atom_name == "CA"]
        if not len(side):
            raise ValueError("residue has no sidechain atoms or CA fallback")
        exposed.append((aa, side.coord))
    scores = {"PSH": 0.0, "PNC": 0.0, "PSC": 0.0}
    pairs = 0
    for i, (aa, coords) in enumerate(exposed):
        for other, other_coords in exposed[i + 1 :]:
            distance = float(np.min(np.linalg.norm(coords[:, None] - other_coords[None, :], axis=-1)))
            if distance < 1e-6:
                raise ValueError("overlapping sidechain coordinates")
            if distance > distance_cutoff:
                continue
            pairs += 1
            scores["PSH"] += HYDROPHOBICITY[aa] * HYDROPHOBICITY[other] / distance**2
            scores["PNC"] += float(aa in "DE" and other in "DE") / distance**2
            scores["PSC"] += float(aa in "KR" and other in "KR") / distance**2
    return {
        "available": True,
        "method": "torch_surface_patch_v1",
        "scores": scores,
        "structure_basis": "isolated binder chains from supplied structure",
        "binder_chains": list(binder_chains),
        "distance_cutoff_angstrom": distance_cutoff,
        "relative_asa_cutoff": exposure_cutoff,
        "exposed_residues": len(exposed),
        "scored_pairs": pairs,
        "sequences": sequences,
        "calibration": "Explicit native descriptors; not the official TAP implementation or calibrated risk thresholds.",
    }


def epimatrix(sequences, adapter: EpiMatrixAdapter | None = None):
    sequences = validate_sequences(sequences)
    if adapter is None:
        return {
            "available": False,
            "method": "EpiMatrix",
            "reason": "licensed EpiMatrix adapter is not configured",
            "scores": None,
        }
    result = adapter.predict(sequences)
    if not isinstance(result, dict) or result.get("method") != "EpiMatrix":
        raise ValueError("EpiMatrix adapter must return a provenance-labelled result")
    # Reject NaN even in nested scores without guessing proprietary units/cutoffs.
    import json

    json.dumps(result, allow_nan=False)
    return result


def assess_developability(sequences, *, structure_path=None, binder_chains=None, epimatrix_adapter=None):
    sequences = validate_sequences(sequences)
    surface = {"available": False, "reason": "binder structure and explicit chain IDs are required", "scores": None}
    if structure_path is not None:
        if not binder_chains or set(binder_chains) != set(sequences):
            raise ValueError("binder chain IDs must match the supplied sequence mapping")
        surface = surface_patches(structure_path, binder_chains)
        if surface.pop("sequences") != sequences:
            raise ValueError("supplied sequences do not match the binder structure")
    return {
        "available": True,
        "ptm_risks": chemical_modification_risks(sequences),
        "surface_patches": surface,
        "immunogenicity": epimatrix(sequences, epimatrix_adapter),
    }
