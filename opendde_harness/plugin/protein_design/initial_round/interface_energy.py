"""Interface-energy calculation adapted from the supplied 1get_interface_energy.py.

Retain its CB/CA contact cutoff, negative interchain pair energies and per-binder
residue sums. Use the existing structure reader instead of plotting/dataframe
packages; retain chain identity and reject malformed scientific output.
"""

from __future__ import annotations

import math
import re

import numpy as np

from .artifacts import structure
from .regions import encode_positions


def residue_pairs(logfile):
    """Read the ResResE table, whose residue IDs encode chain IDs as ASCII."""
    header = None
    pairs = {}
    with logfile.open() as stream:
        for line in stream:
            if not line.startswith("ResResE"):
                continue
            fields = line.split()
            if {"Res1", "Res2", "total"} <= set(fields):
                if header is not None and fields != header:
                    raise ValueError("inconsistent Rosetta ResResE headers")
                header = fields
                continue
            if header is None:
                raise ValueError("Rosetta ResResE data precedes its header")
            if all(set(field) <= {"-", "="} for field in fields[1:]):
                continue
            if len(fields) != len(header):
                raise ValueError("malformed Rosetta ResResE row")
            row = dict(zip(header, fields))
            # Some Rosetta logs repeat a column-label separator after the header.
            if row["Res1"] == "Res1" and row["Res2"] == "Res2":
                continue
            residues = []
            for key in ("Res1", "Res2"):
                match = re.fullmatch(r"[^_]+_([0-9]{2})(-?[0-9]+)", row[key])
                if match is None or not 65 <= int(match[1]) <= 90:
                    raise ValueError(f"unsupported Rosetta residue ID: {row[key]}")
                residues.append((chr(int(match[1])), int(match[2])))
            value = float(row["total"])
            if not math.isfinite(value):
                raise ValueError("non-finite Rosetta pair energy")
            pair = tuple(sorted(residues))
            if pair in pairs and pairs[pair] != value:
                raise ValueError("conflicting duplicate Rosetta pair energies")
            pairs[pair] = value
    if header is None or not pairs:
        raise ValueError(f"no residue-pair energies in {logfile}; Rosetta must emit a ResResE table")
    return pairs


def interface_energies(logfile, pdb_file, chains, distance_threshold=12.0):
    """Return chain -> PDB residue number -> negative contact-energy sum."""
    atoms = structure(pdb_file)
    if not np.isfinite(atoms.coord).all():
        raise ValueError("non-finite interface coordinates")
    coordinates = {}
    for chain in chains.binder + chains.target:
        selected = atoms[atoms.chain_id == chain]
        if not len(selected):
            raise ValueError(f"missing interface chain {chain}")
        for number in set(selected.res_id.tolist()):
            residue = selected[selected.res_id == number]
            if np.any(residue.ins_code != ""):
                raise ValueError("interface scoring cannot encode insertion-coded residues")
            reference = residue[residue.atom_name == "CB"]
            if not len(reference):
                reference = residue[residue.atom_name == "CA"]
            if len(reference) != 1:
                raise ValueError("interface scoring requires an unambiguous CB or CA")
            coordinates[(chain, number)] = reference.coord[0]
    energies = {chain: {} for chain in chains.binder}
    for (first, second), value in residue_pairs(logfile).items():
        if first not in coordinates or second not in coordinates:
            raise ValueError("Rosetta residue pair does not match the scored structure")
        if first[0] in chains.target and second[0] in chains.binder:
            first, second = second, first
        if first[0] not in chains.binder or second[0] not in chains.target:
            continue
        if value < 0 and np.linalg.norm(coordinates[first] - coordinates[second]) <= distance_threshold:
            chain, residue = first
            energies[chain][str(residue)] = energies[chain].get(str(residue), 0.0) + value
    return energies


def partial_positions(atoms, chains, fixed_residues):
    """Translate energy-selected PDB IDs to the exact mutable/fixed partition."""
    import biotite.structure as struc

    from .diagnostics import _ca

    if set(fixed_residues) != set(chains.binder):
        raise ValueError("interface fixed-point chains do not match the binder")
    fixed, mutable = {}, {}
    for chain in chains.binder:
        ca = _ca(atoms, chain)
        if len(ca) != len(struc.get_residues(atoms[atoms.chain_id == chain])[0]):
            raise ValueError("Partial Flow fixed points require complete C-alpha coordinates")
        ids = ca.res_id.tolist()
        requested = fixed_residues[chain]
        if len(set(ids)) != len(ids) or not set(requested) <= set(ids):
            raise ValueError("interface fixed points are missing or ambiguous in the candidate")
        fixed[chain] = {i + 1 for i, number in enumerate(ids) if number in requested}
        mutable[chain] = set(range(1, len(ca) + 1)) - fixed[chain]
    if not any(fixed.values()) or not any(mutable.values()):
        raise ValueError("Partial Flow requires both fixed interface residues and mutable binder residues")
    return {
        "fixed_positions": encode_positions(atoms, fixed),
        "cdr_position": encode_positions(atoms, mutable),
        "mutable_positions": {chain: sorted(positions) for chain, positions in mutable.items()},
        "method": "rosetta_interface_residue_energy",
    }
