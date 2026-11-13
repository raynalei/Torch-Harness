"""Derive repair selections from registered PPIFlow arguments or local geometry."""

from __future__ import annotations

import re

import numpy as np

from .diagnostics import DesignRegion, _ca


def selection_positions(value, atoms, binders):
    """Translate the registered CLI syntax A26-33,B52-54 to sequence indices."""
    if value is None or value == "":
        return {}
    if not isinstance(value, str):
        raise ValueError("unsupported registered residue selection; expected A26-33,B52-54")
    result = {}
    for token in value.split(","):
        match = re.fullmatch(r"([A-Za-z]+):?(\d+)(?:-(\d+))?", token.strip())
        if not match:
            raise ValueError("invalid registered residue range")
        chain, start, end = match.groups()
        if chain not in binders:
            raise ValueError("repair selections must refer to binder chains")
        start, end = int(start), int(end or start)
        if start > end:
            raise ValueError("reversed residue selection")
        ca = _ca(atoms, chain)
        matches = np.flatnonzero((ca.res_id >= start) & (ca.res_id <= end))
        if len(set(ca.res_id[matches].tolist())) != end - start + 1 or np.any(ca.ins_code[matches] != ""):
            raise ValueError("residue selection is missing or ambiguous in this candidate")
        result.setdefault(chain, set()).update(int(i) + 1 for i in matches)
    return result


def encode_positions(atoms, selections):
    """Emit CLI ranges from actual candidate PDB residue IDs, not sequence indices."""
    tokens = []
    for chain, positions in sorted(selections.items()):
        if not chain.isalpha():
            raise ValueError("unsupported PPIFlow chain selection syntax")
        ca = _ca(atoms, chain)
        numbers = []
        for position in sorted(positions):
            if position < 1 or position > len(ca) or ca.ins_code[position - 1] != "":
                raise ValueError("PPIFlow repair cannot encode an ambiguous residue ID")
            numbers.append(int(ca.res_id[position - 1]))
        if len(set(numbers)) != len(numbers) or any(number < 1 for number in numbers):
            raise ValueError("unsupported PPIFlow residue numbering")
        if not numbers:
            continue
        start = end = numbers[0]
        for number in numbers[1:]:
            if number == end + 1:
                end = number
                continue
            tokens.append(f"{chain}{start}" + (f"-{end}" if end != start else ""))
            start = end = number
        tokens.append(f"{chain}{start}" + (f"-{end}" if end != start else ""))
    return ",".join(tokens)


def derive_regions(atoms, chains, request, displacements, threshold):
    """Fixed tool arguments take precedence over Python-computed loop segments."""
    fixed = selection_positions(request.get("fixed_positions"), atoms, chains.binder)
    specified = request.get("cdr_position")
    if specified:
        selected = selection_positions(specified, atoms, chains.binder)
        origin = "registered_cdr_position"
    else:
        import biotite.structure as struc

        selected = {}
        for chain in chains.binder:
            sse = struc.annotate_sse(atoms[atoms.chain_id == chain])
            if len(sse) != len(displacements[chain]):
                raise ValueError("secondary structure cannot be mapped to binder coordinates")
            selected[chain] = {
                index + 1
                for index, (kind, distance) in enumerate(zip(sse, displacements[chain]))
                if kind == "c" and distance > threshold
            }
        origin = "python_displaced_loop"
    regions = {}
    for chain, positions in selected.items():
        remaining = sorted(positions - fixed.get(chain, set()))
        runs = []
        for position in remaining:
            if not runs or position != runs[-1][-1] + 1:
                runs.append([])
            runs[-1].append(position)
        for index, run in enumerate(runs, 1):
            regions[f"{chain}_region_{index}"] = DesignRegion(chain=chain, positions=run)
    return regions, origin
