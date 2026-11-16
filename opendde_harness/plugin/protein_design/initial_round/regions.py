"""Python antibody framework alignment and PDB residue-range encoding."""

from __future__ import annotations

import re

import numpy as np

from .diagnostics import _ca


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


def check_region_encoding(atoms, selections):
    try:
        encoded = encode_positions(atoms, selections)
        if not encoded:
            raise ValueError("empty residue selection")
        return {"ok": True, "encoded": encoded, "reason": None}
    except ValueError as exc:
        return {"ok": False, "encoded": None, "reason": str(exc)}


# Kabat-like reference segments for a deterministic framework-alignment estimate.
# These are not an IMGT numbering service; ambiguous/truncated domains are rejected.
CDR_REFERENCES = {
    "heavy": (
        "EVQLVESGGGLVQPGGSLRLSCAAS",
        "GFTFSSYAMS",
        "WVRQAPGKGLEWVS",
        "AISGSGGSTYYADSVKG",
        "RFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
        "AKDRGYYGMDV",
        "WGQGTLVTVSS",
    ),
    "light": (
        "DIQMTQSPSSLSASVGDRVTITC",
        "RASQDVNTAVA",
        "WYQQKPGKAPKLLIY",
        "SASFLYS",
        "GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC",
        "QQHYTTPPT",
        "FGQGTKVEIK",
    ),
}


def infer_cdr_positions(sequence, role):
    """Map reference CDRs plus query insertions, requiring a conserved framework."""
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import SubstitutionMatrix, align_optimal

    if not 80 <= len(sequence) <= 180:
        raise ValueError("binder is not a supported complete single antibody variable domain")
    segments = CDR_REFERENCES[role]
    reference = "".join(segments)
    ranges, cursor = [], 0
    for index, segment in enumerate(segments):
        if index % 2:
            ranges.append(set(range(cursor, cursor + len(segment))))
        cursor += len(segment)
    cdr = set.union(*ranges)
    trace = align_optimal(
        ProteinSequence(reference),
        ProteinSequence(sequence),
        SubstitutionMatrix.std_protein_matrix(),
        gap_penalty=(-10, -1),
        terminal_penalty=True,
        max_number=1,
    )[0].trace
    mapping = {int(i): int(j) for i, j in trace if i >= 0 and j >= 0}
    framework = set(range(len(reference))) - cdr
    coverage = sum(i in mapping for i in framework) / len(framework)
    identity = sum(i in mapping and reference[i] == sequence[mapping[i]] for i in framework) / len(framework)
    anchors = [
        len(segments[0]) - 4 if role == "heavy" else len(segments[0]) - 1,
        len(segments[0]) + len(segments[1]),
        sum(map(len, segments[:5])) - 1,
        sum(map(len, segments[:6])),
    ]
    if (
        coverage < 0.9
        or identity < 0.7
        or any(i not in mapping or reference[i] != sequence[mapping[i]] for i in anchors)
    ):
        raise ValueError("CDR framework alignment lacks conserved anchors/coverage")
    selected = set()
    previous = -1
    for i, j in trace:
        if i >= 0:
            previous = int(i)
        if j >= 0 and (int(i) in cdr if i >= 0 else previous in cdr):
            selected.add(int(j) + 1)
    if any(sum(mapping.get(i, -1) + 1 in selected for i in region) < 2 for region in ranges):
        raise ValueError("one or more CDRs cannot be mapped reliably")
    return selected


def python_partial_positions(atoms, chains, heavy_chain, light_chain=None):
    """Derive both CLI selections from sequences and the candidate's real PDB IDs."""
    # Use the same C-alpha residue traversal as CLI encoding, not assumed PDB numbering.
    import biotite.structure as struc
    from biotite.sequence import ProteinSequence

    mutable = {}
    roles = {heavy_chain: "heavy", **({light_chain: "light"} if light_chain else {})}
    if set(roles) != set(chains.binder):
        raise ValueError("antibody chain roles do not match binder chains")
    for chain, role in roles.items():
        selected = atoms[atoms.chain_id == chain]
        sequence = "".join(ProteinSequence.convert_letter_3to1(str(res)) for res in struc.get_residues(selected)[1])
        ProteinSequence(sequence)
        if len(_ca(atoms, chain)) != len(sequence):
            raise ValueError("CDR positions need complete C-alpha coordinates")
        mutable[chain] = infer_cdr_positions(sequence, role)
    fixed = {chain: set(range(1, len(_ca(atoms, chain)) + 1)) - positions for chain, positions in mutable.items()}
    cdr_encoding = check_region_encoding(atoms, mutable)
    fixed_encoding = check_region_encoding(atoms, fixed)
    if not cdr_encoding["ok"] or not fixed_encoding["ok"]:
        raise ValueError("Python-derived CDR/framework positions cannot be encoded by PPIFlow")
    return {
        "cdr_position": cdr_encoding["encoded"],
        "fixed_positions": fixed_encoding["encoded"],
        "mutable_positions": {chain: sorted(positions) for chain, positions in mutable.items()},
        "method": "python_reference_framework_alignment",
    }
