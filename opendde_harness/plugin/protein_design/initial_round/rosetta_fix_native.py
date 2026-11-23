"""Extract PPIFlow-compatible interface residue energies without relaxing the input.

The native ScoreCutoffFilter exporter supplies ResResE on the original candidate
pose. The XML is parsed for its filter, but ParsedProtocol / FastRelax is never
applied: the generated backbone and coordinates are preserved for Partial Flow.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path


def canonical_pdb(source: Path, output: Path, binder: list[str], target: list[str]) -> dict[str, str]:
    if len(target) != 1 or len(binder) not in (1, 2):
        raise ValueError("RosettaFix requires one target and one or two binder chains")
    chains = [*binder, *target]
    if len(set(chains)) != len(chains) or any(len(c) != 1 for c in chains):
        raise ValueError("All input chains must have distinct one-character IDs")
    mapping = dict(zip(chains, ["A", "B", "C"] if len(binder) == 2 else ["A", "C"]))
    records = source.read_text().splitlines(keepends=True)
    atoms = {chain: [] for chain in chains}
    for line in records:
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 54 and line[21] in mapping:
            chain = line[21]
            atoms[chain].append(line[:21] + mapping[chain] + line[22:])
    if any(not atoms[chain] for chain in chains):
        raise ValueError("Generated PDB is missing at least one binder/target chain")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for chain in chains:
            stream.writelines(atoms[chain])
            stream.write("TER\n")
        stream.write("END\n")
    return mapping


def original_contacts(path: Path, binder: str, target: str, cutoff: float) -> set[tuple[int, int]]:
    """Reproduce upstream PPIFlow distance gate: CB (fallback CA) from input PDB."""
    atoms: dict[tuple[str, int, str], tuple[float, float, float]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54 or line[21] not in (binder, target):
            continue
        try:
            resno = int(line[22:26])
            coordinate = tuple(float(line[i:i + 8]) for i in (30, 38, 46))
        except ValueError as exc:
            raise ValueError("Invalid coordinates in PDB for RosettaFix contacts") from exc
        atom = line[12:16].strip()
        if atom in {"CB", "CA"}:
            atoms[(line[21], resno, atom)] = coordinate

    def by_chain(chain: str) -> dict[int, tuple[float, float, float]]:
        selected: dict[int, tuple[float, float, float]] = {}
        positions = {idx for ch, idx, _ in atoms if ch == chain}
        for idx in positions:
            point = atoms.get((chain, idx, "CB"), atoms.get((chain, idx, "CA")))
            if point is not None:
                selected[idx] = point
        return selected

    left, right = by_chain(binder), by_chain(target)
    return {
        (i, j)
        for i, a in left.items()
        for j, b in right.items()
        if math.dist(a, b) <= cutoff
    }


def configured_xml(path: Path, normalized: Path, antibody: bool) -> str:
    """Replicate PPIFlow RosettaFixStep's dynamic XML modifications."""
    text = path.read_text()
    chains: dict[str, list[int]] = {}
    for line in normalized.read_text().splitlines():
        if line.startswith("ATOM  ") and len(line) >= 26:
            chains.setdefault(line[21], []).append(int(line[22:26]))
    if not chains:
        raise ValueError("No residues in RosettaFix PDB")
    ranges = ",".join(f"{min(v)}{k}-{max(v)}{k}" for k, v in chains.items())
    text, count = re.subn(r'resnums="[^"]+"', lambda _: f'resnums="{ranges}"', text, count=1)
    if count != 1:
        raise ValueError("PPIFlow native.xml has no resnums selector")
    if antibody:
        movemap = '''<MoveMap name="backbone">
                <Chain number="1" chi="1" bb="1"/>
                <Chain number="2" chi="1" bb="1"/>
                <Chain number="3" chi="1" bb="0"/>
            </MoveMap>'''
        text, count = re.subn(r'<MoveMap name="backbone">.*?</MoveMap>', movemap, text, count=1, flags=re.DOTALL)
        if count != 1:
            raise ValueError("PPIFlow native.xml lacks the antibody MoveMap")
    return text


def parse_resres_e(log: Path) -> dict[tuple[tuple[str, int], tuple[str, int]], float]:
    """Parse official ScoreCutoffFilter ResResE rows; reject missing/malformed data.

    Accept both upstream ASCII-chain IDs (ALA_651 = A:1) and newer
    amino-acid-ASCII IDs (69_A1 = A:1). Skip the optional weight metadata row.
    """
    rows: dict[tuple[tuple[str, int], tuple[str, int]], float] = {}
    header = None
    for line in log.read_text().splitlines():
        if not line.startswith("ResResE"):
            continue
        parts = line.split()
        if {"Res1", "Res2", "total"} <= set(parts):
            if header is not None and parts != header:
                raise ValueError("Inconsistent ResResE column headers")
            header = parts
            continue
        if header is None:
            raise ValueError("ResResE data before header")
        if all(set(token) <= {"-", "="} for token in parts[1:]):
            continue
        # Newer PyRosetta emits a "ResResE nonzero weights ..." metadata row.
        if len(parts) >= 3 and parts[1:3] == ["nonzero", "weights"]:
            continue
        if len(parts) != len(header):
            raise ValueError("Malformed ResResE row")
        fields = dict(zip(header, parts))
        def residue(value: str) -> tuple[str, int]:
            # Old release: ALA_651 = A:1 (65 is ASCII for chain A).
            old = re.fullmatch(r"[^_]+_([0-9]{2})(-?[0-9]+)", value)
            if old is not None and 65 <= int(old[1]) <= 90:
                return chr(int(old[1])), int(old[2])
            # New release: 69_A1 = A:1 (69 is ASCII for amino acid E).
            new = re.fullmatch(r"[0-9]+_([A-Z])(-?[0-9]+)", value)
            if new is not None:
                return new[1], int(new[2])
            raise ValueError(f"Unsupported Rosetta ResResE residue: {value}")
        first, second = residue(fields["Res1"]), residue(fields["Res2"])
        energy = float(fields["total"])
        if not math.isfinite(energy):
            raise ValueError("Nonfinite ResResE energy")
        key = tuple(sorted((first, second)))
        if key in rows and rows[key] != energy:
            raise ValueError("Conflicting duplicate ResResE energies")
        rows[key] = energy
    if header is None or not rows:
        raise ValueError(f"RosettaFix produced no ResResE table: {log}")
    return rows


def interface_energies_from_resres_e(
    rows: dict[tuple[tuple[str, int], tuple[str, int]], float],
    contacts: set[tuple[int, int]],
    binder: str,
    target: str,
) -> dict[str, float]:
    """Match PPIFlow: sum favorable binder/target ResResE contacts at CB/CA cutoff."""
    result: dict[str, float] = {}
    for bnum, tnum in sorted(contacts):
        key = tuple(sorted(((binder, bnum), (target, tnum))))
        energy = rows.get(key)
        if energy is not None and energy < 0:
            result[str(bnum)] = result.get(str(bnum), 0.0) + energy
    return result


def capture_resres_e(xml_objects, pose, destination: Path, stream_factory):
    """Invoke the original XML ScoreCutoffFilter exporter into a C++ ostringstream."""
    score_filter = xml_objects.get_filter("scofilter")
    if not hasattr(score_filter, "output_residue_pair_energies"):
        raise TypeError(
            "PyRosetta binding did not expose ScoreCutoffFilter.output_residue_pair_energies; "
            "check the installed PyRosetta version"
        )
    stream = stream_factory()
    score_filter.output_residue_pair_energies(stream, pose)
    result = stream.str()
    if not any(line.startswith("ResResE") for line in result.splitlines()):
        raise ValueError("ScoreCutoffFilter emitted no ResResE rows")
    destination.write_text(result)
    return destination


def execute(request: dict) -> dict:
    # Import here so pure-Python unit tests work without an installed PyRosetta wheel.
    import pyrosetta
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

    xml_path = Path(request["native_xml"]).resolve(strict=True)
    output = Path(request["output_dir"]).resolve()
    source = Path(request["input_pdb"]).resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    binder = request["binder_chains"]
    target = request["target_chains"]
    normalized = output / "input" / "candidate.pdb"
    mapping = canonical_pdb(source, normalized, binder, target)
    updated_xml = output / "update.xml"
    updated_xml.write_text(configured_xml(xml_path, normalized, antibody=len(binder) == 2))

    pyrosetta.init(request.get("init_options", "-ignore_zero_occupancy false"))
    pose = pyrosetta.pose_from_pdb(str(normalized))

    xml_objects = XmlObjects.create_from_string(
        updated_xml.read_text()
    )

    # Populate Rosetta's energy graph before exporting
    # residue-pair energies. This does NOT relax the pose.
    scorefxn = pyrosetta.create_score_function("ref2015")
    total_score = scorefxn(pose)

    print(
        f"[RosettaFix] ref2015 score = {total_score:.4f}",
        flush=True,
    )
    print(
        f"[RosettaFix] energies updated = "
        f"{pose.energies().energies_updated()}",
        flush=True,
    )

    resres_e = capture_resres_e(
        xml_objects,
        pose,
        output / "residue-pair-energies.out",
        pyrosetta.rosetta.std.ostringstream,
    )

    rows = parse_resres_e(resres_e)
    residue_energies = {}
    threshold = float(request["interface_dist"])
    for original in binder:
        b = mapping[original]
        t = mapping[target[0]]
        contacts = original_contacts(normalized, b, t, threshold)
        residue_energies[original] = interface_energies_from_resres_e(rows, contacts, b, t)

    return {
        "interface_residue_energies": residue_energies,
        "energy_log": str(resres_e),
        "energy_method": "pyrosetta_scorecutoff_resres_e_unrelaxed",
        "relax_applied": False,
        "scored_structure": str(normalized),
        "normalized_chains": mapping,
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: rosetta_fix_native.py <request.json>")
    request = json.loads(Path(sys.argv[1]).read_text())
    result = execute(request)
    result_path = Path(request["result_path"])
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(result_path)


if __name__ == "__main__":
    main()
