"""PyRosetta implementation of PPIFlow's RosettaFix protocol, without Rosetta CLI/API.

Reuses PPIFlow native.xml: dynamic residue neighborhood, antibody MoveMap,
FastRelax and ScoreCutoffFilter. Reports *weighted EnergyGraph* negative
cross-chain pair scores in the original-input CB/CA contact neighborhood.
EnergyGraph pair totals should be compared with upstream ResResE scores before
using the same numerical threshold as a calibrated scientific cut-off.
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


def weighted_interface_energy(pose, scorefxn, contacts: set[tuple[int, int]], binder: str, target: str) -> dict[str, float]:
    """Sum negative weighted pair terms for inter-chain contacts in the EnergyGraph."""
    scorefxn(pose)  # Populate the EnergyGraph after relaxation.
    graph = pose.energies().energy_graph()
    info = pose.pdb_info()
    lookup: dict[tuple[str, int], int] = {}
    for index in range(1, pose.total_residue() + 1):
        key = (info.chain(index), info.number(index))
        if key in lookup:
            raise ValueError(f"Ambiguous Rosetta PDB residue identifier: {key}")
        lookup[key] = index
    energies: dict[str, float] = {}
    weights = scorefxn.weights()
    for bnum, tnum in sorted(contacts):
        bi = lookup.get((binder, bnum))
        ti = lookup.get((target, tnum))
        if bi is None or ti is None:
            raise ValueError(f"Rosetta relaxation lost contact residue {binder}{bnum}/{target}{tnum}")
        edge = graph.find_energy_edge(bi, ti)
        if edge is None:
            continue
        value = float(edge.dot(weights))
        if not math.isfinite(value):
            raise ValueError("Nonfinite Rosetta weighted residue-pair energy")
        if value < 0:
            energies[str(bnum)] = energies.get(str(bnum), 0.0) + value
    return energies


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
    protocol = XmlObjects.create_from_string(updated_xml.read_text()).get_mover("ParsedProtocol")
    protocol.apply(pose)
    relaxed = output / "relaxed.pdb"
    pose.dump_pdb(str(relaxed))
    scorefxn = pyrosetta.get_fa_scorefxn()
    # Note: the XML may use another score function. The official XML currently
    # does not specify a score function; compare numerical scores before production.
    residue_energies = {}
    threshold = float(request["interface_dist"])
    for original in binder:
        b = mapping[original]
        t = mapping[target[0]]
        contacts = original_contacts(normalized, b, t, threshold)
        residue_energies[original] = weighted_interface_energy(pose, scorefxn, contacts, b, t)

    # Store explicit provenance rather than masquerading as a Rosetta CLI ResResE log.
    energy_path = output / "pair-energy.json"
    energy_path.write_text(json.dumps({
        "method": "pyrosetta_weighted_energy_graph",
        "normalized_chains": mapping,
        "interface_dist": threshold,
        "interface_residue_energies": residue_energies,
        "relaxed_structure": str(relaxed),
    }, indent=2, allow_nan=False) + "\n")
    return {
        "interface_residue_energies": residue_energies,
        "energy_log": str(energy_path),
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
