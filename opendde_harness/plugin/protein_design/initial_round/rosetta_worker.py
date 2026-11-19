"""Standalone PyRosetta Slurm worker: stdlib + PyRosetta, no API server."""

import argparse
import hashlib
import json
import math
from pathlib import Path

AA3_TO_1 = dict(
    zip(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
        "ARNDCQEGHILKMFPSTWYV",
    )
)


def pdb_residues(src: Path) -> dict[tuple[str, int], str]:
    """Read canonical PDB IDs without losing chain or residue numbering."""
    residues = {}
    with src.open() as stream:
        for line in stream:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM  "):
                continue
            if len(line) < 27:
                raise ValueError("truncated PDB atom record")
            chain, name = line[21], line[17:20].strip()
            number = int(line[22:26])
            if not ("A" <= chain <= "Z") or line[26].strip() or number < 1 or name not in AA3_TO_1:
                raise ValueError(
                    "ResResE output requires canonical protein residues, uppercase A-Z chains, positive PDB IDs and no insertion codes"
                )
            key = (chain, number)
            if key in residues and residues[key] != name:
                raise ValueError("ambiguous PDB residue identity")
            residues[key] = name
    if not residues:
        raise ValueError("no canonical PDB residues")
    return residues


def initialize_pyrosetta():
    import pyrosetta
    from pyrosetta.rosetta.core.scoring.methods import LongRangeTwoBodyEnergy

    pyrosetta.init("-mute all -ignore_zero_occupancy false")
    scorefxn = pyrosetta.create_score_function("ref2015")
    options = scorefxn.energy_method_options()
    options.hbond_options().decompose_bb_hb_into_pair_energies(True)
    scorefxn.set_energy_method_options(options)
    # Verify bindings required to retain long-range as well as graph pair terms.
    list(scorefxn.all_methods())
    return pyrosetta, scorefxn, LongRangeTwoBodyEnergy


def write_pyrosetta_pairs(pyrosetta, scorefxn, long_range_type, pose, src, output, group_a, group_b):
    """Weighted two-body totals, matching residue_energy_breakdown's definition.

    EnergyGraph stores short-range terms. Explicitly add active long-range energy
    methods as well; do not silently omit disulfide/other long-range terms.
    """
    expected = pdb_residues(src)
    info = pose.pdb_info()
    if info is None:
        raise ValueError("PyRosetta pose has no PDB residue mapping")
    mapping = {}
    for index in range(1, pose.total_residue() + 1):
        residue = pose.residue(index)
        key = (info.chain(index), info.number(index))
        if str(info.icode(index)).strip() or expected.get(key) != residue.name3() or key in mapping.values():
            raise ValueError("PyRosetta pose does not preserve the input protein residue mapping")
        mapping[index] = key
    if set(mapping.values()) != set(expected):
        raise ValueError("PyRosetta pose is missing input residues")
    scorefxn(pose)
    graph = pose.energies().energy_graph()
    weights = scorefxn.weights()
    terms = list(scorefxn.get_nonzero_weighted_scoretypes())
    long_range = [method for method in scorefxn.all_methods() if isinstance(method, long_range_type)]
    lines = ["ResResE Res1 Res2 total", "ResResE ---- ---- -----"]
    pairs = 0
    # Emit every binder-target pair, including true zero energies. No fabricated
    # sentinel is needed for complexes without favorable contacts.
    binders = [i for i, (chain, _) in mapping.items() if chain in group_a]
    targets = [i for i, (chain, _) in mapping.items() if chain in group_b]
    for i in binders:
        for j in targets:
            emap = pyrosetta.rosetta.core.scoring.EMapVector()
            left, right = sorted((i, j))
            edge = graph.find_energy_edge(left, right)
            if edge is not None:
                emap = edge.fill_energy_map()
            for method in long_range:
                if method.defines_residue_pair_energy(pose, left, right):
                    method.residue_pair_energy(pose.residue(left), pose.residue(right), pose, scorefxn, emap)
            total = sum(float(emap[term]) * float(weights[term]) for term in terms)
            if not math.isfinite(total):
                raise ValueError("non-finite PyRosetta residue-pair energy")

            def label(index):
                chain, number = mapping[index]
                return f"{expected[(chain, number)]}_{ord(chain)}{number}"

            lines.append(f"ResResE {label(i)} {label(j)} {total:.12g}")
            pairs += 1
    if not pairs:
        raise ValueError("empty binder/target residue selection")
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output)
    return pairs


def main(request_path):
    metadata = json.loads(request_path.read_text())
    request = metadata["request"]
    source = Path(metadata["input_pdb"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["input_sha256"]:
        raise ValueError("Rosetta input changed after submission")
    output = Path(metadata["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    group_a, group_b = request["chain_group_a"], request["chain_group_b"]
    if not group_a or not group_b or set(group_a) & set(group_b):
        raise ValueError("invalid interface chain groups")
    pyrosetta, scorefxn, long_range = initialize_pyrosetta()
    pose = pyrosetta.pose_from_pdb(str(source))
    total = float(scorefxn(pose))
    if not math.isfinite(total):
        raise ValueError("non-finite total score")
    write_pyrosetta_pairs(
        pyrosetta, scorefxn, long_range, pose, source, output / "residue_energies.out", group_a, group_b
    )
    partners = group_a + "_" + group_b
    constructor = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover
    try:
        mover = constructor(partners)
    except TypeError:
        mover = constructor()
        try:
            mover.set_interface(partners)
        except TypeError:
            docking_partners = pyrosetta.rosetta.protocols.docking.DockingPartners(partners)
            mover.set_interface(docking_partners)
    mover.set_scorefunction(scorefxn)
    mover.set_pack_input(request.get("pack_input", False))
    mover.set_pack_separated(request.get("pack_separated", True))
    mover.set_compute_packstat(False)
    mover.apply(pose.clone())
    dg = float(mover.get_interface_dG())
    if not math.isfinite(dg):
        raise ValueError("non-finite interface dG")
    (output / "interface.sc").write_text(
        f"SCORE: dG_separated total_score description\nSCORE: {dg:.12g} {total:.12g} input\n"
    )
    files = [output / "interface.sc", output / "residue_energies.out"]
    result = {
        "request_hash": metadata["request_hash"],
        "input_sha256": metadata["input_sha256"],
        "engine": {"pyrosetta_version": str(getattr(pyrosetta, "__version__", "unknown")), "score_weights": "ref2015"},
        "metrics": [{"dG_separated": dg, "total_score": total}],
        "files": [str(p) for p in files],
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(result, allow_nan=False, indent=2))
    temporary.replace(output / "result.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    main(parser.parse_args().request)
