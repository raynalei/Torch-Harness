#!/usr/bin/env python3
"""Convert between PDB + native RFD4 constraints and mmCIF + condition spec."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

CONSTRAINT_KEYS = {"contig", "conditions"}
ATOM_FIELDS = ("chain_id", "res_id", "ins_code", "res_name", "atom_name")


def load_one_design(path: Path) -> tuple[str, dict]:
    from rfproteina.inference_utils.condition_spec import load_spec_file, split_global_args
    from rfproteina.utils.common import validate_design_id

    if path.suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError("use a YAML or JSON constraint/spec file")
    document = load_spec_file(path)
    if not isinstance(document, dict):
        raise ValueError("expected a mapping keyed by design ID")
    designs = split_global_args(document)
    if len(designs) != 1:
        raise ValueError("expected exactly one design in the constraint/spec file")
    design_id, fields = next(iter(designs.items()))
    if not isinstance(design_id, str):
        raise ValueError("design ID must be a string")
    return validate_design_id(design_id), fields


def write_yaml(path: Path, design_id: str, fields: dict) -> None:
    import yaml

    path.write_text(yaml.safe_dump({design_id: fields}, sort_keys=False), encoding="utf-8")


def check_atom_ids(before, after, message: str) -> None:
    import numpy as np

    if any(not np.array_equal(before.get_annotation(key), after.get_annotation(key)) for key in ATOM_FIELDS):
        raise ValueError(message)


def read_source_cif(path: Path):
    from biotite.structure.io import pdbx

    cif = pdbx.CIFFile.read(path)
    if pdbx.get_model_count(cif) != 1:
        raise ValueError("the source CIF must contain exactly one model")
    atom_site = cif.block["atom_site"]
    if "label_alt_id" in atom_site and any(
        code not in {".", "?", "", " "} for code in atom_site["label_alt_id"].as_array(str)
    ):
        raise ValueError("alternate atom locations require a resolved source structure")
    labels = pdbx.get_structure(cif, model=1, use_author_fields=False)
    atoms = pdbx.get_structure(
        cif,
        model=1,
        use_author_fields=True,
        extra_fields=["b_factor", "occupancy", "charge"],
        include_bonds=True,
    )
    check_atom_ids(atoms, labels, "CIF author IDs differ from label IDs used by RFD4 selections")
    return atoms


def read_pdb(path: Path):
    from biotite.structure.io.pdb import PDBFile

    pdb = PDBFile.read(path)
    if pdb.get_model_count() != 1:
        raise ValueError("the PDB must contain exactly one model")
    atoms = pdb.get_structure(model=1, altloc="all")
    if any(code.strip() for code in atoms.altloc_id):
        raise ValueError("alternate atom locations require a resolved source structure")
    return atoms


def pack(pdb: Path, constraints: Path, out: Path) -> None:
    from atomworks.io.config import ParseConfig
    from atomworks.io.parser import parse
    from atomworks.io.utils.io_utils import to_cif_file
    from rfproteina.inference_utils.condition_spec import write_condition_spec_cif

    design_id, fields = load_one_design(constraints)
    if fields.keys() - CONSTRAINT_KEYS:
        raise ValueError("constraints may contain only native contig and conditions fields")
    original = read_pdb(pdb)
    config = ParseConfig(
        add_missing_atoms=False,
        remove_ccds=(),
        remove_waters=False,
        fix_ligands_at_symmetry_centers=False,
        fix_arginines=False,
        build_assembly=None,
        return_atom_array_plus=True,
    )
    atoms = parse(pdb, config=config)["asym_unit"][0]
    source = out / "source.cif"
    to_cif_file(atoms, source, author="Torch-Harness", save_standard_annotations=False)
    check_atom_ids(original, read_source_cif(source), "PDB to CIF conversion changed chain, residue or atom IDs")
    spec = {"input": source.name, **fields}
    write_yaml(out / "spec.yaml", design_id, spec)
    write_condition_spec_cif(design_id, spec, out / "annotated.cif", spec_dir=out)


def unpack(spec: Path, out: Path) -> None:
    from atomworks.io.utils.io_utils import to_pdb_string

    design_id, fields = load_one_design(spec)
    input_path = fields.pop("input", None)
    as_annotated = fields.pop("input_as_annotated_cif", False)
    if as_annotated is not False:
        raise ValueError("reverse conversion requires a plain source CIF and explicit constraints in the spec")
    if fields.keys() - CONSTRAINT_KEYS:
        raise ValueError("unrecognized condition-spec fields")
    if not isinstance(input_path, str) or Path(input_path).suffix.lower() != ".cif":
        raise ValueError("reverse conversion requires one plain .cif file referenced by the spec")

    atoms = read_source_cif(spec.parent / input_path)
    if any(len(chain) != 1 or chain.isspace() for chain in atoms.chain_id):
        raise ValueError("the source CIF contains chain IDs that cannot be written to PDB")
    if any((atoms.res_id < -999) | (atoms.res_id > 9999)) or len(atoms) > 99_999:
        raise ValueError("the source CIF exceeds PDB residue numbering or atom-count limits")

    pdb = out / "source.pdb"
    pdb.write_text(to_pdb_string(atoms), encoding="ascii")
    check_atom_ids(atoms, read_pdb(pdb), "CIF to PDB conversion changed chain, residue or atom IDs")
    write_yaml(out / "constraints.yaml", design_id, fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    forward = subcommands.add_parser("pack", help="PDB and constraints to source CIF, spec and annotated CIF")
    forward.add_argument("--pdb", type=Path, required=True)
    forward.add_argument("--constraints", type=Path, required=True)
    reverse = subcommands.add_parser("unpack", help="spec and its source CIF to PDB and constraints")
    reverse.add_argument("--spec", type=Path, required=True)
    for command in (forward, reverse):
        command.add_argument("--out-dir", type=Path, required=True, help="new output directory")

    args = parser.parse_args()
    output = args.out_dir.absolute()
    try:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output directory already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent, prefix=".rfd4-conversion-") as directory:
            staged = Path(directory) / "output"
            staged.mkdir()
            if args.command == "pack":
                pack(args.pdb, args.constraints, staged)
            else:
                unpack(args.spec, staged)
            staged.rename(output)
    except ImportError as exc:
        parser.exit(1, f"Run this script inside a compatible RFD4-Proteína environment: {exc}\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")

    for path in sorted(output.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
