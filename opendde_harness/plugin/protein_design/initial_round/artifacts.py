"""Structure manifests, prepared fold inputs and explicit metric extraction."""

from __future__ import annotations

import copy
import csv
import io
import math
from pathlib import Path
from typing import Any

from .config import Chains, Service
from .storage import SharedFiles, select, write_json


def output_files(result: Any, service: Service, files: SharedFiles) -> list[Path]:
    values = select(result, service.files_field, [])
    if not isinstance(values, list):
        raise ValueError(f"results.{service.files_field} must be a file list; configure files_field")
    paths = []
    for value in values:
        remote = value if isinstance(value, str) else value.get("path") if isinstance(value, dict) else None
        if not isinstance(remote, str):
            raise ValueError("each output file must be a remote absolute path or an object with path")
        path = files.local(remote)
        if not path.is_file():
            raise ValueError(f"output is not readable on the shared filesystem: {remote}")
        paths.append(path)
    return sorted(set(paths))


def structure(path: Path):
    try:
        import biotite.structure as struc
        from biotite.structure.io import load_structure
    except ImportError as exc:
        raise RuntimeError(
            "install initial round dependencies: uv sync --locked --extra dev --extra initial-round"
        ) from exc
    atoms = load_structure(str(path), model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    return atoms


def chain_sequences(path: Path) -> dict[str, str]:
    import biotite.sequence as seq
    import biotite.structure as struc

    atoms = structure(path)
    result = {}
    for chain in struc.get_chains(atoms):
        residues = struc.get_residues(atoms[atoms.chain_id == chain])[1]
        try:
            result[str(chain)] = "".join(seq.ProteinSequence.convert_letter_3to1(str(r)) for r in residues)
        except KeyError as exc:
            raise ValueError(f"noncanonical amino acid in chain {chain} of {path}") from exc
    return result


def write_pdb(source: Path, destination: Path):
    from biotite.structure.io import save_structure

    atoms = structure(source)
    if any(len(str(chain)) != 1 for chain in set(atoms.chain_id)):
        raise ValueError("PDB export requires single-character chain IDs; normalize the service output chain mapping")
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_structure(str(destination), atoms)


def prepare_fold(template: Path, sequences: dict[str, str], chains: Chains, name: str, output: Path):
    import json

    value = copy.deepcopy(json.loads(template.read_text()))
    if value.get("dialect") != "alphafold3" or not isinstance(value.get("sequences"), list):
        raise ValueError("fold_template must use the alphafold3 JSON dialect with sequences")
    seeds = value.get("modelSeeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("fold_template requires nonnegative integer modelSeeds")
    if type(value.get("version")) is not int:
        raise ValueError("fold_template requires an integer format version supported by TorchFold")
    proteins = {}
    for entity in value["sequences"]:
        protein = entity.get("protein", {})
        chain = protein.get("id")
        if not isinstance(chain, str):
            raise ValueError("initial round templates require one protein entity per chain, with a scalar id")
        if chain in proteins:
            raise ValueError("duplicate chain ID in fold template")
        proteins[chain] = protein
    if set(proteins) != set(chains.refold_map.values()):
        raise ValueError("fold template chain IDs must exactly match refold_map values")
    for original, folded in chains.refold_map.items():
        protein = proteins[folded]
        if original in chains.binder:
            protein.update(sequence=sequences[original], unpairedMsa="", pairedMsa="", templates=[])
            for key in ("unpairedMsaPath", "pairedMsaPath"):
                protein.pop(key, None)
        elif protein.get("sequence") != sequences[original]:
            raise ValueError(
                f"target chain {original} sequence differs from prepared template; provide a matching target"
            )
    value["name"] = name

    def absolute_paths(node):
        if isinstance(node, dict):
            for key, item in node.items():
                if key.endswith("Path") and isinstance(item, str) and not Path(item).is_absolute():
                    raise ValueError(f"fold template {key} must be an absolute remote path")
                absolute_paths(item)
        elif isinstance(node, list):
            for item in node:
                absolute_paths(item)

    absolute_paths(value)
    write_json(output, value)


def metric_records(value: Any, service: Service) -> list[dict]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        selected = select(value, service.metrics_field)
        if selected is not None:
            return metric_records(selected, service)
        return [value]
    raise ValueError("metrics must be a JSON object or a list of objects")


def measured(records: list[dict], field: str, *, confidence: bool = False) -> float:
    values = []
    for record in records:
        raw = select(record, field)
        if raw is None:
            continue
        if isinstance(raw, bool):
            raise ValueError(f"boolean is not a scientific measurement: {field}")
        value = float(raw)
        if not math.isfinite(value) or (confidence and not 0 <= value <= 1):
            raise ValueError(f"invalid metric {field}: {raw}")
        values.append(value)
    if len(values) != 1:
        raise ValueError(f"expected exactly one measurement for {field}, got {len(values)}; configure a scalar field")
    return values[0]


def table_records(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        return list(csv.DictReader(io.StringIO(path.read_text())))
    header = None
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] != "SCORE:":
            continue
        if "description" in fields:
            header = fields[1:]
        elif header is not None:
            rows.append(dict(zip(header, fields[1:], strict=True)))
    return rows


def refold_output(result: dict, service: Service, files: SharedFiles, confidence_glob: str) -> tuple[Path, float]:
    import json

    field = service.metric_fields.get("iptm", "iptm")
    if service.structure_field:
        path = files.local(select(result, service.structure_field))
        if not path.is_file():
            raise ValueError("TorchFold structure_field does not name a readable structure")
        return path, measured(metric_records(result, service), field, confidence=True)
    paths = output_files(result, service, files)
    choices = []
    for confidence in paths:
        if not confidence.match(confidence_glob):
            continue
        score = measured([json.loads(confidence.read_text())], field, confidence=True)
        suffix = "summary_confidences.json"
        if not confidence.name.endswith(suffix):
            raise ValueError("use structure_field for nonstandard TorchFold confidence/structure filenames")
        prefix = confidence.name[: -len(suffix)]
        paired = [
            p for p in paths if p.parent == confidence.parent and p.name in {prefix + "model.cif", prefix + "model.pdb"}
        ]
        if len(paired) != 1:
            raise ValueError(f"confidence must pair with exactly one model.cif/model.pdb: {confidence}")
        choices.append((score, paired[0]))
    if not choices:
        raise ValueError("TorchFold returned no paired confidence/model files; configure response fields")
    score, path = sorted(choices, key=lambda pair: (-pair[0], str(pair[1])))[0]
    return path, score
