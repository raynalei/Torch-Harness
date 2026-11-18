"""Prepare new scientific inputs against persistent, target-free design profiles."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from opendde_harness.utils.portable_lock import file_lock

from ..core.model_requests import TARGET_REQUEST_FIELDS
from .artifacts import structure
from .config import PathMapping, RoundModel
from .defaults import merge_defaults, validate_settings
from .storage import SharedFiles, fingerprint, write_json


class TorchCraftAdapter(RoundModel):
    hotspot_numbering: Literal["sequence_0_based", "sequence_1_based", "pdb_residue_id"]
    csv_id_column: str = Field(min_length=1)
    csv_sequence_column: str = Field(min_length=1)
    csv_extra_columns: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def distinct_columns(self):
        if self.csv_id_column == self.csv_sequence_column or {self.csv_id_column, self.csv_sequence_column} & set(
            self.csv_extra_columns
        ):
            raise ValueError("TorchCraft CSV columns must be distinct")
        return self


class DesignTask(RoundModel):
    pdb_path: Path
    target_chain: str = Field(min_length=1, max_length=1)
    design_type: Literal["vhh", "antibody", "minibinder"]
    hotspots: list[StrictInt | str]
    initial_samples: int = Field(ge=1, le=10000)
    name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    generators: list[Literal["ppiflow", "torchcraft"]] | None = None
    generator_samples: dict[str, int] | None = None
    framework_dir: str | None = None
    framework_cif_path: str | None = None
    fixed_positions: str | None = None
    generator_overrides: dict[str, dict] = Field(default_factory=dict)
    target_sequence_csv: Path | None = None


def _stable_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise ValueError("prepared target inputs changed; choose a new task name")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def prepare_design_task(task, settings, *, planner_layout=None):
    """Only scientific task fields vary; all effective defaults are snapshotted."""
    import biotite.structure as struc
    from biotite.sequence import ProteinSequence
    from biotite.structure.io.pdb import PDBFile

    from .automation import prepare_run

    task = DesignTask.model_validate(task)
    from .defaults import deployment_settings

    settings = deployment_settings(settings)
    validate_settings(settings)
    profile = settings.get("design_profiles", {}).get(task.design_type)
    if profile is None:
        raise ValueError(f"configure persistent design_profiles.{task.design_type} once before launching this task")
    value = merge_defaults(settings.get("workflow_defaults", {}), profile)
    if "generators" in profile:
        value["generators"] = copy.deepcopy(profile["generators"])
    value["autonomous"] = value.get("autonomous") or {}
    if planner_layout is not None:
        value["autonomous"].setdefault("planner_layout", planner_layout)
    value.setdefault("evaluation", {}).setdefault("scoring_backend", "torchscore")
    files = SharedFiles([PathMapping.model_validate(m) for m in value.get("path_mappings", [])])
    source = task.pdb_path.expanduser().resolve()
    if not source.is_file():
        raise ValueError("new target PDB must be readable on the mounted shared filesystem")
    files.remote(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    original = structure(source)
    target = original[original.chain_id == task.target_chain]
    if not len(target):
        raise ValueError("requested target chain is absent from the new structure")
    starts = struc.get_residue_starts(target, add_exclusive_stop=True)
    ids, residues, sequence = [], [], ""
    for start, end in zip(starts[:-1], starts[1:]):
        residue = target[start:end]
        key = (int(residue.res_id[0]), str(residue.ins_code[0]))
        if key in ids or key[0] < 1 or key[1]:
            raise ValueError("current autonomous Rosetta requires unique positive residue IDs without insertion codes")
        ids.append(key)
        residues.append({"residue_id": key[0], "sequence_position": len(ids)})
        sequence += ProteinSequence.convert_letter_3to1(str(residue.res_name[0]))
    if set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError("target contains unsupported amino acids")
    positions = []
    for item in task.hotspots:
        token = str(item)
        match = re.fullmatch(r"(?:([A-Za-z0-9]):?)?(-?\d+)([A-Za-z]?)", token)
        if not match or (match[1] and match[1] != task.target_chain) or (int(match[2]), match[3]) not in ids:
            raise ValueError("hotspot is not an explicit residue on the new target chain")
        index = ids.index((int(match[2]), match[3]))
        if index in positions:
            raise ValueError("duplicate hotspot")
        positions.append(index)
    enabled = task.generators or list(value.get("generators", {}))
    if not enabled or len(enabled) != len(set(enabled)):
        raise ValueError("select at least one distinct configured generator")
    if set(enabled) - set(value.get("generators", {})) or set(task.generator_overrides) - set(enabled):
        raise ValueError("task requests a generator not configured in its design profile")
    value["generators"] = {
        kind: merge_defaults(value["generators"][kind], task.generator_overrides.get(kind, {})) for kind in enabled
    }
    known = settings.get("model_services", {})
    for kind, generator in value["generators"].items():
        matches = [
            m
            for alias, m in known.items()
            if m["kind"] == kind
            and (not value.get("model_aliases", {}).get(kind) or alias == value["model_aliases"][kind])
        ]
        if len(matches) > 1:
            raise ValueError(f"select persistent model_aliases.{kind} for ambiguous registrations")
        generator["request"] = {
            **copy.deepcopy(matches[0].get("request_defaults", {}) if matches else {}),
            **generator.get("request", {}),
        }
    binders = {c for g in value["generators"].values() for c in g["chains"]["binder"]}
    normalized_chain = task.target_chain
    if normalized_chain in binders or not re.fullmatch("[A-Z]", normalized_chain):
        normalized_chain = next((c for c in "CABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in binders), None)
        if normalized_chain is None:
            raise ValueError("no free supported chain ID for the new target")
    counts = task.generator_samples
    if counts is None:
        quotient, remainder = divmod(task.initial_samples, len(enabled))
        counts = {kind: quotient + (i < remainder) for i, kind in enumerate(enabled)}
    if (
        set(counts) != set(enabled)
        or any(type(n) is not int or n < 1 for n in counts.values())
        or sum(counts.values()) != task.initial_samples
    ):
        raise ValueError("positive generator sample counts must sum to initial_samples")
    identity = fingerprint(
        {"task": task.model_dump(mode="json"), "source": str(source), "sha256": source_hash, "settings": settings, "planner_layout": planner_layout}
    )
    name = task.name or "design_" + identity[:16]
    work = Path(value["work_dir"])
    inputs = work / name / "inputs"
    files.remote(inputs)
    with file_lock(inputs.parent / "target-preparation.lock", blocking=False):
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
            raise ValueError("target PDB changed during preparation; retry with stable input")
        target = target.copy()
        target.chain_id[:] = normalized_chain
        writer = PDBFile()
        writer.set_structure(target)
        buffer = io.StringIO()
        writer.write(buffer)
        _stable_text(inputs / "target.pdb", buffer.getvalue())
        normalized_path = files.remote(inputs / "target.pdb")
        for kind, generator in value["generators"].items():
            request = generator["request"]
            for key in TARGET_REQUEST_FIELDS:
                request.pop(key, None)
            chains = generator["chains"]
            if len(chains["target"]) != 1:
                raise ValueError("compact target tasks support one target chain")
            previous = chains["target"][0]
            chains["refold_map"] = {
                normalized_chain if key == previous else key: mapped for key, mapped in chains["refold_map"].items()
            }
            chains["target"] = [normalized_chain]
            # A template can carry another antigen's MSAs or prepared target features.
            generator["fold_template"] = None
            if kind == "ppiflow":
                mode = {"vhh": "nanobody", "antibody": "antibody", "minibinder": "binder"}[task.design_type]
                request.update(
                    mode=mode,
                    name=name,
                    samples_per_target=counts[kind],
                    output_dir=files.remote(work / name / "generation" / kind),
                )
                request["input_pdb" if mode == "binder" else "antigen_pdb"] = normalized_path
                request["target_chain" if mode == "binder" else "antigen_chain"] = normalized_chain
                request["specified_hotspots"] = ",".join(normalized_chain + str(ids[i][0]) for i in positions)
                if task.framework_dir is not None:
                    request["framework_dir"] = task.framework_dir
                if task.fixed_positions is not None:
                    request["fixed_positions"] = task.fixed_positions
            else:
                if task.design_type == "antibody":
                    raise ValueError("registered TorchCraft does not support paired VH/VL")
                if not positions:
                    raise ValueError("TorchCraft requires explicit hotspots for a new target")
                adapter = TorchCraftAdapter.model_validate(settings.get("target_adapters", {}).get("torchcraft", {}))
                csv_path = inputs / "target.csv"
                if task.target_sequence_csv is not None:
                    csv_source = task.target_sequence_csv.expanduser().resolve()
                    files.remote(csv_source)
                    text = csv_source.read_text()
                    rows = list(csv.DictReader(io.StringIO(text)))
                    if len(rows) != 1 or rows[0].get(adapter.csv_sequence_column) != sequence:
                        raise ValueError("target sequence CSV does not match the new observed target chain")
                else:
                    output = io.StringIO(newline="")
                    fields = [adapter.csv_id_column, adapter.csv_sequence_column, *adapter.csv_extra_columns]
                    csv_writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
                    csv_writer.writeheader()
                    csv_writer.writerow(
                        {
                            adapter.csv_id_column: name,
                            adapter.csv_sequence_column: sequence,
                            **adapter.csv_extra_columns,
                        }
                    )
                    text = output.getvalue()
                _stable_text(csv_path, text)
                indices = [
                    ids[i][0]
                    if adapter.hotspot_numbering == "pdb_residue_id"
                    else i + (adapter.hotspot_numbering == "sequence_1_based")
                    for i in positions
                ]
                request.update(
                    mode="vhh" if task.design_type == "vhh" else "mini_binder",
                    name=name,
                    pdb_path=normalized_path,
                    target_full_seq=files.remote(csv_path),
                    hotspot_indices=",".join(str(i) for i in indices),
                    num_designs=counts[kind],
                )
                if task.framework_cif_path is not None:
                    request["framework_cif_path"] = task.framework_cif_path
                if task.fixed_positions is not None:
                    raise ValueError(
                        "TorchCraft initial API has no fixed_positions argument; select PPIFlow only for this requirement"
                    )
        value["name"] = name
        config = prepare_run(value, plugin_config=settings)
        provenance = {
            "source_pdb": str(source),
            "source_sha256": source_hash,
            "source_chain": task.target_chain,
            "normalized_chain": normalized_chain,
            "sequence_basis": "observed structure residues; missing biological sequence is not inferred",
            "sequence": sequence,
            "residues": residues,
            "hotspots": task.hotspots,
            "normalized_hotspots": [normalized_chain + str(ids[i][0]) for i in positions],
        }
        path = inputs / "target-provenance.json"
        if path.exists() and json.loads(path.read_text()) != provenance:
            raise ValueError("task target provenance changed; choose a new task name")
        write_json(path, provenance)
        return config
