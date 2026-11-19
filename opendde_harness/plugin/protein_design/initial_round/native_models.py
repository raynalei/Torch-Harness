"""Native RFD4 structures and ProteinMPNN sequence-to-structure optimization."""

from __future__ import annotations

import gzip
import hashlib
import shutil
from pathlib import Path

from ..core.model_artifacts import normalize_model_results, parse_fasta
from ..core.model_requests import validate_request
from .artifacts import chain_sequences, prepare_fold, refold_output, structure, write_pdb

OPTIMIZATION_ACTIONS = {"partial_flow", "torchopt", "rfd4_partial", "proteinmpnn"}


def rfd4_structures(runner, result, directory):
    result = normalize_model_results("rfd4", result)
    paths = []
    for remote in result["files"]:
        source = runner.files.local(remote)
        name = source.name.lower()
        if not any(name.endswith(ext) for ext in (".pdb", ".cif", ".bcif", ".cif.gz", ".bcif.gz")):
            continue
        if not source.is_file():
            raise ValueError(f"RFD4 structure is not readable: {remote}")
        directory.mkdir(parents=True, exist_ok=True)
        # Include the full path so repeated filenames in replicate directories do not collide.
        key = hashlib.sha256(remote.encode()).hexdigest()[:16]
        if name.endswith(".gz"):
            unpacked = directory / (key + (".bcif" if name.endswith(".bcif.gz") else ".cif"))
            with gzip.open(source, "rb") as handle:
                unpacked.write_bytes(handle.read())
            source = unpacked
        destination = directory / (key + ".pdb")
        write_pdb(source, destination)
        paths.append(destination)
    if not paths:
        raise ValueError("RFD4 returned no supported PDB/CIF/BCIF structures (optionally gzip compressed)")
    return paths


def partial_request(runner, candidate, index):
    evaluation = runner.config.evaluation
    root = evaluation.rfd4_input_root
    if not root or not Path(root).is_absolute():
        raise ValueError("configure evaluation.rfd4_input_root from the RFD4 service RFD4_INPUTS directory")
    relative = f"harness/{runner.config.name}/{candidate['candidate_id']}-r{index}.cif"
    path = runner.files.local(str(Path(root) / relative))
    from biotite.structure.io import save_structure

    path.parent.mkdir(parents=True, exist_ok=True)
    atoms = structure(runner.files.local(candidate["generated_structure"]))
    save_structure(str(path), atoms)
    # Inputs and generation are determined by this action, never inherited from an old task.
    return validate_request(
        "rfd4",
        {
            **evaluation.rfd4_options,
            "inputs": relative,
            "generation": "inference_base",
            "partial_trajectory_fraction_steps": evaluation.rfd4_options.get("partial_trajectory_fraction_steps", 0.2),
            "num_replicates": 1,
            "batch_size": 1,
        },
        omit_defaults=True,
    )


def validate_sequences(parent, sequences, chains, partition):
    if set(sequences) != set(parent["chains"]):
        raise ValueError("optimization changed chain identities")
    if any(sequences[c] != parent["chains"][c] for c in chains.target):
        raise ValueError("optimization changed a target sequence")
    for chain in chains.binder:
        before, after = parent["chains"][chain], sequences[chain]
        mutable = set(partition["mask_positions"][chain])
        if len(before) != len(after) or any(
            a != b and position not in mutable for position, (a, b) in enumerate(zip(before, after), 1)
        ):
            raise ValueError("optimization changed binder residues outside the selected mask")


async def materialize_sequence(runner, parent, sequences, generator, key):
    """Fold a sequence design before evaluation; it has no invented backbone or metrics."""
    directory = runner.root / key
    directory.mkdir(parents=True, exist_ok=True)
    input_json = directory / "sequence-input.json"
    options = runner.config.evaluation.torchfold_options
    prepare_fold(
        None,
        sequences,
        generator.chains,
        key,
        input_json,
        seeds=generator.fold_seeds,
        run_data_pipeline=options.get("run_data_pipeline", False),
    )
    result = await runner.job(
        "torchfold",
        f"materialize/{key}",
        "/v1/folds",
        "fold_id",
        {**options, "name": key, "json_path": runner.files.remote(input_json)},
    )
    path, _ = refold_output(
        result,
        runner.config.evaluation.torchfold,
        runner.files,
        runner.config.evaluation.fold_confidence_glob,
    )
    actual = chain_sequences(path)
    if any(actual.get(folded) != sequences[original] for original, folded in generator.chains.refold_map.items()):
        raise ValueError("ProteinMPNN structure realization did not preserve selected sequences")
    atoms = structure(path)
    original_ids = atoms.chain_id.copy()
    for original, folded in generator.chains.refold_map.items():
        atoms.chain_id[original_ids == folded] = original
    from biotite.structure.io import save_structure

    output = directory / "realized.pdb"
    save_structure(str(output), atoms)
    return output


async def optimize_native(runner, decision, index):
    from .runner import digest

    candidates = runner.checkpoint.data["candidates"]
    action = decision["action"]
    stage = f"round-{index}/{action}"
    retained = [c for c in candidates if c.get("generation_stage") == stage]
    if retained:
        return retained
    parent = next(c for c in candidates if c["candidate_id"] == decision["parent_id"])
    generator = runner.config.generators[parent["source"]]
    kind = "rfd4" if action == "rfd4_partial" else "proteinmpnn"
    result = await runner.job(
        kind,
        f"optimize/{stage}",
        "/v1/tasks" if kind == "rfd4" else "/v1/designs",
        "task_id",
        decision["request"],
    )
    errors = []
    admitted = []
    if kind == "rfd4":
        paths = rfd4_structures(runner, result, runner.root / stage / "artifacts")
        proposals = [(path, chain_sequences(path)) for path in paths]
    else:
        result = normalize_model_results(kind, result)
        fasta = runner.files.local(result["fasta_path"])
        records = parse_fasta(fasta.read_text())
        proposals = []
        # Native ProteinMPNN reports designed chains in sorted chain-ID order.
        designed = sorted(decision["request"]["design_chains"])
        for record in records:
            if "sample=" not in record["header"] and "sample =" not in record["header"]:
                continue  # The first FASTA record is the native sequence, not a design.
            parts = record["sequence"].split("/")
            if len(parts) != len(designed):
                errors.append("ProteinMPNN FASTA designed-chain count mismatch")
                continue
            proposals.append((None, {**parent["chains"], **dict(zip(designed, parts))}))
    for path, sequences in proposals:
        try:
            validate_sequences(parent, sequences, generator.chains, decision["mutation_constraints"])
        except ValueError as exc:
            errors.append(str(exc))
            continue
        admitted.append((path, sequences))
    if not admitted:
        raise ValueError(f"no {kind} output preserves task/selector constraints: {errors}")
    # One action produces one admitted descendant; native artifacts stay in its job record.
    path, sequences = admitted[0]
    candidate_id = f"{action}-" + hashlib.sha256(f"{stage}/{parent['candidate_id']}".encode()).hexdigest()[:16]
    if path is None:
        path = await materialize_sequence(runner, parent, sequences, generator, candidate_id)
    snapshot = runner.root / candidate_id / "generated.pdb"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, snapshot)
    child = {
        "candidate_id": candidate_id,
        "source": parent["source"],
        "optimized_by": action,
        "parent_id": parent["candidate_id"],
        "round_index": index,
        "generation_stage": stage,
        "generation_request": parent.get("generation_request", generator.request),
        "chains": sequences,
        "sequence": "|".join(sequences[c] for c in generator.chains.binder),
        "generated_structure": runner.files.remote(snapshot),
        "generated_sha256": digest(snapshot),
        "metrics": {},
        "status": "generated",
        "mutation_constraints": decision["mutation_constraints"],
    }
    candidates.append(child)
    runner.save_candidates()
    return [child]
