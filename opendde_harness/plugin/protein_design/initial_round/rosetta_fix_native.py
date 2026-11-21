"""Run the official PPIFlow RosettaFixStep on one candidate without an HTTP API.

The upstream script assumes binder chains A(/B) and target C. Normalizing chain
IDs and order happens on a dedicated copy, never on the candidate structure.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path


def _load(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing upstream PPIFlow module: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream PPIFlow module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_pdb(source: Path, output: Path, binder: list[str], target: list[str]) -> dict[str, str]:
    if len(target) != 1 or len(binder) not in (1, 2):
        raise ValueError("Official RosettaFix requires one target chain and one or two binder chains")
    chains = [*binder, *target]
    if len(set(chains)) != len(chains) or any(len(c) != 1 for c in chains):
        raise ValueError("All input chains must have distinct one-character IDs")
    normalized = dict(zip(chains, ["A", "B", "C"] if len(binder) == 2 else ["A", "C"]))
    lines = source.read_text().splitlines(keepends=True)
    atom_lines = {key: [] for key in chains}
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27:
            chain = line[21]
            if chain in normalized:
                atom_lines[chain].append(line[:21] + normalized[chain] + line[22:])
    if any(not atom_lines[chain] for chain in chains):
        raise ValueError("Generated PDB does not contain every target/binder chain")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for chain in chains:
            stream.writelines(atom_lines[chain])
            stream.write("TER\n")
        stream.write("END\n")
    return normalized


def execute(request: dict) -> dict:
    source_dir = Path(request["source_dir"]).resolve(strict=True)
    binary = Path(request["rosetta_bin"]).resolve(strict=True)
    class_module = _load(source_dir / "steps/rosetta_fix_step.py", "ppiflow_rosetta_fix_official")
    energy_module = _load(source_dir / "steps/rosetta/1get_interface_energy.py", "ppiflow_interface_energy_official")
    input_pdb = Path(request["input_pdb"]).resolve(strict=True)
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    binders = request["binder_chains"]
    targets = request["target_chains"]
    canonical = output_dir / "input" / "candidate.pdb"
    mapping = canonical_pdb(input_pdb, canonical, binders, targets)
    xml = request.get("native_xml")
    step = class_module.RosettaFixStep(
        gentype="antibody" if len(binders) == 2 else "nanobody",
        output_dir=str(output_dir / "official"),
        rosetta_bin=str(binary),
        native_xml=str(Path(xml).resolve(strict=True)) if xml else None,
        python=sys.executable,
        interface_dist=float(request["interface_dist"]),
        plot_toggle=False,
        overwrite=False,
    )
    # Upstream _prepare_one runs the same native.xml/FastRelax/ScoreCutoffFilter
    # sequence as RosettaFixStep.run(), without spawning a 32-process batch for
    # a single candidate. Use the upstream interface energy functions directly.
    prepared = step._prepare_one(str(canonical))
    log = Path(prepared["rosetta_out"])
    if not log.is_file() or log.stat().st_size == 0:
        raise RuntimeError(f"Official RosettaFix did not write ResResE output: {log}")
    raw = energy_module.get_interchain_score(str(log))
    if raw is None:
        raise RuntimeError(f"Official RosettaFix returned no residue energy table: {log}")
    energies = {}
    for original in binders:
        canonical_binder = mapping[original]
        canonical_target = mapping[targets[0]]
        pair = raw[(raw["binder_id"] == canonical_binder) & (raw["target_id"] == canonical_target)].copy()
        interface = energy_module.get_residue_pairs_within_distance(
            str(canonical),
            canonical_binder,
            canonical_target,
            distance_threshold=float(request["interface_dist"]),
        )
        # Same official selection: negative inter-chain pair energies, within
        # CB/CA distance, summed per binder residue.
        totals = energy_module.get_interface_energy(pair, interface, canonical_binder, "", False)
        cleaned = {str(int(pos)): float(value) for pos, value in totals.items()}
        if any(not math.isfinite(value) for value in cleaned.values()):
            raise ValueError("Non-finite RosettaFix interface residue energy")
        energies[original] = cleaned
    return {"interface_residue_energies": energies, "energy_log": str(log), "normalized_chains": mapping}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: rosetta_fix_native.py <request.json>")
    request = json.loads(Path(sys.argv[1]).read_text())
    result = execute(request)
    result_path = Path(request["result_path"])
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    temporary.replace(result_path)


if __name__ == "__main__":
    main()
