"""Conda/PyRosetta API: FastAPI -> Slurm -> the same environment's Python.

PyRosetta runs in isolated Slurm child processes, never in API request threads.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

WORK = Path(os.environ.get("ROSETTA_API_WORK", str(Path.cwd() / "jobs"))).resolve()
PARTITION = os.environ.get("ROSETTA_PARTITION", "gpu34,gpu31,gpu3-12,gpu37,gpu39,gpu2")
CPUS = int(os.environ.get("ROSETTA_CPUS", "4"))
MEM = os.environ.get("ROSETTA_MEM", "16G")
TIME = os.environ.get("ROSETTA_TIME", "04:00:00")
ALLOWED = tuple(Path(p).resolve() for p in os.environ.get("ROSETTA_ALLOWED_INPUT_ROOTS", "").split(":") if p)
SUPPORTED_TOOLS = ("interface_analyze", "structure_score")
app = FastAPI(title="Rosetta Biological Tools API", version="6.0.0")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class StructureScore(StrictModel):
    input_pdb: str


class InterfaceAnalyze(StrictModel):
    input_pdb: str
    chain_group_a: str = Field(pattern=r"^[A-Z]{1,26}$")
    chain_group_b: str = Field(pattern=r"^[A-Z]{1,26}$")
    pack_input: bool = False
    pack_separated: bool = True

    @model_validator(mode="after")
    def check_groups(self):
        if set(self.chain_group_a) & set(self.chain_group_b):
            raise ValueError("interface chain groups overlap")
        if any(len(group) != len(set(group)) for group in (self.chain_group_a, self.chain_group_b)):
            raise ValueError("duplicate chain IDs in interface group")
        return self


AA3_TO_1 = dict(
    zip(
        ("ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR").split(),
        "ACDEFGHIKLMNPQRSTVWY",
    )
)


def input_file(filename: str) -> Path:
    src = Path(filename).resolve(strict=True)
    if not src.is_file() or src.suffix.lower() != ".pdb":
        raise ValueError("input_pdb must point to an existing .pdb file")
    if not any(src.is_relative_to(root) for root in ALLOWED):
        raise ValueError("input_pdb outside allowed roots")
    return src


def pdb_chains(pdb: Path) -> set[str]:
    chains = set()
    with pdb.open(errors="replace") as f:
        for line in f:
            if line.startswith("ATOM  ") and len(line) >= 22:
                chains.add(line[21].strip())
    return chains


def scorefile_rows(path: Path, *, strict: bool = False) -> list[dict]:
    if not path.is_file():
        return []
    header = None
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("SCORE:"):
            continue
        values = line.split()[1:]
        if "description" in values or "total_score" in values and header is None:
            header = values
            continue
        if header and len(values) == len(header):
            row = {}
            for k, v in zip(header, values):
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
        elif strict:
            raise ValueError("malformed Rosetta SCORE row or missing header")
    return rows


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


def normalize_residue_energies(raw: Path, output: Path, src: Path, group_a: str, group_b: str) -> int:
    """Convert Rosetta residue_energy_breakdown's weighted pair totals to ResResE.

    The official binary emits a silent scorefile with pdbid1/pdbid2 (e.g. 42B),
    restype1/restype2 and total columns, including one-body rows. The uploaded
    1get_interface_energy.py expects NAME_<ASCII chain><PDB number> instead.
    """
    residues = pdb_residues(src)
    if set(group_a + group_b) != {chain for chain, _ in residues}:
        raise ValueError("residue energy input chains do not match the requested complex")
    if set(group_a) & set(group_b):
        raise ValueError("interface groups overlap")
    rows = scorefile_rows(raw, strict=True)
    if not rows:
        raise ValueError("Rosetta residue-energy scorefile is missing or empty")
    pairs, poses = {}, set()
    for row in rows:
        if not {"pdbid1", "pdbid2", "restype1", "restype2", "total", "pose_id"} <= row.keys():
            raise ValueError("unsupported residue_energy_breakdown scorefile header")
        poses.add(str(row["pose_id"]))
        if row["pdbid2"] == "--" and row["restype2"] == "onebody":
            continue
        keys = []
        for index in (1, 2):
            match = re.fullmatch(r"([1-9][0-9]*)([A-Z])", str(row[f"pdbid{index}"]))
            if match is None:
                raise ValueError("unsupported PDB ID in residue-energy scorefile")
            key = (match[2], int(match[1]))
            if residues.get(key) != row[f"restype{index}"]:
                raise ValueError("residue-energy scorefile does not match the input PDB")
            keys.append(key)
        value = row["total"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("non-finite residue pair energy")
        if keys[0] == keys[1]:
            raise ValueError("self pair in residue-energy scorefile")
        pair = tuple(sorted(keys))
        if pair in pairs and pairs[pair] != value:
            raise ValueError("contradictory duplicate residue pair energies")
        pairs[pair] = value
    if len(poses) != 1 or not pairs:
        raise ValueError("expected exactly one pose with residue pair energies")
    lines = ["ResResE Res1 Res2 total", "ResResE ---- ---- -----"]
    for (first, second), value in sorted(pairs.items()):
        # Preserve the uploaded script's Res1=binder/group_a convention.
        if first[0] in group_b and second[0] in group_a:
            first, second = second, first

        def label(key):
            chain, number = key
            return f"{residues[key]}_{ord(chain)}{number}"

        lines.append(f"ResResE {label(first)} {label(second)} {value:.12g}")
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output)
    return len(pairs)


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


def run_task(metadata_file: Path):
    """Worker CLI; a nonzero exception propagates to Slurm's ExitCode."""
    meta = json.loads(metadata_file.read_text())
    src, output = metadata_file.parent / "input.pdb", Path(meta["output_dir"])
    pyrosetta, scorefxn, long_range_type = initialize_pyrosetta()
    pose = pyrosetta.pose_from_pdb(str(src))
    total = float(scorefxn(pose))
    if not math.isfinite(total):
        raise ValueError("non-finite PyRosetta total score")
    if meta["tool"] == "interface_analyze":
        req = InterfaceAnalyze.model_validate(meta["request"])
        # Pair energies are computed BEFORE InterfaceAnalyzer can pack/separate.
        write_pyrosetta_pairs(
            pyrosetta,
            scorefxn,
            long_range_type,
            pose,
            src,
            output / "residue_energies.out",
            req.chain_group_a,
            req.chain_group_b,
        )
        mover = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover()
        mover.set_interface(req.chain_group_a + "_" + req.chain_group_b)
        mover.set_scorefunction(scorefxn)
        mover.set_pack_input(req.pack_input)
        mover.set_pack_separated(req.pack_separated)
        mover.set_compute_packstat(False)
        mover.apply(pose.clone())
        interface_dg = float(mover.get_interface_dG())
        if not math.isfinite(interface_dg):
            raise ValueError("non-finite PyRosetta interface dG")
        (output / "interface.sc").write_text(
            f"SCORE: dG_separated total_score description\nSCORE: {interface_dg:.12g} {total:.12g} input\n"
        )
    elif meta["tool"] == "structure_score":
        (output / "score.sc").write_text(f"SCORE: total_score description\nSCORE: {total:.12g} input\n")
    else:
        raise ValueError("unsupported PyRosetta worker operation")


def check_pyrosetta_backend():
    """Run the actual interface/pair binding path on a tiny two-chain pose."""
    with tempfile.TemporaryDirectory(prefix="rosetta-api-check-") as directory:
        root = Path(directory)
        rows, serial = [], 1
        for chain, residue, y in (("A", "GLY", 0.0), ("B", "ALA", 4.0)):
            for atom, x, z, element in (
                ("N", 0.0, 0.0, "N"),
                ("CA", 1.45, 0.0, "C"),
                ("C", 2.0, 1.4, "C"),
                ("O", 1.5, 2.4, "O"),
            ):
                rows.append(
                    f"ATOM  {serial:5d} {atom:^4s} {residue} {chain}{1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00          {element:>2s}"
                )
                serial += 1
            rows.append("TER")
        (root / "input.pdb").write_text("\n".join(rows) + "\nEND\n")
        outputs = root / "outputs"
        outputs.mkdir()
        metadata = root / "worker-input.json"
        metadata.write_text(
            json.dumps(
                {
                    "tool": "interface_analyze",
                    "output_dir": str(outputs),
                    "request": {
                        "input_pdb": str(root / "input.pdb"),
                        "chain_group_a": "B",
                        "chain_group_b": "A",
                        "pack_input": False,
                        "pack_separated": False,
                    },
                }
            )
        )
        run_task(metadata)
        if not (outputs / "residue_energies.out").is_file() or not (outputs / "interface.sc").is_file():
            raise RuntimeError("PyRosetta interface probe did not produce its outputs")


@lru_cache(maxsize=1)
def deployment_checks():
    """Import/init PyRosetta in a separate process, outside FastAPI threads."""
    try:
        checked = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--check-pyrosetta"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        ok = checked.returncode == 0
        return {
            "backend": "pyrosetta",
            "ok": ok,
            "python": sys.executable,
            "supported_tools": list(SUPPORTED_TOOLS),
            "error": None if ok else (checked.stderr or checked.stdout)[-2500:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"backend": "pyrosetta", "ok": False, "error": str(exc)}


def scheduler_command(name):
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(f"required Slurm command is not on PATH: {name}")
    return executable


def submit_tool(tool, req):
    try:
        if importlib.util.find_spec("pyrosetta") is None:
            raise RuntimeError("PyRosetta is not installed in this API's Python environment")
        src = input_file(req.input_pdb)
        if tool == "interface_analyze":
            residues = pdb_residues(src)
            if set(req.chain_group_a + req.chain_group_b) != {chain for chain, _ in residues}:
                raise ValueError("interface groups must cover exactly the input protein chains")
        task_id = uuid.uuid4().hex
        task = WORK / task_id
        task.mkdir(parents=True, mode=0o700, exist_ok=False)
        output = task / "outputs"
        output.mkdir()
        shutil.copy2(src, task / "input.pdb")
        meta = {
            "task_id": task_id,
            "tool": tool,
            "request": req.model_dump(),
            "backend": "pyrosetta",
            "output_dir": str(output),
            "score_weights": "ref2015",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # Worker reads this immutable input record; publish before submitting.
        worker_input = task / "worker-input.json"
        worker_input.write_text(json.dumps(meta, indent=2))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-task",
            str(worker_input),
        ]
        script = task / "run.sh"
        script.write_text(
            "#!/bin/bash\nset -euo pipefail\ncd " + shlex.quote(str(task)) + "\nexec " + shlex.join(command) + "\n"
        )
        script.chmod(0o700)
        submitted = subprocess.run(
            [
                scheduler_command("sbatch"),
                "--parsable",
                "--export=ALL",
                "--job-name",
                "ros-" + tool[:8] + "-" + task_id[:6],
                "--partition",
                PARTITION,
                "--cpus-per-task",
                str(CPUS),
                "--mem",
                MEM,
                "--time",
                TIME,
                "--output",
                str(task / "slurm-%j.out"),
                "--error",
                str(task / "slurm-%j.err"),
                str(script),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if submitted.returncode:
            raise RuntimeError(submitted.stderr.strip() or "sbatch failed")
        job_id = submitted.stdout.strip().split(";")[0]
        if not job_id.isdecimal():
            raise RuntimeError("unexpected sbatch output: " + submitted.stdout)
        meta.update(slurm_job_id=job_id, command=command)
        (task / "metadata.json").write_text(json.dumps(meta, indent=2))
        return {
            "task_id": task_id,
            "tool": tool,
            "slurm_job_id": job_id,
            "state": "PENDING",
            "backend": "pyrosetta",
        }
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/health")
def health():
    return deployment_checks()


@app.post("/v1/tools/interface_analyze", status_code=202)
def interface_analyze(req: InterfaceAnalyze):
    return submit_tool("interface_analyze", req)


@app.post("/v1/tools/structure_score", status_code=202)
def structure_score(req: StructureScore):
    return submit_tool("structure_score", req)


@app.post("/v1/tools/structure_relax")
@app.post("/v1/tools/constrained_relax")
@app.post("/v1/tools/interface_local_relax")
@app.post("/v1/tools/mutation_binding_ddg")
def unsupported_tool():
    raise HTTPException(
        501,
        "This PyRosetta adapter supports interface_analyze and structure_score; advanced relax/flex-ddG protocols are not implemented",
    )


def read_meta(task_id: str) -> dict:
    if not re.fullmatch("[0-9a-f]{32}", task_id):
        raise HTTPException(400, "invalid task_id")
    path = WORK / task_id / "metadata.json"
    if not path.is_file():
        raise HTTPException(404, "task not found")
    return json.loads(path.read_text())


def job_status(job_id: str) -> dict:
    """Only parent-allocation accounting establishes completion and exit status."""
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("invalid Slurm job ID")

    def response(phase, code=None):
        return {"status": phase, "state": phase, "exit_code": code}

    try:
        active = subprocess.run(
            [scheduler_command("squeue"), "-h", "-j", job_id, "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if active.returncode == 0 and active.stdout.strip():
            phase = active.stdout.strip().splitlines()[0].split()[0].split("+")[0].upper()
            if phase in {
                "PENDING",
                "RUNNING",
                "CONFIGURING",
                "COMPLETING",
                "SUSPENDED",
                "RESIZING",
                "REQUEUED",
            }:
                return response(phase)
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        accounting = subprocess.run(
            [scheduler_command("sacct"), "-n", "-P", "-X", "-j", job_id, "-o", "JobIDRaw,State,ExitCode"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if accounting.returncode == 0:
            rows = [line.strip().split("|") for line in accounting.stdout.splitlines()]
            parents = [row for row in rows if len(row) >= 3 and row[0] == job_id]
            if len(parents) == 1:
                _, phase, code, *_ = parents[0]
                phase = phase.strip().split()[0].split("+")[0].upper() if phase.strip() else "UNKNOWN"
                code = code.strip() if re.fullmatch(r"[0-9]+:[0-9]+", code.strip()) else None
                # Never expose COMPLETED + a failing exit code as successful completion.
                if phase == "COMPLETED":
                    if code is None:
                        phase = "UNKNOWN"  # Accounting is not complete yet.
                    elif code != "0:0":
                        phase = "FAILED"
                return response(phase, code)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return response("UNKNOWN")


def state(job_id: str):
    return job_status(job_id)["status"]


@app.get("/v1/tasks/{task_id}")
def task_status(task_id: str):
    meta = read_meta(task_id)
    return {**meta, **job_status(meta["slurm_job_id"])}


@app.get("/v1/tasks/{task_id}/results")
def task_results(task_id: str):
    meta = read_meta(task_id)
    status = job_status(meta["slurm_job_id"])
    if status["status"] != "COMPLETED" or status["exit_code"] != "0:0":
        raise HTTPException(
            409,
            {
                "message": "results require completed Slurm allocation with ExitCode=0:0",
                **status,
            },
        )
    out = Path(meta["output_dir"])
    scores = sorted(out.rglob("*.sc"))
    csvs = sorted((WORK / task_id).rglob("*.csv"))
    dbs = sorted(out.rglob("*.db3"))
    pdbs = sorted(out.rglob("*.pdb"))
    energies = []
    if meta["tool"] == "interface_analyze":
        expected = out / "residue_energies.out"
        if not expected.is_file() or expected.stat().st_size == 0:
            raise HTTPException(
                500,
                "interface task has no ResResE output; rerun an old task using the updated API",
            )
        energies = [expected]
    # Raw pair tables are available as files; do not inflate JSON with every pair.
    scalar_scores = [p for p in scores if p.name != "residue_pairs.sc"]
    score_rows = {str(p.relative_to(out)): scorefile_rows(p) for p in scalar_scores}
    files = sorted(set([*pdbs, *scores, *csvs, *dbs, *energies]))
    return {
        "task_id": task_id,
        "tool": meta["tool"],
        **status,
        "files": [str(p) for p in files],
        "residue_energy_files": [str(p) for p in energies],
        "pdb_files": [str(p) for p in pdbs],
        "score_files": [str(p) for p in scores],
        "csv_files": [str(p) for p in csvs],
        "sqlite_files": [str(p) for p in dbs],
        "metrics": [row for rows in score_rows.values() for row in rows],
        "score_rows": score_rows,
    }


@app.get("/v1/tasks/{task_id}/logs")
def task_logs(task_id: str, lines: int = Query(100, ge=1, le=500)):
    meta = read_meta(task_id)
    task = WORK / task_id
    jid = meta["slurm_job_id"]

    def tail(p: Path):
        if not p.is_file():
            return ""
        with p.open(errors="replace") as f:
            return "".join(deque(f, maxlen=lines))[-30000:]

    return {
        "task_id": task_id,
        "stdout": tail(task / f"slurm-{jid}.out"),
        "stderr": tail(task / f"slurm-{jid}.err"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conda/PyRosetta API worker")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-task", type=Path)
    mode.add_argument("--check-pyrosetta", action="store_true")
    mode.add_argument("--normalize-residue-energies", nargs=5)
    args = parser.parse_args()
    if args.check_pyrosetta:
        check_pyrosetta_backend()
        print("PyRosetta ref2015 interface and pair scoring verified")
    elif args.run_task:
        run_task(args.run_task)
    else:
        raw, output, src, group_a, group_b = args.normalize_residue_energies
        normalize_residue_energies(Path(raw), Path(output), Path(src), group_a, group_b)
