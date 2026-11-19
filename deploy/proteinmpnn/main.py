"""Submit ProteinMPNN inference to Slurm and retrieve its FASTA output."""

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path("/lustre/grp/cmclab/leiy")
IMAGE = Path(os.getenv("PROTEINMPNN_SIF", str(ROOT / "containers/proteinmpnn.sif")))
WEIGHTS = Path(os.getenv("PROTEINMPNN_WEIGHTS", str(ROOT / "models/proteinMPNN/v_48_020.pt")))
WORK = Path(os.getenv("PROTEINMPNN_WORK", str(ROOT / "proteinmpnn_api_jobs"))).resolve()
INPUT_ROOTS = tuple(
    Path(root).resolve()
    for root in os.getenv("PROTEINMPNN_INPUT_ROOTS", f"{ROOT}:/lustre/grp/cmclab/share").split(":")
    if root
)
APPTAINER = os.getenv("PROTEINMPNN_APPTAINER", "apptainer")
RUN_SCRIPT = os.getenv("PROTEINMPNN_RUN_SCRIPT", "")
CONTAINER_PYTHON = os.getenv("PROTEINMPNN_CONTAINER_PYTHON", "")
PARTITION = os.getenv("PROTEINMPNN_PARTITION", "gpu51")
GRES = os.getenv("PROTEINMPNN_GRES", "gpu:1")
CPUS = os.getenv("PROTEINMPNN_CPUS", "4")
MEM = os.getenv("PROTEINMPNN_MEM", "16G")
WALLTIME = os.getenv("PROTEINMPNN_TIME", "02:00:00")
FASTA = Path("output/seqs/input.fa")

# The SIF layout can vary. An explicit in-container path takes precedence.
RUN_IN_CONTAINER = r"""
set -eu
script=$1
python_bin=$2
shift 2
if [ -z "$script" ]; then
    script=$(command -v protein_mpnn_run.py || true)
fi
if [ -z "$script" ]; then
    script=$(find / -xdev -name protein_mpnn_run.py -print -quit 2>/dev/null || true)
fi
if [ ! -f "$script" ]; then
    echo 'Cannot locate protein_mpnn_run.py. Set PROTEINMPNN_RUN_SCRIPT.' >&2
    exit 127
fi
if [ -z "$python_bin" ]; then
    python_bin=$(command -v python || command -v python3)
fi
exec "$python_bin" -u "$script" "$@"
"""

app = FastAPI(title="ProteinMPNN API", version="1.0.0")
log = logging.getLogger("uvicorn.error")
SLURM_ENV = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}


class DesignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pdb_path: str
    design_chains: list[str] = Field(min_length=1, max_length=62)
    num_seq_per_target: int = Field(default=8, ge=1, le=512)
    sampling_temp: float = Field(default=0.1, gt=0, le=2)
    seed: int = Field(default=42, ge=1, le=2**32 - 1)


def input_pdb(request: DesignRequest) -> Path:
    path = Path(request.pdb_path)
    if not path.is_absolute() or path.suffix.lower() != ".pdb":
        raise HTTPException(422, "pdb_path must be an absolute path to a .pdb file")
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(422, "PDB file is missing or inaccessible")
    if not any(path.is_relative_to(root) for root in INPUT_ROOTS):
        raise HTTPException(422, "PDB file is outside PROTEINMPNN_INPUT_ROOTS")
    if not path.is_file():
        raise HTTPException(422, "pdb_path must point to a regular file")

    chains = set()
    try:
        with path.open(encoding="ascii", errors="replace") as pdb:
            for line in pdb:
                if line.startswith("ATOM  ") and len(line) > 21:
                    chains.add(line[21])
    except OSError:
        raise HTTPException(422, "PDB file cannot be read")

    requested = set(request.design_chains)
    if len(requested) != len(request.design_chains) or any(
        not re.fullmatch("[A-Za-z0-9]", chain) for chain in requested
    ):
        raise HTTPException(422, "design_chains must contain unique one-character PDB chain IDs")
    if missing := sorted(requested - chains):
        raise HTTPException(422, f"Design chains absent from PDB ATOM records: {', '.join(missing)}")
    return path


def dependencies() -> dict[str, bool]:
    return {
        "image_readable": IMAGE.is_file() and os.access(IMAGE, os.R_OK),
        "weights_readable": WEIGHTS.is_file() and os.access(WEIGHTS, os.R_OK),
        "work_writable": WORK.is_dir() and os.access(WORK, os.W_OK | os.X_OK),
        **{cmd: shutil.which(cmd) is not None for cmd in ("sbatch", "squeue", "sacct", APPTAINER)},
    }


@app.get("/health")
def health():
    checks = dependencies()
    return {
        "service": "proteinmpnn",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "status": "ok" if all(checks.values()) else "degraded",
        "checks": checks,
    }


def write_json(path: Path, data: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@app.post("/v1/designs", status_code=202)
def submit(request: DesignRequest):
    path = input_pdb(request)
    missing = [key for key, available in dependencies().items() if not available]
    if missing:
        raise HTTPException(503, f"Service dependencies unavailable: {', '.join(missing)}")

    task_id = uuid4().hex
    run_dir = WORK / task_id
    run_dir.mkdir(mode=0o700)
    shutil.copyfile(path, run_dir / "input.pdb")
    write_json(run_dir / "request.json", request.model_dump())

    resolved_weights = WEIGHTS.resolve()
    command = [
        APPTAINER,
        "exec",
        "--nv",
        "--bind",
        f"{run_dir}:/mpnn_job",
        "--bind",
        f"{resolved_weights.parent}:/mpnn_weights:ro",
        str(IMAGE.resolve()),
        "bash",
        "-c",
        RUN_IN_CONTAINER,
        "proteinmpnn",
        RUN_SCRIPT,
        CONTAINER_PYTHON,
        "--pdb_path",
        "/mpnn_job/input.pdb",
        "--pdb_path_chains",
        " ".join(request.design_chains),
        "--out_folder",
        "/mpnn_job/output",
        "--use_soluble_model",
        "--path_to_model_weights",
        "/mpnn_weights",
        "--model_name",
        resolved_weights.stem,
        "--num_seq_per_target",
        str(request.num_seq_per_target),
        "--batch_size",
        "1",
        "--sampling_temp",
        str(request.sampling_temp),
        "--seed",
        str(request.seed),
    ]
    # Record success only after the runner exits successfully and writes the expected file.
    wrapper = (
        "set -eu\n"
        + shlex.join(command)
        + "\ntest -s output/seqs/input.fa"
        + "\ntouch completed.tmp\nmv completed.tmp completed\n"
    )
    submit_command = [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--job-name",
        f"mpnn-{task_id[:12]}",
        "--partition",
        PARTITION,
        "--gres",
        GRES,
        "--cpus-per-task",
        CPUS,
        "--mem",
        MEM,
        "--time",
        WALLTIME,
        "--chdir",
        str(run_dir),
        "--output",
        str(run_dir / "slurm.out"),
        "--error",
        str(run_dir / "slurm.err"),
        "--wrap",
        wrapper,
    ]
    try:
        result = subprocess.run(submit_command, env=SLURM_ENV, capture_output=True, text=True, timeout=20, check=True)
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdecimal():
            raise ValueError(f"Unexpected sbatch response: {result.stdout!r}")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        log.exception("Slurm submission error for task %s", task_id)
        error = getattr(exc, "stderr", None)
        detail = error.strip() if isinstance(error, str) and error.strip() else str(exc)
        raise HTTPException(503, {"task_id": task_id, "submission_error": detail}) from exc

    write_json(run_dir / "job.json", {"job_id": job_id})
    return {"task_id": task_id, "job_id": job_id, "status": "SUBMITTED"}


def load_task(task_id: UUID) -> tuple[Path, str]:
    run_dir = WORK / task_id.hex
    record = run_dir / "job.json"
    if not record.is_file():
        raise HTTPException(404, "Task not found")
    return run_dir, json.loads(record.read_text(encoding="utf-8"))["job_id"]


def slurm_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, env=SLURM_ENV, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def task_status(run_dir: Path, job_id: str) -> tuple[str, int | None]:
    if (run_dir / "completed").is_file() and (run_dir / FASTA).is_file():
        return "COMPLETED", 0
    queue = slurm_output(["squeue", "--noheader", "--jobs", job_id, "--format", "%T"])
    if queue:
        return queue.splitlines()[0].strip(), None
    accounting = slurm_output(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job_id,
            "-o",
            "JobIDRaw,State,ExitCode",
        ]
    )
    for row in accounting.splitlines():
        fields = row.split("|")
        if len(fields) < 3 or fields[0] != job_id:
            continue
        state = fields[1].split()[0].rstrip("+") if fields[1].strip() else "UNKNOWN"
        code = fields[2].split(":", 1)[0]
        if state == "COMPLETED":
            if (run_dir / "completed").is_file() and (run_dir / FASTA).is_file():
                return "COMPLETED", 0
            return "FAILED", 1
        return state, int(code) if code.isdecimal() and state not in {"PENDING", "RUNNING"} else None
    return "UNKNOWN", None


@app.get("/v1/tasks/{task_id}")
@app.get("/v1/tasks/{task_id}/results")
def get_task(task_id: UUID):
    run_dir, job_id = load_task(task_id)
    status, exit_code = task_status(run_dir, job_id)
    return {
        "task_id": task_id.hex,
        "job_id": job_id,
        "status": status,
        "exit_code": exit_code,
        "fasta_path": str(run_dir / FASTA) if status == "COMPLETED" else None,
        "stdout_path": str(run_dir / "slurm.out"),
        "stderr_path": str(run_dir / "slurm.err"),
    }


@app.get("/v1/tasks/{task_id}/fasta")
def get_fasta(task_id: UUID):
    run_dir, _ = load_task(task_id)
    if not (run_dir / "completed").is_file() or not (run_dir / FASTA).is_file():
        raise HTTPException(409, "FASTA is not ready")
    return FileResponse(run_dir / FASTA, media_type="text/plain", filename=f"{task_id.hex}.fa")
