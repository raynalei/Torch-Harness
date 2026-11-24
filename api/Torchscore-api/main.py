"""TorchScore API and checked Slurm worker for Mingchenchen/TorchX."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

ROOT = Path(os.getenv("TORCHSCORE_ROOT", "/lustre/grp/cmclab/leiy/TorchX")).resolve()
WORK = Path(os.getenv("TORCHSCORE_API_WORK", "/lustre/grp/cmclab/leiy/torchscore_api_jobs")).resolve()
PYTHON = os.getenv("TORCHSCORE_PYTHON", sys.executable)
MODEL_DIR = os.getenv("TORCHSCORE_MODEL_DIR", "")
CHECKPOINT = os.getenv("TORCHSCORE_CHECKPOINT", "")
ALLOWED_ROOTS = tuple(
    Path(p).resolve() for p in os.getenv("TORCHSCORE_ALLOWED_INPUT_ROOTS", "/lustre/grp/cmclab").split(":")
)
SOURCE = ROOT / "torchscore_gpu" / "src"
SCRIPTS = SOURCE / "pipeline_scripts"
API_FILE = Path(__file__).resolve()
log = logging.getLogger("uvicorn.error")
app = FastAPI(title="TorchScore Slurm API", version="1.0.0")


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["pdb", "metrics"] = "pdb"
    input_pdb_dir: str
    # Compatibility wire field retained by the native Harness TorchScore schema.
    af3score_output_dir: str | None = None
    run_data_pipeline: StrictBool = False
    run_inference: StrictBool = True
    init_guess: StrictBool = True
    num_samples: Literal[1] = 1
    seed: Literal[10] = 10
    num_workers: int = Field(default=4, ge=1, le=64)
    write_fold_input_json_file: StrictBool = False

    @model_validator(mode="after")
    def supported_mode(self):
        if self.run_data_pipeline or not self.init_guess or self.write_fold_input_json_file:
            raise ValueError(
                "TorchScore requires run_data_pipeline=false, init_guess=true and write_fold_input_json_file=false"
            )
        if self.mode == "pdb" and (not self.run_inference or self.af3score_output_dir is not None):
            raise ValueError("pdb mode requires inference and no existing output directory")
        if self.mode == "metrics":
            if not self.af3score_output_dir:
                raise ValueError("metrics mode requires af3score_output_dir (TorchScore raw outputs)")
            self.run_inference = False
        return self


def atomic_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def permitted(value: str, *, directory: bool = True) -> Path:
    path = Path(value).resolve(strict=True)
    if not any(path.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise ValueError(f"path outside TORCHSCORE_ALLOWED_INPUT_ROOTS: {value}")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"wrong path type: {value}")
    return path


def task_dir(task_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", task_id):
        raise HTTPException(400, "invalid task_id")
    folder = WORK / task_id
    if not (folder / "metadata.json").is_file():
        raise HTTPException(404, "task not found")
    return folder


def metadata(folder: Path) -> dict:
    return json.loads((folder / "metadata.json").read_text())


def run_command(args: list[str], logfile: Path, *, cpu: bool = False) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    if cpu:
        env["JAX_PLATFORMS"] = "cpu"
    with logfile.open("ab") as handle:
        handle.write((shlex.join(args) + "\n").encode())
        handle.flush()
        subprocess.run(
            args,
            cwd=SOURCE,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def read_metrics(folder: Path) -> list[dict]:
    meta = metadata(folder)
    with (folder / "torchscore_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = set(meta["inputs"])
    names = [row.get("description") for row in rows]
    if len(names) != len(expected) or set(names) != expected:
        raise ValueError(f"incomplete/duplicate metrics: expected {sorted(expected)}, got {names}")
    records = []
    raw_dir = Path(meta["raw_output_dir"])
    for row in rows:
        summary_path = raw_dir / row["description"] / "seed-10_sample-0" / "summary_confidences.json"
        summary = json.loads(summary_path.read_text())
        if not isinstance(summary, dict):
            raise ValueError(f"invalid summary object: {summary_path}")
        confidence = summary_path.with_name("confidences.json")
        if not isinstance(json.loads(confidence.read_text()), dict):
            raise ValueError(f"invalid confidences: {confidence}")
        record = {"description": row["description"]}
        for key, value in row.items():
            if key == "description":
                continue
            number = float(value) if value else None
            record[key] = number if number is not None and math.isfinite(number) else None
        for key in ("iptm", "ptm"):
            value = summary.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"missing/invalid {key} in {summary_path}")
            if record.get(key) is None or not math.isclose(record[key], value, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"CSV {key} differs from raw summary: {summary_path}")
        records.append(record)
    return records


def worker(folder: Path) -> int:
    try:
        meta = metadata(folder)
        req = ScoreRequest.model_validate(meta["request"])
        workers = min(req.num_workers, int(os.getenv("SLURM_CPUS_PER_TASK", str(req.num_workers))))
        workers = max(1, workers)
        inputs = folder / "inputs"
        logfile = folder / "worker.log"
        if req.mode == "pdb":
            batches = folder / "tf_input_batch"
            run_command(
                [
                    PYTHON,
                    str(SCRIPTS / "01_prepare_get_json.py"),
                    "--input_dir",
                    str(inputs),
                    "--output_dir_cif",
                    str(folder / "single_chain_cif"),
                    "--save_csv",
                    str(folder / "single_seq.csv"),
                    "--output_dir_json",
                    str(folder / "json"),
                    "--batch_dir",
                    str(batches),
                    "--num_jobs",
                    "1",
                    "--num_workers",
                    str(workers),
                ],
                logfile,
                cpu=True,
            )
            found = set()
            for batch in sorted((batches / "pdb").glob("batch_*")):
                h5_dir = batches / "jax" / batch.name
                run_command(
                    [
                        PYTHON,
                        str(SCRIPTS / "02_prepare_pdb2jax.py"),
                        "--pdb_folder",
                        str(batch),
                        "--output_folder",
                        str(h5_dir),
                        "--num_workers",
                        str(workers),
                    ],
                    logfile,
                    cpu=True,
                )
                for pdb in sorted(batch.glob("*.pdb")):
                    name = pdb.stem
                    json_path = batches / "json" / batch.name / f"{name}.json"
                    h5_path = h5_dir / f"{name}.h5"
                    if name in found or not json_path.is_file() or not h5_path.is_file():
                        raise ValueError(f"duplicate or missing prepared input for {name}")
                    found.add(name)
                    command = [
                        PYTHON,
                        str(SOURCE / "run_torchscore.py"),
                        "--run_data_pipeline=false",
                        "--run_inference=true",
                        "--init_guess=true",
                        "--num_diffusion_samples=1",
                        f"--json_path={json_path}",
                        f"--path={h5_path}",
                        f"--output_dir={meta['raw_output_dir']}",
                    ]
                    command.append(f"--checkpoint_path={CHECKPOINT}" if CHECKPOINT else f"--model_dir={MODEL_DIR}")
                    run_command(command, logfile)
            if found != set(meta["inputs"]):
                raise ValueError(f"preprocessing lost inputs: expected {meta['inputs']}, got {sorted(found)}")
        run_command(
            [
                PYTHON,
                str(SCRIPTS / "03_get_metrics.py"),
                "--input_pdb_dir",
                str(inputs),
                "--tfscore_output_dir",
                meta["raw_output_dir"],
                "--save_metric_csv",
                str(folder / "torchscore_metrics.csv"),
                "--num_workers",
                str(workers),
            ],
            logfile,
            cpu=True,
        )
        records = read_metrics(folder)
        atomic_json(folder / "result.json", {"status": "COMPLETED", "metrics": records})
        rc = 0
    except Exception as exc:
        log.exception("TorchScore worker failed")
        atomic_json(
            folder / "result.json",
            {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"},
        )
        rc = 1
    (folder / "exit_code.txt").write_text(str(rc) + "\n")
    return rc


def submit_slurm(task_id: str) -> None:
    folder = task_dir(task_id)
    meta = metadata(folder)
    meta["submission_status"] = "SUBMITTING"
    atomic_json(folder / "metadata.json", meta)
    args = [
        "sbatch",
        "--parsable",
        "--job-name",
        "torchscore-" + task_id[:8],
        "--partition",
        os.getenv("TORCHSCORE_PARTITION", "gpu31,gpu2,gpu3-12,gpu37,gpu39"),
        "--cpus-per-task",
        os.getenv("TORCHSCORE_CPUS", "8"),
        "--mem",
        os.getenv("TORCHSCORE_MEM", "64G"),
        "--time",
        os.getenv("TORCHSCORE_TIME", "12:00:00"),
        "--output",
        str(folder / "slurm-%j.out"),
        "--error",
        str(folder / "slurm-%j.err"),
    ]
    gres = os.getenv("TORCHSCORE_GRES", "gpu:1") if meta["request"]["mode"] == "pdb" else ""
    if gres:
        args.extend(["--gres", gres])
    args.append(str(folder / "run.sh"))
    env = os.environ.copy()
    env.pop("SBATCH_WAIT", None)
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, env=env)
        if result.returncode:
            meta["submission_status"] = "FAILED"
            raise RuntimeError(result.stderr.strip() or "sbatch rejected")
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdecimal():
            raise RuntimeError(f"invalid sbatch response: {result.stdout}")
        meta.update(slurm_job_id=job_id, submission_status="SUBMITTED")
    except OSError as exc:
        meta.update(submission_status="FAILED", error=str(exc))
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        if meta["submission_status"] != "FAILED":
            meta["submission_status"] = "SUBMISSION_UNCERTAIN"
        meta["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(folder / "metadata.json", meta)


def scheduler_status(job_id: str) -> tuple[str, str | None]:
    for args in (
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        ["sacct", "-n", "-P", "-X", "-j", job_id, "-o", "State,ExitCode"],
    ):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=8)
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().splitlines()[0].split("|")
                return parts[0].split()[0].rstrip("+"), (parts[1].strip() if len(parts) > 1 else None)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return "UNKNOWN", None


def status(folder: Path) -> dict:
    meta = metadata(folder)
    state, code = meta["submission_status"], None
    if meta.get("slurm_job_id"):
        state, code = scheduler_status(meta["slurm_job_id"])
    error = meta.get("error")
    terminal_failures = {
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
    }
    if state not in terminal_failures:
        result_file = folder / "result.json"
        result = json.loads(result_file.read_text()) if result_file.is_file() else {}
        exit_file = folder / "exit_code.txt"
        exit_code = exit_file.read_text().strip() if exit_file.is_file() else None
        if result.get("status") == "FAILED" or (exit_code is not None and exit_code != "0"):
            state, code, error = (
                "FAILED",
                f"{exit_code or '1'}:0",
                result.get("error", "worker failed"),
            )
        elif state == "COMPLETED" and code not in {"0:0", "0"}:
            state, error = "FAILED", "Slurm completion lacks a successful exit code"
        elif result.get("status") == "COMPLETED" and exit_code == "0":
            try:
                read_metrics(folder)
                state, code = "COMPLETED", "0:0"
            except (OSError, ValueError, KeyError, TypeError) as exc:
                state, error = "FAILED", f"invalid scoring results: {exc}"
        elif state == "COMPLETED":
            state, error = "FAILED", "Slurm completed without validated worker results"
    return {
        "task_id": meta["task_id"],
        "slurm_job_id": meta.get("slurm_job_id"),
        "status": state,
        "state": state,
        "exit_code": code,
        "error": error,
        "submission_error": error,
        "output_dir": str(folder),
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "torchscore",
        "execution": "slurm",
        "gpu_validated": False,
    }


@app.post("/v1/tasks", status_code=202)
async def submit(
    req: ScoreRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    if idempotency_key and not re.fullmatch(r"[0-9a-f]{32}", idempotency_key):
        raise HTTPException(400, "Idempotency-Key must be 32 lowercase hexadecimal characters")
    task_id = idempotency_key or uuid.uuid4().hex
    signature = hashlib.sha256(json.dumps(await request.json(), sort_keys=True).encode()).hexdigest()
    WORK.mkdir(parents=True, exist_ok=True)
    folder = WORK / task_id
    try:
        folder.mkdir(mode=0o700)
    except FileExistsError:
        if not (folder / "metadata.json").is_file():
            raise HTTPException(409, "task preparation in progress; query the same ID shortly")
        meta = metadata(folder)
        if meta["request_hash"] != signature:
            raise HTTPException(409, "Idempotency-Key already used for another request")
        return {
            "task_id": task_id,
            "status": meta["submission_status"],
            "slurm_job_id": meta.get("slurm_job_id"),
        }
    try:
        source = permitted(req.input_pdb_dir)
        pdbs = sorted(source.glob("*.pdb"))
        if not pdbs:
            raise ValueError("input_pdb_dir contains no .pdb files")
        for path in pdbs:
            permitted(str(path), directory=False)
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", path.stem):
                raise ValueError(f"unsupported PDB filename: {path.name}")
        if not Path(PYTHON).is_file():
            raise ValueError(f"TORCHSCORE_PYTHON does not exist: {PYTHON}")
        required = [SCRIPTS / "03_get_metrics.py"]
        if req.mode == "pdb":
            required += [
                SOURCE / "run_torchscore.py",
                SCRIPTS / "01_prepare_get_json.py",
                SCRIPTS / "02_prepare_pdb2jax.py",
            ]
            if bool(MODEL_DIR) == bool(CHECKPOINT):
                raise ValueError("set exactly one of TORCHSCORE_MODEL_DIR / TORCHSCORE_CHECKPOINT")
            permitted(MODEL_DIR) if MODEL_DIR else permitted(CHECKPOINT, directory=False)
        if any(not path.is_file() for path in required):
            raise ValueError(f"TorchScore entry scripts missing under {SOURCE}")
        raw_output = permitted(req.af3score_output_dir) if req.mode == "metrics" else folder / "torchscore_outputs"
        inputs = folder / "inputs"
        inputs.mkdir()
        for path in pdbs:
            shutil.copyfile(path, inputs / path.name)
        script = (
            "#!/bin/bash\nset -euo pipefail\nexec "
            + shlex.join([PYTHON, str(API_FILE), "--worker", str(folder)])
            + "\n"
        )
        (folder / "run.sh").write_text(script)
        (folder / "run.sh").chmod(0o700)
        meta = {
            "task_id": task_id,
            "slurm_job_id": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": req.model_dump(),
            "request_hash": signature,
            "submission_key": idempotency_key,
            "inputs": [p.stem for p in pdbs],
            "raw_output_dir": str(raw_output),
            "submission_status": "SUBMITTING",
        }
        atomic_json(folder / "metadata.json", meta)
    except (OSError, ValueError) as exc:
        shutil.rmtree(folder)
        raise HTTPException(400, str(exc)) from exc
    background_tasks.add_task(submit_slurm, task_id)
    return {
        "task_id": task_id,
        "slurm_job_id": None,
        "status": "SUBMITTING",
        "state": "SUBMITTING",
    }


@app.get("/v1/tasks/{task_id}")
def get_status(task_id: str):
    return status(task_dir(task_id))


def successful(task_id: str) -> tuple[Path, dict]:
    folder = task_dir(task_id)
    value = status(folder)
    if value["status"] != "COMPLETED":
        raise HTTPException(409, value)
    return folder, value


@app.get("/v1/tasks/{task_id}/metrics")
def get_metrics(task_id: str):
    folder, value = successful(task_id)
    return {**value, "metrics": read_metrics(folder)}


@app.get("/v1/tasks/{task_id}/results")
def get_results(task_id: str):
    folder, value = successful(task_id)
    files = sorted(
        str(p) for p in folder.rglob("*") if p.is_file() and p.suffix in {".json", ".csv", ".pdb", ".cif", ".txt"}
    )
    return {**value, "files": files, "metrics": read_metrics(folder)}


@app.get("/v1/tasks/{task_id}/logs")
def get_logs(task_id: str, tail: int = 100):
    if not 1 <= tail <= 500:
        raise HTTPException(400, "tail must be between 1 and 500")
    folder = task_dir(task_id)
    paths = [
        folder / "worker.log",
        *folder.glob("slurm-*.out"),
        *folder.glob("slurm-*.err"),
    ]
    return {
        "task_id": task_id,
        "logs": {p.name: "\n".join(p.read_text(errors="replace").splitlines()[-tail:]) for p in paths if p.is_file()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, type=Path)
    args = parser.parse_args()
    sys.exit(worker(args.worker))
