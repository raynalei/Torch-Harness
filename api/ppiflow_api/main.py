"""PPIFLOW v3 API: FastAPI -> Slurm -> fixed PPIFLOW CLI entrypoints."""

from __future__ import annotations

import hashlib
import json
import tempfile
import os
import re
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import logging

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger("uvicorn.error")
ROOT = Path(
    os.environ.get("PPIFLOW_ROOT", "/lustre/grp/cmclab/leiy/v3_basic")
).resolve()
BINDER_CWD = Path(
    os.environ.get("PPIFLOW_BINDER_CWD", "/lustre/grp/cmclab/share/changdr/Projects/v34_binder_0919")
).resolve()

WORK = Path(
    os.environ.get("PPIFLOW_API_WORK", "/lustre/grp/cmclab/leiy/ppiflow_api_jobs")
).resolve()
PYTHON = os.environ.get("PPIFLOW_PYTHON", sys.executable)
PARTITION = os.environ.get("PPIFLOW_PARTITION", "gpu3-12")
GRES = os.environ.get("PPIFLOW_GRES", "gpu:1")
CPUS = int(os.environ.get("PPIFLOW_CPUS", "8"))
MEM = os.environ.get("PPIFLOW_MEM", "32G")
TIME = os.environ.get("PPIFLOW_TIME", "12:00:00")
AB_WEIGHTS = os.environ.get(
    "PPIFLOW_AB_WEIGHTS",
    "/lustre/grp/cmclab/share/changdr/model/20260618_distill/model_214.ckpt.enc",
)
BINDER_WEIGHTS = os.environ.get(
    "PPIFLOW_BINDER_WEIGHTS",
    "/lustre/grp/cmclab/share/changdr/Projects/v34_binder_0919/experiments/results/binder_ppi_monomer/20251013/17_58_00/ckpt/epoch=102-step=92700.ckpt",
)

# CLI flags confirmed from the supplied examples; no speculative flags are forwarded.
MODES = {
    "antibody": {
        "cwd": "antibody_inference",
        "pkg": "ppiflowV3_1017",
        "module": "sample_antibody",
        "config": "test_antibody_v3.yaml",
        "weight": AB_WEIGHTS,
    },
    "nanobody": {
        "cwd": "antibody_inference",
        "pkg": "ppiflowV3_1017",
        "module": "sample_nanobody",
        "config": "test_antibody.yaml",
        "weight": AB_WEIGHTS,
    },
    "binder": {
        "pkg": "",
        "module": "sample_binder_design",
        "config": "test_ppi_v3_wd.yaml",
        "weight": BINDER_WEIGHTS,
    },
    "partial_antibody": {
        "cwd": "partial_flow_ab",
        "pkg": "ppiflowV3_partial_flow_ab",
        "module": "sample_antibody_partial_flow",
        "config": "test_antibody_v3.yaml",
        "weight": AB_WEIGHTS,
    },
    "partial_nanobody": {
        "cwd": "partial_flow_ab",
        "pkg": "ppiflowV3_partial_flow_ab",
        "module": "sample_nanobody_partial_flow",
        "config": "test_antibody_v3.yaml",
        "weight": AB_WEIGHTS,
    },
}

COMMON_FLAGS = (
    "antigen_pdb",
    "antigen_chain",
    "specified_hotspots",
    "framework_dir",
    "heavy_chain",
    "light_chain",
    "cdr_length",
    "samples_per_target",
    "num_timesteps",
    "complex_pdb",
    "fixed_positions",
    "cdr_position",
    "start_t",
    "input_pdb",
    "target_chain",
    "binder_chain",
    "samples_min_length",
    "samples_max_length",
)
ALLOWED = {
    "antibody": {
        "antigen_pdb",
        "antigen_chain",
        "specified_hotspots",
        "framework_dir",
        "heavy_chain",
        "light_chain",
        "cdr_length",
        "samples_per_target",
        "num_timesteps",
    },
    "nanobody": {
        "antigen_pdb",
        "antigen_chain",
        "specified_hotspots",
        "framework_dir",
        "heavy_chain",
        "cdr_length",
        "samples_per_target",
        "num_timesteps",
    },
    "binder": {
        "input_pdb",
        "target_chain",
        "binder_chain",
        "specified_hotspots",
        "samples_min_length",
        "samples_max_length",
        "samples_per_target",
    },
    "partial_antibody": {
        "complex_pdb",
        "antigen_chain",
        "heavy_chain",
        "light_chain",
        "fixed_positions",
        "cdr_position",
        "specified_hotspots",
        "start_t",
        "samples_per_target",
        "num_timesteps",
    },
    "partial_nanobody": {
        "complex_pdb",
        "antigen_chain",
        "heavy_chain",
        "fixed_positions",
        "cdr_position",
        "specified_hotspots",
        "start_t",
        "samples_per_target",
        "num_timesteps",
    },
}
REQUIRED = {
    "antibody": {
        "antigen_pdb",
        "antigen_chain",
        "framework_dir",
        "heavy_chain",
        "light_chain",
        "cdr_length",
    },
    "nanobody": {"antigen_pdb", "antigen_chain", "framework_dir", "heavy_chain"},
    "binder": {
        "input_pdb",
        "target_chain",
        "binder_chain",
        "samples_min_length",
        "samples_max_length",
    },
    "partial_antibody": {
        "complex_pdb",
        "antigen_chain",
        "heavy_chain",
        "light_chain",
        "fixed_positions",
        "cdr_position",
        "start_t",
    },
    "partial_nanobody": {
        "complex_pdb",
        "antigen_chain",
        "heavy_chain",
        "fixed_positions",
        "cdr_position",
        "start_t",
    },
}


class DesignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal[
        "antibody", "nanobody", "binder", "partial_antibody", "partial_nanobody"
    ]
    name: str = Field(default="target", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    config: str | None = None
    model_weights: str | None = None
    output_dir: str | None = None
    antigen_pdb: str | None = None
    antigen_chain: str | None = None
    specified_hotspots: str | None = None
    framework_dir: str | None = None
    heavy_chain: str | None = None
    light_chain: str | None = None
    cdr_length: str | None = None
    samples_per_target: int = Field(default=5, ge=1, le=1000)
    num_timesteps: int | None = Field(default=None, ge=1, le=2000)
    complex_pdb: str | None = None
    fixed_positions: str | None = None
    cdr_position: str | None = None
    start_t: float | None = Field(default=None, ge=0, le=1)
    input_pdb: str | None = None
    target_chain: str | None = None
    binder_chain: str | None = None
    samples_min_length: int | None = Field(default=None, ge=1)
    samples_max_length: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_mode(self):
        data = self.model_dump(exclude_none=True)
        extra = (
            set(data) - {"mode", "name", "config", "model_weights", "output_dir"}
        ) - ALLOWED[self.mode]
        missing = REQUIRED[self.mode] - set(data)
        if extra or missing:
            raise ValueError(
                f"unsupported fields: {sorted(extra)}; missing fields: {sorted(missing)}"
            )
        if (
            self.samples_min_length is not None
            and self.samples_max_length is not None
            and self.samples_min_length > self.samples_max_length
        ):
            raise ValueError("samples_min_length must be <= samples_max_length")
        return self


app = FastAPI(title="PPIFLOW Model API", version="1.0.0")


def validate_inputs(data: dict) -> None:
    # Example deployment: only allow user-owned/shared dataset storage (adjust for your cluster).
    input_roots = tuple(
        Path(p).resolve()
        for p in os.environ.get(
            "PPIFLOW_ALLOWED_INPUT_ROOTS", "/lustre/grp/cmclab"
        ).split(":")
    )
    for key in (
        "antigen_pdb",
        "input_pdb",
        "complex_pdb",
        "framework_dir",
        "config",
        "model_weights",
    ):
        value = data.get(key)
        if value:
            p = Path(value).resolve(strict=True)
            if not any(p.is_relative_to(root) for root in input_roots):
                raise ValueError(f"{key} outside permitted input roots")
            if key == "framework_dir" and not p.is_dir():
                raise ValueError("framework_dir must be a directory")
            if key != "framework_dir" and not p.is_file():
                raise ValueError(f"{key} must be a file")
    for key in (
        "antigen_chain",
        "heavy_chain",
        "light_chain",
        "binder_chain",
        "target_chain",
    ):
        value = data.get(key)
        if value and not re.fullmatch(r"[A-Za-z0-9,]+", value):
            raise ValueError(f"bad {key}: expected chain IDs")
    for key in ("specified_hotspots", "fixed_positions", "cdr_position"):
        value = data.get(key)
        if value and not re.fullmatch(r"[A-Za-z0-9,-]+", value):
            raise ValueError(f"bad {key}: expected residue ID/range syntax")


def generate_script(req: DesignRequest, job_dir: Path) -> tuple[str, list[str]]:
    spec = MODES[req.mode]
    cwd = BINDER_CWD if req.mode == "binder" else (ROOT / spec["cwd"]).resolve()
    package_path = (cwd / spec["pkg"]).resolve() if spec["pkg"] else cwd
    if req.mode == "binder":
        entrypoint = cwd / f"{spec['module']}.py"
        if not entrypoint.is_file():
            raise ValueError(f"PPIFLOW_BINDER_CWD does not contain the binder entrypoint: {entrypoint}")
        cmd = [PYTHON, str(entrypoint)]
    else:
        cmd = [PYTHON, "-m", f"{spec['pkg']}.{spec['module']}"]
    config = Path(req.config) if req.config else ROOT / "configs" / spec["config"]
    weights = Path(req.model_weights or spec["weight"])
    for path in (cwd, package_path, config, weights):
        if not path.exists():
            raise ValueError(f"missing deployment resource: {path}")
    if not Path(PYTHON).exists():
        raise ValueError(f"PPIFLOW_PYTHON does not exist: {PYTHON}")
    output = job_dir / "outputs"
    if req.output_dir:
        base = Path(req.output_dir).resolve()
        roots = [
            Path(p).resolve()
            for p in os.environ.get(
                "PPIFLOW_ALLOWED_INPUT_ROOTS", "/lustre/grp/cmclab"
            ).split(":")
        ]
        if not any(base.is_relative_to(root) for root in roots):
            raise ValueError("output_dir outside permitted roots")
        output = base / job_dir.name
    output.mkdir(parents=True)
    supplied = req.model_dump(
        exclude_none=True,
        exclude={"mode", "name", "config", "model_weights", "output_dir"},
    )
    for key in COMMON_FLAGS:
        if key in supplied:
            cmd.extend(["--" + key, str(supplied[key])])
    cmd.extend(
        [
            "--config",
            str(config),
            "--model_weights",
            str(weights),
            "--output_dir",
            str(output),
            "--name",
            req.name,
        ]
    )
    script = "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "record_exit() {",
            '  rc=$?; printf "%s\\n" "$rc" > '
            + shlex.quote(str(job_dir / "exit_code.txt")),
            '  exit "$rc"',
            "}",
            "trap record_exit EXIT",
            # Include both the inference cwd and the PPIFlow package directory:
            # the encrypted entrypoint does a bare `import experiments`, and
            # `experiments` is a subpackage of the package dir (no __init__.py),
            # so the package dir itself must be on sys.path.
            "export PYTHONPATH="
            + shlex.quote(str(package_path))
            + ":"
            + shlex.quote(str(cwd))
            + ':"${PYTHONPATH:-}"',
            "cd " + shlex.quote(str(cwd)),
            shlex.join(cmd),
            "",
        ]
    )
    return script, cmd


def job_path(design_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", design_id):
        raise HTTPException(400, "invalid design_id")
    path = WORK / design_id
    if not (path / "metadata.json").is_file():
        raise HTTPException(404, "design not found")
    return path


def read_meta(design_id: str) -> dict:
    return json.loads((job_path(design_id) / "metadata.json").read_text())


def write_meta(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".metadata-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def slurm_status(job_id: str) -> tuple[str, str | None]:
    for command in (
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        ["sacct", "-n", "-P", "-X", "-j", job_id, "-o", "State,ExitCode"],
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=8)
            if result.returncode == 0 and result.stdout.strip():
                fields = result.stdout.strip().splitlines()[0].split("|")
                state = fields[0].split()[0].rstrip("+")
                return state, fields[1].strip() or None if len(fields) > 1 else None
        except (OSError, subprocess.TimeoutExpired):
            pass
    return "UNKNOWN", None


def status_for(meta: dict, directory: Path) -> dict:
    state = meta.get("submission_state", "SUBMITTED")
    code = None
    if meta.get("slurm_job_id"):
        state, code = slurm_status(meta["slurm_job_id"])
    elif state == "SUBMISSION_FAILED":
        state, code = "FAILED", "1:0"
    exit_path = directory / "exit_code.txt"
    if exit_path.is_file():
        try:
            rc = int(exit_path.read_text().strip())
            code = f"{rc}:0"
            # Script completion can precede accounting visibility.
            if state in {"UNKNOWN", "RUNNING", "COMPLETING"}:
                state = "COMPLETED" if rc == 0 else "FAILED"
        except ValueError:
            pass
    return {"state": state, "status": state, "exit_code": code}


def submit_slurm(design_id: str) -> None:
    directory = job_path(design_id)
    path = directory / "metadata.json"
    meta = json.loads(path.read_text())
    # Persist before sbatch; a restart must never resubmit an uncertain job.
    meta["submission_state"] = "SUBMITTING"
    meta["submission_started_at"] = datetime.now(timezone.utc).isoformat()
    write_meta(path, meta)
    args = [
        "sbatch",
        "--parsable",
        "--job-name",
        "ppiflow-" + design_id[:8],
        "--partition",
        PARTITION,
        "--cpus-per-task",
        str(CPUS),
        "--mem",
        MEM,
        "--time",
        TIME,
        "--output",
        str(directory / "slurm-%j.out"),
        "--error",
        str(directory / "slurm-%j.err"),
    ]
    if GRES:
        args.extend(["--gres", GRES])
    args.append(str(directory / "run.sh"))
    env = os.environ.copy()
    # sbatch --wait may also be inherited through the environment.
    env.pop("SBATCH_WAIT", None)
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=30, env=env
        )
        if result.returncode != 0:
            meta["submission_state"] = "SUBMISSION_FAILED"
            raise RuntimeError(result.stderr.strip() or "sbatch failed")
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdecimal():
            raise RuntimeError(f"unexpected sbatch output: {result.stdout}")
        meta.update(slurm_job_id=job_id, submission_state="SUBMITTED")
    except OSError as exc:
        meta.update(submission_state="SUBMISSION_FAILED", submission_error=str(exc))
        logger.exception("PPIFlow sbatch could not start")
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        if meta["submission_state"] != "SUBMISSION_FAILED":
            meta["submission_state"] = "SUBMISSION_UNCERTAIN"
        meta["submission_error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("PPIFlow design submission failed or is uncertain")
    write_meta(path, meta)


@app.get("/health")
def health():
    return {"ok": True, "service": "ppiflow", "idempotency": "Idempotency-Key"}


@app.post("/v1/designs", status_code=202)
async def submit_design(
    req: DesignRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    if idempotency_key is not None and not re.fullmatch(
        r"[0-9a-f]{32}", idempotency_key
    ):
        raise HTTPException(
            400, "Idempotency-Key must be 32 lowercase hexadecimal characters"
        )
    design_id = idempotency_key or uuid.uuid4().hex
    request_hash = hashlib.sha256(
        json.dumps(await request.json(), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    data = req.model_dump(exclude_none=True)
    WORK.mkdir(parents=True, exist_ok=True)
    job_dir = WORK / design_id
    try:
        job_dir.mkdir(mode=0o700)
    except FileExistsError:
        path = job_dir / "metadata.json"
        if not path.is_file():
            raise HTTPException(
                409, "request is being prepared; query the same design_id shortly"
            )
        meta = json.loads(path.read_text())
        if meta.get("request_hash") != request_hash:
            raise HTTPException(
                409, "Idempotency-Key already used with a different request"
            )
        return {
            "design_id": design_id,
            "slurm_job_id": meta.get("slurm_job_id"),
            "submission_key": meta.get("submission_key"),
            "request_hash": request_hash,
            "state": meta["submission_state"],
            "status": meta["submission_state"],
        }
    try:
        validate_inputs(data)
        script, command = generate_script(req, job_dir)
        script_path = job_dir / "run.sh"
        script_path.write_text(script)
        script_path.chmod(0o700)
        output = command[command.index("--output_dir") + 1]
        meta = {
            "design_id": design_id,
            "slurm_job_id": None,
            "mode": req.mode,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": data,
            "command": command,
            "output_dir": output,
            "submission_state": "SUBMITTING",
            "submission_key": idempotency_key,
            "request_hash": request_hash,
        }
        write_meta(job_dir / "metadata.json", meta)
    except (ValueError, OSError) as exc:
        # No sbatch has run, so validation failures can safely release the key.
        for filename in ("run.sh", "metadata.json"):
            (job_dir / filename).unlink(missing_ok=True)
        outputs = job_dir / "outputs"
        if outputs.is_dir():
            outputs.rmdir()
        job_dir.rmdir()
        raise HTTPException(400, str(exc)) from exc
    background_tasks.add_task(submit_slurm, design_id)
    return {
        "design_id": design_id,
        "slurm_job_id": None,
        "submission_key": idempotency_key,
        "request_hash": request_hash,
        "state": "SUBMITTING",
        "status": "SUBMITTING",
    }


@app.get("/v1/designs/{design_id}")
def get_design(design_id: str):
    meta = read_meta(design_id)
    return {**meta, **status_for(meta, job_path(design_id))}


@app.get("/v1/designs/{design_id}/results")
def get_results(design_id: str):
    meta = read_meta(design_id)
    output = Path(meta["output_dir"])
    pdbs = sorted(str(p) for p in output.rglob("*.pdb") if p.is_file())
    return {
        "design_id": design_id,
        **status_for(meta, job_path(design_id)),
        "output_dir": str(output),
        "files": pdbs,
        "pdb_files": pdbs,
        "count": len(pdbs),
    }
