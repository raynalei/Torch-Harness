"""TorchCraft API: FastAPI -> Slurm -> Conda or Apptainer.

Modes: mini_binder, vhh, monomer_unconditional (TorchCraft CLI task_type=monomer).
No model runs in this API process; GPU inference is submitted via sbatch.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

ROOT = Path(os.getenv("TORCHX_ROOT", "/lustre/grp/cmclab/leiy/TorchX")).resolve()
WORK = Path(os.getenv("TORCHCRAFT_API_WORK", "/lustre/grp/cmclab/leiy/torchcraft_api_jobs")).resolve()
PYTHON = os.getenv("TORCHCRAFT_PYTHON", "/lustre/grp/cmclab/leiy/conda_envs/torchx_env/bin/python")
PARTITION = os.getenv("TORCHCRAFT_PARTITION", "gpu3-12")
GRES = os.getenv("TORCHCRAFT_GRES", "gpu:1")
CPUS = int(os.getenv("TORCHCRAFT_CPUS", "8"))
MEM = os.getenv("TORCHCRAFT_MEM", "32G")
TIME = os.getenv("TORCHCRAFT_TIME", "12:00:00")
SIF = os.getenv("TORCHCRAFT_SIF", "")
CKPT = os.getenv("TORCHCRAFT_CHECKPOINT", "")
MODEL_DIR = os.getenv("TORCHCRAFT_MODEL_DIR", "")
ALLOWED_ROOT = Path(os.getenv("TORCHCRAFT_ALLOWED_ROOT", "/lustre/grp/cmclab")).resolve()
IMAGE_ROOT = "/opt/TorchX"
IMAGE_PYTHON = "/opt/conda/envs/torchx/bin/python"

app = FastAPI(title="TorchCraft Slurm API", version="1.1")


class DesignRequest(BaseModel):
    mode: Literal["mini_binder", "vhh", "monomer_unconditional"]
    name: str = Field(default="target", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    pdb_path: str | None = None
    hotspot_indices: str | None = Field(default=None, pattern=r"^\d+(,\d+)*$")
    target_full_seq: str | None = None
    json_path: str | None = None
    num_designs: int = Field(default=1, ge=1, le=50)
    seed: int = Field(default=1, ge=0, le=2147483647)
    design_length: int | None = Field(default=None, ge=1, le=1000)
    framework_cif_path: str | None = None

    @model_validator(mode="after")
    def check_mode(self):
        if self.seed + self.num_designs - 1 > 2147483647:
            raise ValueError("seed + num_designs - 1 must be <= 2147483647")
        if self.mode == "monomer_unconditional":
            if not self.json_path or self.design_length is None:
                raise ValueError("monomer_unconditional requires json_path and design_length")
            if any(v is not None for v in (self.pdb_path, self.hotspot_indices, self.target_full_seq, self.framework_cif_path)):
                raise ValueError("monomer_unconditional does not accept PDB/binder/framework inputs")
        else:
            if not self.pdb_path or not self.target_full_seq:
                raise ValueError(f"{self.mode} requires pdb_path and target_full_seq")
            # Hotspots are optional in upstream run_task.py; if omitted use empty string.
            if self.json_path is not None:
                raise ValueError(f"{self.mode} does not accept json_path")
            if self.mode == "mini_binder":
                if self.design_length is None:
                    raise ValueError("mini_binder requires design_length")
                if self.framework_cif_path is not None:
                    raise ValueError("mini_binder does not accept framework_cif_path")
            elif self.mode == "vhh":
                if not self.framework_cif_path:
                    raise ValueError("vhh requires framework_cif_path")
                if self.design_length is not None:
                    raise ValueError("vhh does not accept design_length")
        return self


def secure_file(value: str) -> Path:
    p = Path(value).resolve(strict=True)
    if not p.is_file() or not p.is_relative_to(ALLOWED_ROOT):
        raise ValueError(f"not an allowed input file: {value}")
    return p


def secure_model_dir(value: str) -> Path:
    p = Path(value).resolve(strict=True)
    if not p.is_dir() or not p.is_relative_to(ALLOWED_ROOT):
        raise ValueError(f"not an allowed model directory: {value}")
    return p


def run(args: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def job_folder(design_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", design_id):
        raise HTTPException(400, "invalid design_id")
    p = WORK / design_id
    if not (p / "metadata.json").is_file():
        raise HTTPException(404, "design_id not found")
    return p


def metadata(design_id: str) -> dict:
    return json.loads((job_folder(design_id) / "metadata.json").read_text())


def slurm_state(job_id: str) -> str:
    try:
        p = run(["squeue", "-h", "-j", job_id, "-o", "%T"])
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip().splitlines()[0]
        p = run(["sacct", "-n", "-P", "-X", "-j", job_id, "-o", "State"])
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip().splitlines()[0].split("|")[0].split("+")[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "UNKNOWN"


def task_outcomes(folder: Path) -> dict:
    exits = {}
    for path in sorted((folder / "status").glob("*.exit")):
        try:
            exits[path.stem] = int(path.read_text().strip())
        except ValueError:
            continue
    return {"task_exit_codes": exits, "failed_tasks": [k for k, v in exits.items() if v != 0]}


@app.get("/health")
def health():
    return {"ok": True, "service": "torchcraft", "execution": "apptainer" if SIF else "conda"}


@app.post("/v1/designs", status_code=202)
def create_design(req: DesignRequest):
    folder = None
    submitted = False
    try:
        task_root = ROOT / "torchcraft_gpu" / "task"
        run_task = task_root / "run_task.py"
        config_src = task_root / "config"
        if not run_task.is_file() or not (config_src / "base.yaml").is_file():
            raise ValueError("TorchCraft task/run_task.py or config/base.yaml missing")
        task_type = "monomer" if req.mode == "monomer_unconditional" else req.mode
        if not (config_src / f"{task_type}.yaml").is_file():
            raise ValueError(f"TorchCraft config/{task_type}.yaml missing")
        if bool(CKPT) == bool(MODEL_DIR):
            raise ValueError("Set exactly one of TORCHCRAFT_CHECKPOINT or TORCHCRAFT_MODEL_DIR")
        model_path = secure_file(CKPT) if CKPT else secure_model_dir(MODEL_DIR)
        if SIF and not Path(SIF).is_file():
            raise ValueError(f"SIF not found: {SIF}")
        if not SIF and not Path(PYTHON).is_file():
            raise ValueError(f"Python not found: {PYTHON}")

        paths = {}
        if req.mode == "monomer_unconditional":
            paths["json_path"] = str(secure_file(req.json_path))
        else:
            paths["pdb_path"] = str(secure_file(req.pdb_path))
            paths["target_full_seq"] = str(secure_file(req.target_full_seq))
            if req.framework_cif_path:
                paths["framework_cif_path"] = str(secure_file(req.framework_cif_path))

        WORK.mkdir(parents=True, exist_ok=True)
        design_id = uuid.uuid4().hex
        folder = WORK / design_id
        folder.mkdir(mode=0o700)
        (folder / "status").mkdir()
        (folder / "outputs").mkdir()
        config_dir = folder / "config"
        shutil.copytree(config_src, config_dir)
        base_file = config_dir / "base.yaml"
        base = yaml.safe_load(base_file.read_text()) or {}
        base.pop("checkpoint_path", None)
        base.pop("model_dir", None)
        base["checkpoint_path" if CKPT else "model_dir"] = str(model_path)
        base_file.write_text(yaml.safe_dump(base, sort_keys=False))

        python_bin = IMAGE_PYTHON if SIF else PYTHON
        runner = f"{IMAGE_ROOT}/torchcraft_gpu/task/run_task.py" if SIF else str(run_task)
        tasks = []
        script_lines = [
            "#!/usr/bin/env bash", "set -euo pipefail",
            "export PYTHON_BIN=" + shlex.quote(python_bin),
            "cd " + shlex.quote(str(task_root)),
            'echo "Host $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"',
        ]
        for i in range(req.num_designs):
            seed = req.seed + i
            outdir = folder / "outputs" / f"seed_{seed}"
            outdir.mkdir()
            cmd = [runner, f"--outdir={outdir}", f"--task_type={task_type}",
                   f"--config={config_dir}", f"--random_seed={seed}"]
            if req.mode == "monomer_unconditional":
                # The upstream monomer CLI expects a JSON input, not a target PDB.
                cmd += [f"--design_length={req.design_length}", f"--json_path={paths['json_path']}"]
            else:
                cmd += [f"--pdb_path={paths['pdb_path']}",
                        f"--hotspot_indices={req.hotspot_indices or ''}",
                        f"--target_full_seq={paths['target_full_seq']}"]
                if req.mode == "mini_binder":
                    cmd.append(f"--design_length={req.design_length}")
                else:
                    cmd.append(f"--framework_cif_path={paths['framework_cif_path']}")

            if SIF:
                full_cmd = ["apptainer", "exec", "--nv", "--bind", "/lustre:/lustre",
                            SIF, IMAGE_PYTHON, *cmd]
            else:
                full_cmd = [PYTHON, *cmd]
            key = f"seed_{seed}"
            script_lines += [
                f"echo 'Running {key}'", "set +e",
                " ".join(map(shlex.quote, full_cmd))
                + " > " + shlex.quote(str(outdir / "stdout.log"))
                + " 2> " + shlex.quote(str(outdir / "stderr.log")),
                "rc=$?", "set -e",
                "echo \"$rc\" > " + shlex.quote(str(folder / "status" / f"{key}.exit")),
            ]
            tasks.append({"seed": seed, "cmd": full_cmd, "outdir": str(outdir)})
        script_lines += [
            "for f in " + shlex.quote(str(folder / "status")) + '/*.exit; do '
            '[ -f "$f" ] || exit 1; [ "$(cat "$f")" = "0" ] || exit 1; done'
        ]
        script = folder / "run.sh"
        script.write_text("\n".join(script_lines) + "\n")
        script.chmod(0o700)
        args = ["sbatch", "--parsable", "--job-name", f"torchcraft-{design_id[:8]}",
                "--partition", PARTITION, "--gres", GRES, "--cpus-per-task", str(CPUS),
                "--mem", MEM, "--time", TIME,
                "--output", str(folder / "slurm-%j.out"),
                "--error", str(folder / "slurm-%j.err"), str(script)]
        result = run(args, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "sbatch failed")
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdecimal():
            raise RuntimeError(f"unexpected sbatch output: {result.stdout}")
        submitted = True
        meta = {"design_id": design_id, "slurm_job_id": job_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "request": req.model_dump(), "tasks": tasks,
                "output_dir": str(folder / "outputs")}
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2))
        return {"design_id": design_id, "slurm_job_id": job_id, "state": "PENDING"}
    except (ValueError, OSError) as e:
        if folder is not None and not submitted:
            shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, str(e)) from e
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        if folder is not None and not submitted:
            shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(503, str(e)) from e


@app.get("/v1/designs/{design_id}")
def get_design(design_id: str):
    meta = metadata(design_id)
    return {**meta, "state": slurm_state(meta["slurm_job_id"]), **task_outcomes(job_folder(design_id))}


@app.get("/v1/designs/{design_id}/results")
def get_results(design_id: str):
    meta = metadata(design_id)
    folder = job_folder(design_id)
    output = folder / "outputs"
    files = sorted(str(p) for p in output.rglob("*") if p.is_file()
                   and p.suffix.lower() in (".pdb", ".cif", ".fa", ".fasta", ".csv"))
    return {"design_id": design_id, "state": slurm_state(meta["slurm_job_id"]),
            "output_dir": str(output), "files": files, "count": len(files),
            **task_outcomes(folder)}


@app.get("/v1/designs/{design_id}/logs")
def get_logs(design_id: str, seed: int | None = None, tail: int = 100):
    if not 1 <= tail <= 500:
        raise HTTPException(400, "tail must be 1..500")
    folder = job_folder(design_id)
    meta = metadata(design_id)
    if seed is None:
        paths = sorted(folder.glob("slurm-*.err")) + sorted(folder.glob("slurm-*.out"))
    else:
        if seed not in [task["seed"] for task in meta["tasks"]]:
            raise HTTPException(404, "seed not in design")
        paths = [folder / "outputs" / f"seed_{seed}" / name
                 for name in ("stderr.log", "stdout.log")]
    return {"logs": {str(p.relative_to(folder)): "\n".join(p.read_text(errors="replace").splitlines()[-tail:])
                     for p in paths if p.is_file()}}
