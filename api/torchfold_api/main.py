"""TorchFold API: FastAPI -> Slurm -> Apptainer / conda inference."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

ROOT = Path(os.getenv('TORCHFOLD_ROOT', '/lustre/grp/cmclab/leiy/TorchX')).resolve()
WORK = Path(os.getenv('TORCHFOLD_API_WORK', '/lustre/grp/cmclab/leiy/torchfold_api_jobs')).resolve()
ALLOWED_ROOT = Path(os.getenv('TORCHFOLD_ALLOWED_ROOT', '/lustre/grp/cmclab')).resolve()
PYTHON = os.getenv('TORCHFOLD_PYTHON', '/lustre/grp/cmclab/leiy/conda_envs/torchx_env/bin/python')
SIF = os.getenv('TORCHFOLD_SIF', '')
CKPT = os.getenv('TORCHFOLD_CHECKPOINT', '')
MODEL_DIR = os.getenv('TORCHFOLD_MODEL_DIR', '')
DB_DIR = os.getenv('TORCHFOLD_DB_DIR', '')
PARTITION = os.getenv('TORCHFOLD_PARTITION', 'gpu3-13')
GRES = os.getenv('TORCHFOLD_GRES', 'gpu:1')
CPUS = int(os.getenv('TORCHFOLD_CPUS', '8'))
MEM = os.getenv('TORCHFOLD_MEM', '64G')
TIME = os.getenv('TORCHFOLD_TIME', '12:00:00')

app = FastAPI(title='TorchFold Slurm API', version='1.0.0')


class FoldRequest(BaseModel):
    name: str = Field(default='prediction', pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')
    json_path: str = Field(description='Absolute path to an AF3/TorchFold input JSON, visible on shared Lustre')
    run_data_pipeline: bool = False
    num_diffusion_samples: int = Field(default=2, ge=1, le=20)

    @model_validator(mode='after')
    def validate_input(self):
        if not self.json_path.lower().endswith('.json'):
            raise ValueError('json_path must be a .json file')
        return self


def allowed_existing(path: str, *, directory: bool = False) -> Path:
    p = Path(path).resolve(strict=True)
    if not p.is_relative_to(ALLOWED_ROOT):
        raise ValueError(f'path outside TORCHFOLD_ALLOWED_ROOT: {path}')
    if directory and not p.is_dir():
        raise ValueError(f'not a directory: {path}')
    if not directory and not p.is_file():
        raise ValueError(f'not a file: {path}')
    return p


def call(args: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def folder_for(fold_id: str) -> Path:
    if not re.fullmatch(r'[0-9a-f]{32}', fold_id):
        raise HTTPException(400, 'invalid fold_id')
    folder = WORK / fold_id
    if not (folder / 'metadata.json').is_file():
        raise HTTPException(404, 'fold_id not found')
    return folder


def get_meta(fold_id: str) -> dict[str, Any]:
    return json.loads((folder_for(fold_id) / 'metadata.json').read_text())


def slurm_state(job_id: str) -> str:
    try:
        queued = call(['squeue', '-h', '-j', job_id, '-o', '%T'])
        if queued.returncode == 0 and queued.stdout.strip():
            return queued.stdout.strip().splitlines()[0]
        completed = call(['sacct', '-n', '-P', '-X', '-j', job_id, '-o', 'State'])
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip().splitlines()[0].split('|')[0].split('+')[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return 'UNKNOWN'


def exit_code(folder: Path) -> int | None:
    p = folder / 'exit_code.txt'
    if p.is_file():
        try:
            return int(p.read_text().strip())
        except ValueError:
            return None
    return None


@app.get('/health')
def health():
    return {'ok': True, 'service': 'torchfold', 'execution': 'apptainer' if SIF else 'conda'}


@app.post('/v1/folds', status_code=202)
def submit_fold(req: FoldRequest):
    try:
        source = allowed_existing(req.json_path)
        # Lightweight validation; full scientific schema validated inside TorchFold.
        with source.open() as f:
            json.load(f)
        if bool(CKPT) == bool(MODEL_DIR):
            raise ValueError('Set exactly one of TORCHFOLD_CHECKPOINT or TORCHFOLD_MODEL_DIR')
        weight = allowed_existing(CKPT) if CKPT else allowed_existing(MODEL_DIR, directory=True)
        if req.run_data_pipeline:
            if not DB_DIR:
                raise ValueError('TORCHFOLD_DB_DIR required with run_data_pipeline=true')
            db = allowed_existing(DB_DIR, directory=True)
        else:
            db = Path(DB_DIR).resolve() if DB_DIR else ALLOWED_ROOT
        script = ROOT / 'torchfold_gpu' / 'src' / 'run_torchfold.py'
        if not script.is_file():
            raise ValueError(f'TorchFold inference entry missing: {script}')
        if SIF and not Path(SIF).is_file():
            raise ValueError(f'SIF not found: {SIF}')
        if not SIF and not Path(PYTHON).is_file():
            raise ValueError(f'Python executable not found: {PYTHON}')

        WORK.mkdir(parents=True, exist_ok=True)
        fold_id = uuid.uuid4().hex
        folder = WORK / fold_id
        folder.mkdir(mode=0o700)
        outputs = folder / 'outputs'
        outputs.mkdir()
        # Copy input to immutable per-job location; JSON assets with relative paths
        # must be resolved upstream or remain valid after staging.
        staged = folder / 'input.json'
        staged.write_bytes(source.read_bytes())
        if SIF:
            executable = '/opt/conda/envs/torchx/bin/python'
            entry = '/opt/TorchX/torchfold_gpu/src/run_torchfold.py'
            argv = ['apptainer', 'exec', '--nv', '--bind', '/lustre:/lustre', SIF, executable, entry]
        else:
            argv = [PYTHON, str(script)]
        argv.extend([
            f'--json_path={staged}',
            f'--output_dir={outputs}',
            f'--db_dir={db}',
            f'--num_diffusion_samples={req.num_diffusion_samples}',
            f'--run_data_pipeline={str(req.run_data_pipeline).lower()}',
        ])
        argv.append(f'--checkpoint_path={weight}' if CKPT else f'--model_dir={weight}')

        workdir = '/opt/TorchX/torchfold_gpu/src' if SIF else str(script.parent)
        # `cd` is outside container for mounted source; no relative imports rely on it.
        lines = [
            '#!/usr/bin/env bash', 'set -uo pipefail',
            'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"',
            'export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"',
            f'cd {shlex.quote(str(script.parent))}',
            'set +e',
            ' '.join(shlex.quote(x) for x in argv) + ' > ' + shlex.quote(str(folder / 'stdout.log')) + ' 2> ' + shlex.quote(str(folder / 'stderr.log')),
            'rc=$?',
            f'printf "%s\\n" "$rc" > {shlex.quote(str(folder / "exit_code.txt"))}',
            'exit "$rc"',
        ]
        runfile = folder / 'run.sh'
        runfile.write_text('\n'.join(lines) + '\n')
        runfile.chmod(0o700)
        submit = [
            'sbatch', '--parsable', '--job-name', 'torchfold-' + fold_id[:8],
            '--partition', PARTITION, '--gres', GRES,
            '--cpus-per-task', str(CPUS), '--mem', MEM, '--time', TIME,
            '--output', str(folder / 'slurm-%j.out'), '--error', str(folder / 'slurm-%j.err'),
            str(runfile),
        ]
        result = call(submit, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or 'sbatch failed')
        job_id = result.stdout.strip().split(';')[0]
        if not job_id.isdecimal():
            raise RuntimeError(f'unexpected sbatch response: {result.stdout}')
        meta = {
            'fold_id': fold_id, 'slurm_job_id': job_id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'request': req.model_dump(), 'output_dir': str(outputs),
            'input_json': str(staged),
        }
        (folder / 'metadata.json').write_text(json.dumps(meta, indent=2))
        return {'fold_id': fold_id, 'slurm_job_id': job_id, 'state': 'PENDING'}
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get('/v1/folds/{fold_id}')
def fold_status(fold_id: str):
    folder = folder_for(fold_id)
    meta = get_meta(fold_id)
    return {**meta, 'state': slurm_state(meta['slurm_job_id']), 'exit_code': exit_code(folder)}


@app.get('/v1/folds/{fold_id}/results')
def fold_results(fold_id: str):
    folder = folder_for(fold_id)
    meta = get_meta(fold_id)
    outputs = folder / 'outputs'
    allowed = {'.cif', '.pdb', '.json', '.csv', '.txt'}
    files = sorted(str(p) for p in outputs.rglob('*') if p.is_file() and p.suffix.lower() in allowed)
    return {
        'fold_id': fold_id, 'state': slurm_state(meta['slurm_job_id']),
        'exit_code': exit_code(folder), 'output_dir': str(outputs),
        'files': files, 'count': len(files),
    }


@app.get('/v1/folds/{fold_id}/logs')
def fold_logs(fold_id: str, tail: int = 100):
    if not 1 <= tail <= 500:
        raise HTTPException(400, 'tail must be between 1 and 500')
    folder = folder_for(fold_id)
    paths = [folder / 'stdout.log', folder / 'stderr.log', *sorted(folder.glob('slurm-*.out')), *sorted(folder.glob('slurm-*.err'))]
    return {'fold_id': fold_id, 'logs': {
        str(p.relative_to(folder)): '\n'.join(p.read_text(errors='replace').splitlines()[-tail:])
        for p in paths if p.is_file()
    }}
