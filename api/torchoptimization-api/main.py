"""FastAPI adapter for the unmodified TorchOptimization 2.0.13 launcher."""
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

ROOT = Path(os.getenv('TORCHOPT_ROOT', '/lustre/grp/cmclab/share/huangjm/Project/TorchOptimization_v2.0.13')).resolve()
WORK = Path(os.getenv('TORCHOPT_API_WORK', str(Path(__file__).resolve().parent / 'jobs'))).resolve()
ALLOWED = Path(os.getenv('TORCHOPT_ALLOWED_ROOT', '/lustre')).resolve()
PYTHON = os.getenv('TORCHOPT_PYTHON', '/lustre/grp/cmclab/share/wangmm/Optimization/env/torchx_env/bin/python')
MODEL_DIR = os.getenv('TORCHOPT_MODEL_DIR', str(ROOT / 'assets/torchcraft'))
RDE_DIR = os.getenv('TORCHOPT_RDE_DIR', str(ROOT / 'assets/RDE-PPI'))
app = FastAPI(title='TorchOptimization Slurm API', version='2.0.13')

class Request(BaseModel):
    name: str = Field(default='target', pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')
    pdb_paths: list[str] = Field(min_length=1)
    binder_type: Literal['mini-binder', 'vhh'] = 'mini-binder'
    method: Literal['mcmc', 'ddg', 'mcmc-ddg'] = 'mcmc-ddg'
    binder_chain: str = Field(default='A', min_length=1, max_length=1)
    target_chain: str = Field(default='C', min_length=1, max_length=1)
    interface_cutoff: float = Field(default=5, gt=0)
    mcmc_rounds: int = Field(default=5, ge=1)
    mcmc_proposals: int = Field(default=5, ge=1)
    ddg_rounds: int = Field(default=5, ge=1)
    rescue_iptm_cutoff: float = Field(default=.8, ge=0, le=1)
    rescue_bottom_percent: float = Field(default=25, ge=0, le=100)
    partition: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_,.-]+$')
    full_jobs: int = Field(default=1, ge=1)
    full_gpus: int = Field(default=1, ge=1)
    prepare_only: bool = False

    @model_validator(mode='after')
    def chains(self):
        if self.binder_chain == self.target_chain:
            raise ValueError('binder_chain and target_chain must differ')
        return self

def folder(job_id):
    if not re.fullmatch('[0-9a-f]{32}', job_id):
        raise HTTPException(400, 'invalid optimization_id')
    path = WORK / job_id
    if not (path / 'metadata.json').exists():
        raise HTTPException(404, 'optimization not found')
    return path

@app.get('/health')
def health():
    return {'backend': str(ROOT), 'backend_available': (ROOT / 'launch.py').is_file()}

@app.post('/v1/optimizations', status_code=201)
def submit(req: Request):
    inputs = []
    for value in req.pdb_paths:
        p = Path(value).resolve()
        if not p.is_relative_to(ALLOWED) or not p.is_file() or p.suffix.lower() != '.pdb':
            raise HTTPException(400, f'not an allowed PDB: {value}')
        inputs.append(p)
    if not (ROOT / 'launch.py').is_file():
        raise HTTPException(503, 'TORCHOPT_ROOT is unavailable')
    job_id = uuid.uuid4().hex
    path = WORK / job_id
    path.mkdir(parents=True)
    meta = {'optimization_id': job_id, 'request': req.model_dump(), 'state': 'PREPARING'}
    def save():
        (path / 'metadata.json').write_text(json.dumps(meta, indent=2))
    save()
    try:
        # Each request has its own launcher root, inputs, control and runs.
        for name in ('src', 'engine', 'model_engine', 'external', 'assets', 'support', '.envs'):
            if (ROOT / name).exists():
                (path / name).symlink_to(ROOT / name, target_is_directory=True)
        shutil.copy2(ROOT / 'launch.py', path / 'launch.py')
        dest = path / 'inputs/pdbs'
        dest.mkdir(parents=True)
        for i, p in enumerate(inputs):
            shutil.copy2(p, dest / f'{i:06d}_{p.name}')
        config = req.model_dump(exclude={'name', 'pdb_paths', 'prepare_only'})
        (path / 'control.yaml').write_text(yaml.safe_dump(config))
        args = [PYTHON, '-I', '-B', str(path / 'launch.py'), 'full']
        if req.prepare_only:
            args.append('--prepare-only')
        # Keep output even when submission partially succeeds; never automatically retry.
        with (path / 'submission.log').open('w') as log:
            env = dict(os.environ, TORCHOPT_PYTHON=PYTHON,
                       TORCHOPT_MODEL_DIR=MODEL_DIR, TORCHOPT_RDE_DIR=RDE_DIR)
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, cwd=path, env=env)
        meta['state'] = ('PREPARED' if req.prepare_only else 'SUBMITTED') if result.returncode == 0 else 'SUBMISSION_FAILED'
        meta['returncode'] = result.returncode
    except Exception as exc:
        meta.update(state='SUBMISSION_FAILED', error=str(exc))
    save()
    return meta

@app.get('/v1/optimizations/{job_id}')
def status(job_id: str):
    path = folder(job_id)
    meta = json.loads((path / 'metadata.json').read_text())
    meta['runs'] = {str(p.parent.relative_to(path)): json.loads(p.read_text()) for p in sorted(path.glob('runs/*/summary.json'))}
    meta['submissions'] = {str(p.relative_to(path)): json.loads(p.read_text()) for p in sorted(path.glob('runs/*/*submission.json'))}
    return meta

@app.get('/v1/optimizations/{job_id}/logs')
def logs(job_id: str):
    path = folder(job_id)
    files = [path / 'submission.log', *sorted(path.glob('runs/*/logs/*'))]
    out = {}
    for p in files:
        if p.is_file():
            with p.open('rb') as f:
                f.seek(max(0, p.stat().st_size - 65536))
                out[str(p.relative_to(path))] = f.read().decode(errors='replace')
    return out

@app.get('/v1/optimizations/{job_id}/results')
def results(job_id: str):
    path = folder(job_id)
    return {'optimization_id': job_id, 'results': [str(p) for p in sorted(path.glob('runs/*/results.csv'))], 'structures': [str(p) for p in sorted(path.glob('runs/*/cases/**/final/*.pdb'))]}
