"""AF3Score REST API + Slurm worker in one file. Run `python main.py --worker TASK_DIR` internally."""
"""AF3Score async Slurm API; paths are shared host paths (Lustre)."""
import csv
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field, model_validator

import shutil

# AF3Score environment
AF3_ENV = Path("/lustre/grp/cmclab/leiy/conda_envs/af3score")
AF3_PYTHON = str(AF3_ENV / "bin/python")

# Add Conda binaries to PATH
os.environ["PATH"] = (
    str(AF3_ENV / "bin") + os.pathsep + os.environ.get("PATH", "")
)

# HMMER binaries
HMMER_BINARIES = [
    "jackhmmer",
    "nhmmer",
    "hmmsearch",
    "hmmbuild",
    "hmmalign",
]

for binary in HMMER_BINARIES:
    path = shutil.which(binary)
    if path is None:
        print(f"[WARNING] {binary} not found in PATH")
    else:
        print(f"[OK] {binary}: {path}")
ROOT = Path(os.getenv('AF3SCORE_API_WORK', '/work')).resolve()
SOURCE = Path(os.getenv('AF3SCORE_ROOT', '/opt/AF3Score')).resolve()
PYTHON = os.getenv('AF3SCORE_PYTHON', 'python')
DB = Path(os.getenv('AF3SCORE_DB_DIR', '/models/databases'))
MODEL = Path(os.getenv('AF3SCORE_MODEL_DIR', '/models/parameters'))
PARTITION = os.getenv('AF3SCORE_PARTITION', 'gpu3-13')
CPUS = int(os.getenv('AF3SCORE_CPUS', '8'))
MEM = os.getenv('AF3SCORE_MEM', '32G')
TIME = os.getenv('AF3SCORE_TIME', '12:00:00')
GPU = os.getenv('AF3SCORE_GRES', 'gpu:1')
MODE = os.getenv('AF3SCORE_EXECUTION', 'slurm')
ALLOWED_ROOTS = [Path(s).resolve() for s in os.getenv('AF3SCORE_ALLOWED_INPUT_ROOTS', str(ROOT)).split(':') if s]
app = FastAPI(title='AF3Score API', version='1.0.0', description='PDB-to-AF3Score scoring / prepared JSON+H5 / metrics, with Slurm task tracking')

class TaskIn(BaseModel):
    mode: Literal['pdb', 'prepared', 'metrics'] = 'pdb'
    input_pdb_dir: str | None = None
    json_dir: str | None = None
    h5_dir: str | None = None
    af3score_output_dir: str | None = None
    num_samples: int = Field(1, ge=1, le=20)
    num_workers: int = Field(4, ge=1, le=64)
    seed: int = Field(10, ge=0)
    run_data_pipeline: bool = False
    run_inference: bool = True
    init_guess: bool = True
    write_cif_model: bool = False
    write_summary_confidences: bool = True
    write_full_confidences: bool = True
    write_ranking_scores_csv: bool = False
    write_best_model_root: bool = False
    write_terms_of_use_file: bool = False
    write_fold_input_json_file: bool = False
    @model_validator(mode='after')
    def validate_mode(self):
        if self.mode != 'metrics' and not (self.run_data_pipeline or self.run_inference):
            raise ValueError('run_data_pipeline and run_inference cannot both be false')
        if self.mode == 'metrics' and (self.run_data_pipeline or not self.run_inference):
            raise ValueError('metrics mode does not run data pipeline or inference; use defaults')
        if self.mode in ('pdb','metrics') and not self.input_pdb_dir:
            raise ValueError('input_pdb_dir required')
        if self.mode == 'prepared' and (not self.json_dir or not self.h5_dir):
            raise ValueError('json_dir and h5_dir required')
        if self.mode == 'metrics' and not self.af3score_output_dir:
            raise ValueError('af3score_output_dir required for metrics')
        if self.mode == 'pdb' and self.run_inference and (not self.write_summary_confidences or not self.write_full_confidences):
            raise ValueError('pdb mode requires summary and full confidences for metrics')
        return self

def resolve_input(raw: str, directory=True) -> str:
    try:
        p = Path(raw).resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError):
        raise HTTPException(422, f'Input path does not exist: {raw}')
    if not any(p == r or r in p.parents for r in ALLOWED_ROOTS):
        raise HTTPException(403, f'Path not in AF3SCORE_ALLOWED_INPUT_ROOTS: {p}')
    if directory and not p.is_dir():
        raise HTTPException(422, f'Not a directory: {p}')
    return str(p)

def taskdir(task_id):
    if not re.fullmatch(r'[0-9a-f]{32}', task_id): raise HTTPException(404, 'Task not found')
    p = ROOT / 'tasks' / task_id
    if not (p / 'task.json').exists(): raise HTTPException(404, 'Task not found')
    return p

def read_task(p): return json.loads((p/'task.json').read_text())

def slurm_state(job_id):
    if not job_id: return 'UNKNOWN'
    try:
        r = subprocess.run(['squeue','-h','-j',str(job_id),'-o','%T'],capture_output=True,text=True,timeout=8)
        if r.returncode == 0 and r.stdout.strip(): return r.stdout.splitlines()[0].strip()
        r = subprocess.run(['sacct','-n','-P','-j',str(job_id),'--format=State'],capture_output=True,text=True,timeout=8)
        if r.returncode == 0:
            states=[x.strip().split('|')[0].split()[0] for x in r.stdout.splitlines() if x.strip()]
            if states: return states[0].split('+')[0]
    except (OSError,subprocess.TimeoutExpired): pass
    return 'UNKNOWN'

def view_task(p):
    d=read_task(p)
    result=p/'result.json'
    if result.exists():
        try: result_data=json.loads(result.read_text())
        except json.JSONDecodeError: result_data={'status':'UNKNOWN'}
        status=result_data.get('status','UNKNOWN')
    elif d['execution']=='slurm': status=slurm_state(d.get('slurm_job_id'))
    else: status='RUNNING' if (p/'started').exists() else 'SUBMITTED'
    return {'task_id':d['task_id'],'status':status,'slurm_job_id':d.get('slurm_job_id'), 'mode':d['request']['mode'], 'work_dir':str(p),'result_available':result.exists()}

@app.get('/health')
def health(): return {'status':'ok','execution':MODE,'work_dir':str(ROOT),'upstream':str(SOURCE)}

@app.post('/v1/tasks',status_code=202)
def create_task(payload:TaskIn):
    d=payload.model_dump()
    if d['mode']!='metrics' and d['run_data_pipeline'] and not d['run_inference']:
        d['write_fold_input_json_file']=True
    for k in ('input_pdb_dir','json_dir','h5_dir','af3score_output_dir'):
        if d.get(k): d[k]=resolve_input(d[k])
    if d['mode']=='pdb' and not list(Path(d['input_pdb_dir']).glob('*.pdb')):
        raise HTTPException(422,'No .pdb files found')
    if d['mode']=='prepared':
        jsons={p.stem for p in Path(d['json_dir']).glob('*.json')}
        h5s={p.stem for p in Path(d['h5_dir']).glob('*.h5')}
        if not jsons or jsons!=h5s: raise HTTPException(422,'JSON/H5 basenames must match and be nonempty')
    ROOT.joinpath('tasks').mkdir(parents=True,exist_ok=True)
    task_id=uuid.uuid4().hex
    p=ROOT/'tasks'/task_id
    p.mkdir(mode=0o700)
    data={'task_id':task_id,'request':d,'execution':MODE,'slurm_job_id':None}
    (p/'task.json').write_text(json.dumps(data,indent=2))
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',str(p)]
    worker_image=os.getenv('AF3SCORE_WORKER_IMAGE','').strip()
    if worker_image:
        # This command is submitted by the native API process; Apptainer is executed on the Slurm node.
        cmd=['apptainer','exec','--bind','/lustre:/lustre',worker_image,
             'python','/app/main.py','--worker',str(p)]
    if MODE=='slurm':
        sb=['sbatch','--parsable','--job-name=af3s-'+task_id[:8], '--partition='+PARTITION,
            '--cpus-per-task='+str(CPUS),'--mem='+MEM,'--time='+TIME,
            '--output='+str(p/'slurm-%j.out'),'--error='+str(p/'slurm-%j.err')]
        if d['mode']!='metrics' and d['run_inference']: sb.append('--gres='+GPU)
        r=subprocess.run(sb+['--wrap', ' '.join(__import__('shlex').quote(x) for x in cmd)],capture_output=True,text=True)
        if r.returncode:
            (p/'submit_error.txt').write_text(r.stderr)
            raise HTTPException(503,{'error':'sbatch failed','details':r.stderr.strip(),'task_id':task_id})
        data['slurm_job_id']=r.stdout.strip().split(';')[0]
    elif MODE=='local':
        with (p/'local.log').open('wb') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        data['local_pid']=process.pid
    else: raise HTTPException(500,'Unsupported AF3SCORE_EXECUTION')
    (p/'task.json').write_text(json.dumps(data,indent=2))
    return view_task(p)

@app.get('/v1/tasks')
def list_tasks(limit:int=Query(50,ge=1,le=500)):
    paths=sorted((ROOT/'tasks').glob('*/task.json'),key=lambda x:x.stat().st_mtime,reverse=True) if (ROOT/'tasks').exists() else []
    return [view_task(p.parent) for p in paths[:limit]]

@app.get('/v1/tasks/{task_id}')
def task_status(task_id:str): return view_task(taskdir(task_id))

@app.get('/v1/tasks/{task_id}/results')
def task_results(task_id:str):
    p=taskdir(task_id); result=p/'result.json'
    if not result.exists(): return {'task_id':task_id,'status':view_task(p)['status'],'ready':False}
    return {'task_id':task_id,'ready':True,**json.loads(result.read_text())}

@app.get('/v1/tasks/{task_id}/metrics')
def task_metrics(task_id:str,limit:int=Query(100,ge=1,le=5000)):
    p=taskdir(task_id); f=p/'af3score_metrics.csv'
    if not f.exists(): raise HTTPException(404,'Metrics CSV not ready')
    with f.open(newline='') as file:
        reader=csv.DictReader(file)
        return {'task_id':task_id,'columns':reader.fieldnames,'rows':[row for _,row in zip(range(limit),reader)]}

@app.get('/v1/tasks/{task_id}/metrics.csv')
def download_metrics(task_id:str):
    f=taskdir(task_id)/'af3score_metrics.csv'
    if not f.exists(): raise HTTPException(404,'Metrics CSV not ready')
    return FileResponse(f,filename=f'af3score-{task_id}.csv',media_type='text/csv')

@app.get('/v1/tasks/{task_id}/logs',response_class=PlainTextResponse)
def logs(task_id:str,tail:int=Query(250,ge=1,le=10000)):
    p=taskdir(task_id)
    files=sorted(p.glob('slurm-*.out'))+sorted(p.glob('slurm-*.err'))+[p/'local.log',p/'worker.log']
    lines=[]
    for f in files:
        if f.exists(): lines += [f'### {f.name} ###']+f.read_text(errors='replace').splitlines()[-tail:]
    return '\n'.join(lines[-tail*3:])

@app.post('/v1/tasks/{task_id}/cancel')
def cancel(task_id:str):
    p=taskdir(task_id); d=read_task(p)
    if d['execution']!='slurm' or not d.get('slurm_job_id'):
        raise HTTPException(409,'Only Slurm tasks support cancellation')
    r=subprocess.run(['scancel',str(d['slurm_job_id'])],capture_output=True,text=True)
    if r.returncode: raise HTTPException(502,r.stderr.strip())
    return {'task_id':task_id,'cancellation_requested':True}


# --- Worker (runs on GPU node, not in API process) ---
"""Executed on a Slurm GPU node or in local mode. No nested sbatch jobs."""
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


def run(cmd, log):
    print('+', ' '.join(map(str,cmd)), flush=True)
    with log.open('a') as fp:
        r=subprocess.run(list(map(str,cmd)),stdout=fp,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f'Exit code {r.returncode}: {cmd[0]} {cmd[1]} (see worker.log)')


def prepare_pdb_files(source, python, pdb_dir, p, opts):
    """Invoke upstream PDB->chain-CIF/AF3 JSON logic, then upstream PDB->H5.
    Avoid upstream's split_by_total_length() edge cases for single inputs.
    """
    # Must import upstream using the inference environment: that environment has Bio/pandas.
    script=p/'prepare_native.py'
    script.write_text('''
import importlib.util, json, pathlib, sys
from Bio import PDB
from Bio.PDB import Structure, Model, MMCIFIO
src, in_dir, outdir, seed, pipeline_enabled = sys.argv[1:]
pipeline_enabled = pipeline_enabled == 'true'
out = pathlib.Path(outdir); out.mkdir(exist_ok=True,parents=True)
cif_dir=out/'single_chain_cif'; cif_dir.mkdir(exist_ok=True)
json_dir=out/'json'; json_dir.mkdir(exist_ok=True)
# Load upstream to reuse its exact chain/JSON extraction behavior.
spec=importlib.util.spec_from_file_location('af3_prepare', str(pathlib.Path(src)/'01_prepare_get_json.py'))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
for pdb in sorted(pathlib.Path(in_dir).glob('*.pdb')):
    name, seqs, total = mod.process_single_pdb((str(pdb),str(cif_dir)))
    if name is None or not seqs or total < 1: raise RuntimeError('PDB parse failed: '+str(pdb))
    chain_list=[]
    for chain_id, seq in seqs.items():
        if not seq: continue
        if len(chain_id)!=1 or not ('A' <= chain_id <= 'Z'):
            raise ValueError(f'Chain IDs must be uppercase A-Z: {pdb}, {chain_id}')
        if 'X' in seq: raise ValueError(f'Unsupported residue in {pdb} chain {chain_id}')
        cif=str(cif_dir/f'{name}_chain_{chain_id}.cif')
        protein = {'id': chain_id, 'sequence': seq, 'modifications': []}
        if not pipeline_enabled:
            # Fast AF3Score: query-only MSA plus each input chain as an exact template.
            protein.update({
                'unpairedMsa': f'>query\\n{seq}\\n',
                'pairedMsa': f'>query\\n{seq}\\n',
                'templates': [{'mmcifPath': cif,
                               'queryIndices': list(range(len(seq))),
                               'templateIndices': list(range(len(seq)))}],
            })
        # With pipeline enabled, omit MSA/templates to allow upstream database search.
        chain_list.append({'protein': protein})
    if not chain_list: raise RuntimeError('No protein chains: '+str(pdb))
    out_json={'dialect':'alphafold3','version':1,'name':name,'sequences':chain_list,
       'modelSeeds':[int(seed)],'bondedAtomPairs':None,'userCCD':None}
    (json_dir/f'{name}.json').write_text(json.dumps(out_json,indent=2))
''')
    run([python,script,source,pdb_dir,p,opts['seed'],str(opts['run_data_pipeline']).lower()],p/'worker.log')
    json_dir=p/'json'
    h5_dir=p/'h5';h5_dir.mkdir()
    # Upstream uses a bucket as last folder suffix; group input by bucket.
    buckets=[256,512,768,1024,1280,1536,2048,2560,3072]
    groups={}
    for srcpdb in sorted(Path(pdb_dir).glob('*.pdb')):
        doc=json.loads((json_dir/f'{srcpdb.stem}.json').read_text())
        length=sum(len(x['protein']['sequence']) for x in doc['sequences'])
        bucket=next((v for v in buckets if v>=length),None)
        if not bucket: raise ValueError(f'{srcpdb.name}: >3072 residues not supported by upstream converter')
        inp=p/'batches'/f'batch_0_{bucket}';inp.mkdir(parents=True,exist_ok=True)
        (inp/srcpdb.name).symlink_to(srcpdb)
        groups.setdefault(bucket,[]).append(srcpdb.stem)
    for bucket in sorted(groups):
        folder=p/'batches'/f'batch_0_{bucket}'
        output=p/'h5_buckets'/str(bucket)
        output.mkdir(parents=True,exist_ok=True)
        run([python,source/'02_prepare_pdb2jax.py','--pdb_folder',folder,
             '--output_folder',output,'--num_workers',opts['num_workers']],p/'worker.log')
        for stem in groups[bucket]:
            f=output/f'{stem}.h5'
            if not f.exists(): raise RuntimeError(f'PDB to H5 failed: {stem}')
            (h5_dir/f.name).symlink_to(f)
    return json_dir,h5_dir


def infer(source,python,db,model,json_dir,h5_dir,p,opts):
    output=p/'af3score_outputs';output.mkdir(exist_ok=True)
    # Run batch mode to reuse loaded CCD/models; matching JSON/H5 stems required.
    args=[python,source/'run_af3score.py',
       '--db_dir='+str(db),'--model_dir='+str(model),
       '--batch_json_dir='+str(json_dir),'--batch_h5_dir='+str(h5_dir),
       '--output_dir='+str(output),
       '--run_data_pipeline='+str(opts['run_data_pipeline']).lower(),
       '--run_inference='+str(opts['run_inference']).lower(),
       '--init_guess='+str(opts['init_guess']).lower(),
       '--num_samples='+str(opts['num_samples'])]
    for key in ('write_cif_model','write_summary_confidences','write_full_confidences',
                'write_ranking_scores_csv','write_best_model_root',
                'write_terms_of_use_file','write_fold_input_json_file'):
        args.append('--'+key+'='+str(opts[key]).lower())
    # GPU JAX env settings, overridable through Slurm --export/worker environment.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('XLA_CLIENT_MEM_FRACTION','0.85')
    run(args,p/'worker.log')
    return output


def metrics(source,python,pdb_dir,output,p,opts):
    csvpath=p/'af3score_metrics.csv'
    run([python,source/'04_get_metrics.py','--input_pdb_dir',pdb_dir,
         '--af3score_output_dir',output,'--save_metric_csv',csvpath,
         '--num_workers',opts['num_workers']],p/'worker.log')
    if not csvpath.exists(): raise RuntimeError('Metrics CSV was not written')
    with csvpath.open(newline='') as f: rows=list(csv.DictReader(f))
    expected=len(list(Path(pdb_dir).glob('*.pdb')))
    if len(rows)!=expected:
        raise RuntimeError(f'Metric extraction incomplete: {len(rows)}/{expected}; inspect failed_records.txt')
    return {'num_records':len(rows),'metrics_csv':str(csvpath),'records':rows[:100]}


def run_worker(p):
    d=json.loads((p/'task.json').read_text()); opts=d['request']
    source=Path(os.environ.get('AF3SCORE_ROOT','/opt/AF3Score'))
    python=os.environ.get('AF3SCORE_PYTHON','python')
    db=Path(os.environ.get('AF3SCORE_DB_DIR','/models/databases'))
    model=Path(os.environ.get('AF3SCORE_MODEL_DIR','/models/parameters'))
    start=time.time();(p/'started').touch()
    try:
        if not source.joinpath('run_af3score.py').is_file():
            raise FileNotFoundError(f'AF3SCORE_ROOT invalid: {source}')
        if opts['mode']=='pdb':
            if opts['run_data_pipeline'] and not db.is_dir():
                raise FileNotFoundError('AF3SCORE_DB_DIR missing (required for data pipeline)')
            if opts['run_inference'] and not model.is_dir():
                raise FileNotFoundError('AF3SCORE_MODEL_DIR missing (required for inference)')
            json_dir,h5_dir=prepare_pdb_files(source,python,Path(opts['input_pdb_dir']),p,opts)
            output=infer(source,python,db,model,json_dir,h5_dir,p,opts)
            if opts['run_inference']:
                res=metrics(source,python,opts['input_pdb_dir'],output,p,opts)
            else:
                res={'af3score_output_dir':str(output), 'note':'Data pipeline only: no inference or metrics. Set write_fold_input_json_file=true to save processed input JSONs.'}
        elif opts['mode']=='prepared':
            if opts['run_data_pipeline'] and not db.is_dir():
                raise FileNotFoundError('AF3SCORE_DB_DIR missing (required for data pipeline)')
            if opts['run_inference'] and not model.is_dir():
                raise FileNotFoundError('AF3SCORE_MODEL_DIR missing (required for inference)')
            output=infer(source,python,db,model,opts['json_dir'],opts['h5_dir'],p,opts)
            res={'af3score_output_dir':str(output),'note':('Prepared input inference: raw confidence outputs; metrics require original input PDBs' if opts['run_inference'] else 'Data pipeline only; no inference or metrics. Set write_fold_input_json_file=true to save processed JSONs.')}
        else:
            res=metrics(source,python,opts['input_pdb_dir'],opts['af3score_output_dir'],p,opts)
        result={'status':'COMPLETED','duration_seconds':round(time.time()-start,2),'output_dir':str(p),**res}
    except Exception as e:
        traceback.print_exc()
        result={'status':'FAILED','duration_seconds':round(time.time()-start,2),'error':str(e),'traceback':traceback.format_exc()}
    temp=p/'result.tmp'
    temp.write_text(json.dumps(result,indent=2,default=str));temp.replace(p/'result.json')
    if result['status']=='FAILED': sys.exit(1)




if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--worker":
    run_worker(Path(sys.argv[2]).resolve())
