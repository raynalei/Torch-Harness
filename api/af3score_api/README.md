# AF3Score API

Upstream: https://github.com/Mingchenchen/AF3Score

A lightweight FastAPI service that submits AF3Score tasks to Slurm. Keep only `main.py`, `af3score.def` (optional), and this `README.md` in the API project. Install upstream AF3Score, its inference environment, model parameters, and optional MSA/template databases separately. This API does not distribute model weights.

## 1. Setup: native API on a login/CPU node

```bash
# Clone upstream separately; omit this step if already cloned.
git clone https://github.com/Mingchenchen/AF3Score.git /lustre/grp/cmclab/leiy/AF3Score

# In the API Python environment (NOT necessarily the AF3/JAX inference environment):
python3 -m pip install fastapi uvicorn 'pydantic>=2'

mkdir -p /lustre/grp/cmclab/leiy/af3score_api_jobs/input_pdbs

# All cluster paths and Slurm settings live here. Set these in the shell that starts uvicorn.
export AF3SCORE_ROOT=/lustre/grp/cmclab/leiy/AF3Score
export AF3SCORE_API_WORK=/lustre/grp/cmclab/leiy/af3score_api_jobs
export AF3SCORE_PYTHON=/lustre/grp/cmclab/leiy/conda_envs/af3score_env/bin/python
export AF3SCORE_DB_DIR=/lustre/grp/cmclab/leiy/databases/alphafold3
export AF3SCORE_MODEL_DIR=/lustre/grp/cmclab/leiy/af3_model_parameters
export AF3SCORE_ALLOWED_INPUT_ROOTS=/lustre/grp/cmclab/leiy
export AF3SCORE_PARTITION=gpu3-12
export AF3SCORE_GRES=gpu:1
export AF3SCORE_CPUS=8
export AF3SCORE_MEM=32G
export AF3SCORE_TIME=12:00:00
export AF3SCORE_EXECUTION=slurm
# Optional. Leave unset for native worker (recommended).
# export AF3SCORE_WORKER_IMAGE=/lustre/grp/cmclab/leiy/containers/af3score-api.sif

# Start from the API project directory containing main.py on shared Lustre.
uvicorn main:app --host 127.0.0.1 --port 8030
```

**Before running:** set `AF3SCORE_MODEL_DIR` to the **actual directory containing the installed AF3 model parameters**; the above is a suggested location, not a guarantee that files exist. `AF3SCORE_DB_DIR` is required only for tasks with `run_data_pipeline=true` (the upstream may still read basic non-MSA resources when inference runs). Download AF3 database files using the upstream `fetch_databases.sh` into this directory and make sure HMMER binaries (jackhmmer, nhmmer, hmmalign, hmmsearch, hmmbuild) are on the Slurm worker's `PATH`. See upstream setup instructions. Its actual database filenames/layout must match the installed AF3Score version.

`main.py` **must be on shared storage** reachable from Slurm compute nodes. The native FastAPI process needs only FastAPI/Uvicorn/Pydantic; `AF3SCORE_PYTHON` executes upstream preprocessing and AF3/JAX inference in the worker. Run on a trusted cluster network only: this API has **no authentication**. Input directories are restricted to `AF3SCORE_ALLOWED_INPUT_ROOTS`.

### Important: environment propagation

Slurm jobs normally inherit the submitting process environment. Start `uvicorn` from the same shell in which you exported these settings. Check your cluster's Slurm environment export policies; if jobs do not inherit them, configure `sbatch --export` appropriately. When using a container worker, the container must also expose the native AF3 inference Python and its dependencies; the lightweight `.def` image alone does not.

## 2. Task parameters

`POST /v1/tasks` accepts these controls **per task**:

| JSON field | Default | Meaning |
| --- | --- | --- |
| `mode` | `pdb` | `pdb`, `prepared`, or `metrics` |
| `run_data_pipeline` | `false` | Run upstream MSA and structural template searches (needs database and HMMER) |
| `run_inference` | `true` | Run AF3Score model inference (needs weights/GPU) |
| `init_guess` | `true` | Use per-PDB H5 coordinates as initial guess during inference |
| `num_samples` | `1` | Number of model samples (1–20) |
| `num_workers` | `4` | PDB preprocessing and metrics workers |
| `seed` | `10` | Model seed stored in generated JSON |
| `write_fold_input_json_file` | `false` | Save processed fold-input JSON; automatically enabled for pipeline-only runs |

`pdb` mode: input PDB directory → chain CIF/AF3 JSON → coordinate H5 → AF3Score → metrics CSV (when inference enabled).

- `run_data_pipeline=false`: generated JSON has **query-only MSA** and each original PDB chain as an exact template. No database-based MSA search. This is the original fast PDB scoring behavior.
- `run_data_pipeline=true`: generated JSON **omits the prefilled MSA/templates**, allowing upstream AF3Score's data pipeline to search the configured databases. The coordinate H5 is still used when `init_guess=true`.
- `run_inference=false`, `run_data_pipeline=true`: preprocessing only, no GPU or metrics CSV. The processed fold-input JSON is automatically written to the task output.
- Both flags cannot be false (except `metrics` mode, which performs neither of these operations).

`prepared` mode: uses existing `json_dir` and `h5_dir` with exactly matching file basenames; your supplied JSON determines whether MSA/templates are already populated. If `run_data_pipeline=true`, the API **does not rewrite** the prepared JSON: omit those fields yourself if you want an upstream search. Produces raw AF3Score outputs; does not extract per-PDB metrics automatically.

`metrics` mode: extracts metrics from existing input PDBs and AF3Score output directory. Does not run the data pipeline or inference.

**Caution:** running the data pipeline is not automatically equivalent to obtaining new MSA/template features if a JSON already contains them. Template search and MSA behavior depend on upstream AF3Score/AlphaFold 3 implementation and installed database versions. Also, `init_guess` and structural `templates` are different input mechanisms. Compare ranking scores between fast and MSA modes only after evaluating their consistency.

## 3. Submit / inspect results (port 8030)

```bash
curl -sS http://127.0.0.1:8030/health

# A. Fast scoring: PDB -> JSON/H5 -> inference -> metrics; no database searches
curl -sS -X POST http://127.0.0.1:8030/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"mode":"pdb","input_pdb_dir":"/lustre/grp/cmclab/leiy/af3score_api_jobs/input_pdbs","run_data_pipeline":false,"run_inference":true,"init_guess":true,"num_samples":1}'

# B. Database searches + AF3Score inference + metrics
curl -sS -X POST http://127.0.0.1:8030/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"mode":"pdb","input_pdb_dir":"/lustre/grp/cmclab/leiy/af3score_api_jobs/input_pdbs","run_data_pipeline":true,"run_inference":true,"init_guess":true,"num_samples":1}'

# C. MSA/template preprocessing only, NO inference or metrics
curl -sS -X POST http://127.0.0.1:8030/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"mode":"pdb","input_pdb_dir":"/lustre/grp/cmclab/leiy/af3score_api_jobs/input_pdbs","run_data_pipeline":true,"run_inference":false}'

# D. Infer from already prepared JSON and H5 (basenames must match)
curl -sS -X POST http://127.0.0.1:8030/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"mode":"prepared","json_dir":"/lustre/grp/cmclab/leiy/prepared/json","h5_dir":"/lustre/grp/cmclab/leiy/prepared/h5","run_data_pipeline":false,"run_inference":true}'

# Replace TASK_ID with the task_id returned from POST.
curl -sS http://127.0.0.1:8030/v1/tasks/TASK_ID
curl -sS http://127.0.0.1:8030/v1/tasks/TASK_ID/results
curl -sS http://127.0.0.1:8030/v1/tasks/TASK_ID/metrics
curl -o metrics.csv http://127.0.0.1:8030/v1/tasks/TASK_ID/metrics.csv
curl -sS http://127.0.0.1:8030/v1/tasks/TASK_ID/logs
```

Additional endpoints: `GET /v1/tasks`, `POST /v1/tasks/TASK_ID/cancel`; interactive OpenAPI: `http://127.0.0.1:8040/docs`. Each job's files and Slurm logs live in `$AF3SCORE_API_WORK/tasks/TASK_ID/`. `/metrics` and `/metrics.csv` return 404 for pipeline-only jobs because no metrics exist.

## 4. Optional Apptainer image

Build from the API directory that contains `af3score.def`:

```bash
mkdir -p /lustre/grp/cmclab/leiy/{apptainer_cache,apptainer_tmp,containers}
export APPTAINER_CACHEDIR=/lustre/grp/cmclab/leiy/apptainer_cache
export APPTAINER_TMPDIR=/lustre/grp/cmclab/leiy/apptainer_tmp
apptainer build /lustre/grp/cmclab/leiy/containers/af3score-api.sif af3score.def
```

The `.def` image is lightweight and is **not** a complete AF3/CUDA/JAX environment. Prefer the native API + Slurm worker + existing AF3Score inference environment. If using `AF3SCORE_WORKER_IMAGE`, make sure `python /app/main.py` exists inside that image and the AF3Score Python interpreter at `AF3SCORE_PYTHON` is runnable inside it; otherwise leave the image setting unset.

**Verification:** Python syntax checked; end-to-end predictions, database search, and Slurm operation have **not** been tested on your cluster. Run a small PDB test and inspect task logs before production use.
