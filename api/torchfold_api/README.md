# TorchFold API 

 The FastAPI process runs on a Slurm submission host (CPU is sufficient); each `/v1/folds` request submits a GPU inference job with `sbatch`. No GPU is held by the API process.

The actual upstream inference entry is `TorchX/torchfold_gpu/src/run_torchfold.py`. TorchFold is distinct from TorchCraft's `mini_binder` / `vhh` design modes.

## API

| Method | Route | Description |
|---|---|---|
| GET | `/health` | API health |
| POST | `/v1/folds` | Submit prediction, returns `fold_id`, `slurm_job_id` |
| GET | `/v1/folds/{fold_id}` | Slurm status and exit code |
| GET | `/v1/folds/{fold_id}/results` | Output files and directory |
| GET | `/v1/folds/{fold_id}/logs?tail=100` | stdout, stderr, Slurm logs |

## 1. Deploy files

```bash
mkdir -p /lustre/grp/cmclab/leiy/torchfold_api
# Copy this entire folder to that directory.
```

The `TorchX` checkout must contain:

```text
/lustre/grp/cmclab/leiy/TorchX/torchfold_gpu/src/run_torchfold.py
/lustre/grp/cmclab/leiy/TorchX/torchx_gpu/torchx/environment.yml
```

## 2. Build the GPU inference SIF

This includes TorchX source and its shared `torchx` environment, but deliberately excludes checkpoints and databases.

```bash
cd /lustre/grp/cmclab/leiy/torchfold_api
apptainer build --fakeroot \
  /lustre/grp/cmclab/leiy/containers/torchfold.sif \
  torchfold.def
```

Image destination:

```text
/lustre/grp/cmclab/leiy/containers/torchfold.sif
```

`apptainer build --fakeroot` needs functional fakeroot support, internet access for the CUDA OCI base / packages, and enough disk quota. If the cluster prohibits fakeroot, build on a permitted image-build host and transfer the SIF. No Docker daemon is required. The `torchfold.def` uses a fixed source location in its `%files`; update that if TorchX lives elsewhere. Validate TorchX package/API version compatibility before production use.

Verify **on a Slurm-allocated NVIDIA GPU node**:

```bash
apptainer exec --nv --bind /lustre:/lustre \
  /lustre/grp/cmclab/leiy/containers/torchfold.sif \
  /opt/conda/envs/torchx/bin/python -c \
  'import torch, torchx.cpp; print(torch.cuda.is_available())'
```

## 3. Start FastAPI on a Slurm submission host

FastAPI only needs Python 3.10+ and `requirements-api.txt`; it does not need to run inside the heavyweight GPU SIF.

```bash
cd /lustre/grp/cmclab/leiy/torchfold_api

export TORCHFOLD_PYTHON=/lustre/grp/cmclab/leiy/conda_envs/torchx_env/bin/python
export TORCHFOLD_ROOT=/lustre/grp/cmclab/leiy/TorchX
export TORCHFOLD_API_WORK=/lustre/grp/cmclab/leiy/torchfold_api_jobs
export TORCHFOLD_ALLOWED_ROOT=/lustre/grp/cmclab
export TORCHFOLD_SIF=/lustre/grp/cmclab/leiy/containers/torchfold.sif
export TORCHFOLD_PARTITION=gpu3-12
export TORCHFOLD_GRES=gpu:1
export TORCHFOLD_CPUS=8
export TORCHFOLD_MEM=64G
export TORCHFOLD_TIME=12:00:00

# Choose EXACTLY ONE:
#export TORCHFOLD_CHECKPOINT=/lustre/grp/cmclab/leiy/weights/torchfold.pt
#unset TORCHFOLD_MODEL_DIR
# Alternative: unset TORCHFOLD_CHECKPOINT and set TORCHFOLD_MODEL_DIR（alphafold3）
export TORCHFOLD_MODEL_DIR=/lustre/grp/cmclab/leiy/

# Only needed if run_data_pipeline=true:
export TORCHFOLD_DB_DIR=/lustre/grp/cmclab/leiy/af3_databases
uvicorn main:app --host 127.0.0.1 --port 8040
```

If you have an existing native conda environment instead of a SIF, use `unset TORCHFOLD_SIF` and set `TORCHFOLD_PYTHON` to the interpreter in that environment.



## 4. Submit a fold

The input must be a **TorchFold/AlphaFold3-compatible JSON file** already residing on shared Lustre. This API does not generate MSA data automatically unless `run_data_pipeline=true`, which requires configured local databases and binaries. `run_data_pipeline=false` needs prepared inputs with the requisite features / MSA data.

```bash
curl -sS -X POST http://127.0.0.1:8040/v1/folds \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "antibody_antigen_test",
    "json_path": "/lustre/grp/cmclab/leiy/TorchX/torchfold_gpu/test.json",
    "run_data_pipeline": false,
    "num_diffusion_samples": 2
  }' | python3 -m json.tool
```

Use the `fold_id` from the response:

```bash
curl -sS http://127.0.0.1:8040/v1/folds/FOLD_ID | python3 -m json.tool
curl -sS http://127.0.0.1:8040/v1/folds/FOLD_ID/results | python3 -m json.tool
curl -sS 'http://127.0.0.1:8040/v1/folds/FOLD_ID/logs?tail=100' | python3 -m json.tool
squeue -j SLURM_JOB_ID
```

Result paths are under `TORCHFOLD_API_WORK/<fold_id>/outputs/`. TorchFold typically emits mmCIF and ranking scores. The API reports whatever `.cif`, `.pdb`, `.json`, `.csv`, `.txt` files are actually produced. It does **not** fabricate an output PDB.

curl -sS 'http://127.0.0.1:8040/v1/folds/151dc8c503504a8ab880e3b7d43a5df4/logs?tail=100' | python3 -m json.tool