# TorchCraft API — FastAPI + Slurm + Apptainer

A lightweight submission API for TorchCraft protein design. FastAPI runs on a Slurm login/service node; GPU computation is submitted to Slurm and runs in a local Conda environment or an Apptainer SIF. Inputs, checkpoints, and output directories remain on shared `/lustre` storage.

## Supported design modes

| API `mode` | TorchCraft `--task_type` | Required request parameters |
|---|---|---|
| `mini_binder` | `mini_binder` | `pdb_path`, `target_full_seq`, `design_length` |
| `vhh` | `vhh` | `pdb_path`, `target_full_seq`, `framework_cif_path` |
| `monomer_unconditional` | `monomer` | `json_path`, `design_length` |

For binder modes, `hotspot_indices` is optional; use a comma-separated string such as `"31,55,99"` when appropriate. All modes accept `name`, `num_designs` (1–50), and `seed`. The API runs seeds `seed, seed+1, ...` sequentially within **one Slurm allocation**, not as a Slurm array. The upstream TorchCraft task runner currently also supports `scfv`, but this API does not expose it.

Upstream CLI: `torchcraft_gpu/task/run_task.py`. Monomer is implemented as `--task_type=monomer --json_path=... --design_length=...` (not `--task_type=monomer_unconditional`).

## 1. Files and directories

Place the updated `main.py` and this README into your existing API directory:

```text
/lustre/grp/cmclab/leiy/
├── TorchX/
├── torchcraft_api/
│   ├── main.py                 # replace with updated file
│   ├── README.md               # replace with updated file
│   ├── requirements-api.txt    # keep the existing file
│   ├── torchcraft.def          # keep the existing file
│   └── build_apptainer.sh      # keep the existing file
├── torchcraft_api_jobs/
└── containers/torchcraft.sif
```

The API process needs `fastapi`, `uvicorn`, `pydantic>=2`, `PyYAML`. Reuse the existing `requirements-api.txt` and install it with the API Python environment.

## 2. Runtime configuration

Set these environment variables **before starting the API**, ideally in `start_api.sh`. They are for API/Slurm submission, **not** for the Apptainer build:

```bash
export TORCHX_ROOT=/lustre/grp/cmclab/leiy/TorchX
export TORCHCRAFT_API_WORK=/lustre/grp/cmclab/leiy/torchcraft_api_jobs
export TORCHCRAFT_PARTITION=gpu3-12
export TORCHCRAFT_GRES=gpu:1
export TORCHCRAFT_CPUS=8
export TORCHCRAFT_MEM=32G
export TORCHCRAFT_TIME=12:00:00
export TORCHCRAFT_ALLOWED_ROOT=/lustre/grp/cmclab

# Set EXACTLY ONE, using a real path:
#export TORCHCRAFT_CHECKPOINT=/lustre/grp/cmclab/leiy/
#unset TORCHCRAFT_MODEL_DIR
# OR: unset TORCHCRAFT_CHECKPOINT;
export TORCHCRAFT_MODEL_DIR=/lustre/grp/cmclab/leiy/
```

Choose one execution mode:

**Apptainer inference:**

```bash
#export TORCHCRAFT_SIF=/lustre/grp/cmclab/leiy/containers/torchcraft.sif
```

**Native Conda inference:**

```bash
unset TORCHCRAFT_SIF
export TORCHCRAFT_PYTHON=/lustre/grp/cmclab/leiy/conda_envs/torchx_env/bin/python
```

In Apptainer mode, the SIF runs `/opt/conda/envs/torchx/bin/python` and the TorchX source embedded at `/opt/TorchX`. The Slurm script binds `/lustre:/lustre` for inputs, model weights, task config, and results. The wrapper sets `PYTHON_BIN` so the upstream `run_task.py` launches its child process using the correct Python.

Start API on a host with Slurm commands (`sbatch`, `squeue`, `sacct`) accessible:

```bash
cd /lustre/grp/cmclab/leiy/torchcraft_api
/lustre/grp/cmclab/leiy/conda_envs/torchx_env/bin/python -m pip install -r requirements-api.txt
/lustre/grp/cmclab/leiy/conda_envs/torchx_env/bin/python -m uvicorn main:app \
  --host 127.0.0.1 --port 8020
```
`127.0.0.1` is accessible only from that host (or through SSH port forwarding). Protect the API before exposing it over a network: it accepts file paths and submits compute jobs.

## 3. API examples

All requests use `POST /v1/designs` and return HTTP **202** with `design_id`, `slurm_job_id`, and state `PENDING`. Paths must already exist on the API host and be inside `TORCHCRAFT_ALLOWED_ROOT`.

### 3.1 Unconditional monomer (NEW)

```bash
curl -sS -X POST http://127.0.0.1:8020/v1/designs \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "monomer_unconditional",
    "name": "monomer_demo",
    "json_path": "/lustre/grp/cmclab/leiy/TorchX/torchcraft_gpu/task/monomer_unconditional/test-demo.json",
    "design_length": 100,
    "num_designs": 3,
    "seed": 1
  }' | python3 -m json.tool
```

This schedules 3 runs (`seed_1`, `seed_2`, `seed_3`). The input JSON is the **TorchCraft/TorchFold dialect JSON expected by the upstream monomer task**, not an arbitrary JSON file. In particular, an unconditional monomer request does not require PDB, target sequence, hotspots, or framework input.

### 3.2 Mini-binder

```bash
curl -sS -X POST http://127.0.0.1:8020/v1/designs \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "mini_binder",
    "name": "pdl1_test",
    "pdb_path": "/lustre/grp/cmclab/leiy/TorchX/torchcraft_gpu/task/mini_binder/input/pdl1_8znl.pdb",
    "hotspot_indices": "31,55,99,107",
    "target_full_seq": "/lustre/grp/cmclab/leiy/TorchX/torchcraft_gpu/task/mini_binder/input/input_sequence.csv",
    "design_length": 80,
    "num_designs": 1,
    "seed": 1
  }' | python3 -m json.tool
```

### 3.3 VHH

```bash
curl -sS -X POST http://127.0.0.1:8020/v1/designs \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "vhh",
    "name": "bhrf1_test",
    "pdb_path": "/lustre/grp/cmclab/leiy/TorchX/torchcraft_gpu/task/vhh/input/bhrf1_2wh6.pdb",
    "hotspot_indices": "63,70,79,86,105",
    "target_full_seq": "/lustre/grp/cmclab/leiy/TorchX/torchcraft_gpu/task/vhh/input/input_sequence.csv",
    "framework_cif_path": "/lustre/grp/cmclab/leiy/TorchX/torchcraft_gpu/task/vhh/input/frameworks/7eow_nanobody_framework.cif",
    "num_designs": 1,
    "seed": 1
  }' | python3 -m json.tool
```

### 3.4 Job status, results, and logs

Replace `DESIGN_ID` with the returned ID.

```bash
curl -sS http://127.0.0.1:8020/health | python3 -m json.tool
curl -sS http://127.0.0.1:8020/v1/designs/DESIGN_ID | python3 -m json.tool
curl -sS http://127.0.0.1:8020/v1/designs/DESIGN_ID/results | python3 -m json.tool
curl -sS 'http://127.0.0.1:8020/v1/designs/DESIGN_ID/logs?seed=1&tail=100' | python3 -m json.tool
```

The results endpoint reports generated `.pdb`, `.cif`, `.fa`, `.fasta`, and `.csv` files on shared storage. `state` is read from Slurm, and per-seed `.exit` files indicate task failures. This API returns paths, not file bytes. Paths are usable by another service only if its host/container can access the same shared storage.

## 4. Apptainer image

Existing `torchcraft.def` and `build_apptainer.sh` may be reused **if the image contains the upstream monomer task and model dependencies**. The upstream `monomer` runner lives under the same `TorchX/torchcraft_gpu/task/run_task.py` already embedded by your `%files` section. Adding a new API route does not by itself require rebuilding the image. If the SIF was built from an older TorchX checkout without monomer support, rebuild it.

```bash
cd /lustre/grp/cmclab/leiy/torchcraft_api
bash build_apptainer.sh
```

The build uses an OCI CUDA base plus conda/pip downloads; the HPC builder must support `--fakeroot` and have network access or local caches. Existing definition includes:

```bash
conda run -p /opt/conda/envs/torchx build_data
```

Confirm CCD assets inside the image after build:

```bash
apptainer exec /lustre/grp/cmclab/leiy/containers/torchcraft.sif \
  ls -lh /opt/TorchX/torchx_gpu/torchx/torchx/constants/converters/ccd.pickle
```

Check CUDA on a **Slurm-allocated GPU node**:

```bash
apptainer exec --nv --bind /lustre:/lustre \
  /lustre/grp/cmclab/leiy/containers/torchcraft.sif \
  /opt/conda/envs/torchx/bin/python -c 'import torch; print(torch.cuda.is_available())'
```

## 5. Notes and limitations

- The API server does not use a GPU; `sbatch` schedules jobs on configured GPU partition.
- Each `num_designs` request creates one Slurm job running seeds sequentially; increase to concurrent Slurm arrays later if needed.
- The `TORCHCRAFT_CHECKPOINT` path must contain **TorchCraft-compatible weights**; the README placeholder is not a verified real checkpoint.
- For security, the API restricts input and model paths to `TORCHCRAFT_ALLOWED_ROOT`. Keep the API behind authenticated access or an SSH tunnel.
- `TORCHCRAFT_SIF` takes precedence over native `TORCHCRAFT_PYTHON` for inference.
- `run_task.py` first loads `base.yaml`, then `{task_type}.yaml`; the API copies the entire config directory per design and overrides model weight location in the copied `base.yaml`.
- If you use native Conda and see `ccd.pickle` missing, run `build_data` in the native `torchx_env`; building CCD inside the SIF does not populate the host source tree.

Source references: [TorchX task runner](https://github.com/Mingchenchen/TorchX/blob/main/torchcraft_gpu/task/run_task.py) and [official unconditional monomer batch script](https://github.com/Mingchenchen/TorchX/blob/main/torchcraft_gpu/task/monomer_unconditional/batch_submit.sh).
