# TorchOptimization FastAPI + Slurm

This API uses the shared TorchOptimization v2.0.13 deployment documented in `multi_target_pipeline_v2.0.13_shared_delivery`. Each POST creates an isolated input snapshot, control file and run directory. The original v2.0.13 launcher handles Slurm sharding, cohort-wide MCMC selection and independent ddG rescue.

## Shared environment and paths

| Setting | Default |
| --- | --- |
| `TORCHOPT_ROOT` | `/lustre/grp/cmclab/share/huangjm/Project/TorchOptimization_v2.0.13` |
| `TORCHOPT_PYTHON` | `/lustre/grp/cmclab/share/wangmm/Optimization/env/torchx_env/bin/python` |
| `TORCHOPT_MODEL_DIR` | `$TORCHOPT_ROOT/assets/torchcraft` |
| `TORCHOPT_RDE_DIR` | `$TORCHOPT_ROOT/assets/RDE-PPI` |
| `TORCHOPT_API_WORK` | `jobs/` inside your API folder |
| `TORCHOPT_ALLOWED_ROOT` | `/lustre` |

The shared source, environment and model weights stay in place. Copy only this API folder to your own writable work directory on Lustre. The API host must support Slurm submission, and compute nodes must see the shared backend and your work directory at the same absolute paths.

## Deploy the API service with Slurm

Copy this API folder to `/lustre/grp/cmclab/leiy/torchoptimization-api`.
The service uses the same API environment as PPIFlow:
`/lustre/grp/cmclab/leiy/conda_envs/ppi_env/bin/python`.
Optimization jobs continue to use the shared `torchx_env` listed above.
No additional virtual environment is needed.

Install the API dependencies once if they are not already available, then submit
from a Slurm submission host:

```bash
cd /lustre/grp/cmclab/leiy/torchoptimization-api
/lustre/grp/cmclab/leiy/conda_envs/ppi_env/bin/python 
sbatch run_api.sbatch
```

The log directory must exist before submission because Slurm opens the output
file before the script starts. `run_api.sbatch` requests partition `gpu51`, one
CPU, 32 GB memory and 12 hours for the API service. It does not request a GPU.
The script sets the shared model paths, stores tasks in
`/lustre/grp/cmclab/leiy/torchoptimization_api_jobs`, and starts Uvicorn on
`0.0.0.0:8040`. It also sets `TORCHOPT_PARTITION=gpu51` for optimization jobs;
an explicit request `partition` overrides that default.

Find the service node and inspect its log:

```bash
squeue -j <api_job_id> -o "%.18i %.10T %.30N"
tail -f /lustre/grp/cmclab/leiy/logs/torchoptimization-api-<api_job_id>.out
curl http://<api_node>:8060/health
```

Use `http://<api_node>:8060` for API requests from another reachable host.
The localhost examples below apply when running on the API node. The cluster
must permit network access to that node and Slurm submission from inside the
API allocation. Place the service behind authenticated access if exposing it
outside the trusted cluster network. The service stops when its 12-hour
allocation ends; already submitted optimization jobs remain independent.

The shared model environment and weights must already be prepared. Environment
variables are read when the service starts; resubmit the API job after editing
its configuration. The API passes the shared Python and model/RDE paths to the
backend launcher.

## Prepare and submit

In a second terminal, prepare a single-example request:

```bash
curl -X POST http://127.0.0.1:8060/v1/optimizations \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "PDL1",
    "pdb_paths": ["/lustre/grp/cmclab/share/huangjm/Project/TorchOptimization_v2.0.13/examples/mini-binder/PDL1_8ZNL.pdb"],
    "binder_type": "mini-binder",
    "binder_chain": "A",
    "target_chain": "C",
    "method": "mcmc-ddg",
    "full_jobs": 1,
    "full_gpus": 1,
    "prepare_only": true
  }'
```

All input paths refer to files on the server. Replace the example path with your own PDB paths as needed. Check chain identifiers against each input structure. `prepare_only: true` generates run directories and Slurm scripts without submitting jobs, but still requires valid model resources.

Inspect `jobs/<optimization_id>/submission.log` and `runs/`. For actual submission, send a new request with `prepare_only: false` and a valid cluster `partition` if required. Every POST creates a new task; it does not submit a previously prepared task.

Defaults are 5 MCMC rounds with 5 proposals per round, at most 5 ddG rounds, a rescue iPTM cutoff of 0.8 and the bottom 25% of the eligible low-iPTM subset. `full_jobs` controls sharding and `full_gpus` is the GPU count per job. The shared pipeline template uses 30 shards; this API defaults to 1 for the initial test. Set `full_jobs: 30` explicitly when appropriate for a full batch.

Submit separate requests for separate targets. All PDBs in one request form one independent rescue-selection cohort. The `name` field is a descriptive label; the UUID is the actual task directory identifier. This API uses its own `jobs/<UUID>/runs/` layout rather than the shared delivery's `0_inputs/`, `1_outputs/` and `group_###` layout.

## Status, logs and results

Use the returned `optimization_id`:

```text
GET /health
GET /v1/optimizations/{optimization_id}
GET /v1/optimizations/{optimization_id}/logs
GET /v1/optimizations/{optimization_id}/results
```

The status endpoint returns submission metadata, native summaries and Slurm submission receipts. `SUBMITTED` means the launcher completed submission; it does not mean optimization finished. Query `squeue` or `sacct` with the job IDs in the receipts for scheduler status. Logs return the final 64 KiB of each file. Results return server paths to CSV tables and final PDB files.

Each task retains its input copies, configuration, submission log and run outputs. Code and resources are linked to the shared backend, and the native launcher snapshots code into each run. Keep the shared backend and model environment available throughout execution.

A failed submission can leave successfully submitted jobs behind. Inspect the retained receipts and queue before any manual recovery. There is no automatic submission retry. This API does not expose the shared delivery's collect/export commands; its results endpoint lists native output files.

## Validation

Python syntax, shell syntax and the configured path defaults were checked locally. Linux GPU execution, shared filesystem access and Slurm submission require validation on the cluster. Start with preparation, then run a single-sample GPU test before a full batch.
