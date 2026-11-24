# TorchScore API

A FastAPI/Slurm wrapper for [Mingchenchen/TorchX](https://github.com/Mingchenchen/TorchX),
compatible with the native `torchscore` model kind in Torch-Harness branch
`feat/initial_round`, commit `d0b299a` or later.
The TorchX entry points were checked at commit `ebf9dc6676ecb1be505bc72f87de408995f0d8fd`.
TorchScore and AF3Score are separate scoring backends. Register this API as
`kind: torchscore`; no AF3Score alias is needed.

## Files

```text
Torchscore-api/
├── README.md
├── main.py
└── torchscore-api.sbatch
```

`main.py` contains both the HTTP service and the checked inference worker.
The API starts without importing Torch/CUDA; its health endpoint confirms HTTP
availability, not successful model loading or GPU inference.

## Requirements

1. Install TorchX, TorchScore runtime dependencies, and chemical data following
   `TorchX/torchx_gpu/README.md` and `TorchX/torchscore_gpu/README.md`.
2. Install FastAPI, Uvicorn, and Pydantic v2 in the same Python environment:

   ```bash
   /path/to/torchx_env/bin/python -m pip install 'fastapi>=0.115' 'uvicorn>=0.30' 'pydantic>=2,<3'
   ```

3. All API and compute nodes must see the same source, API directory, job storage,
   model parameters, and input paths. Slurm commands must be available to the API.
4. Set exactly one weight source: `TORCHSCORE_MODEL_DIR` for official model
   parameters, or `TORCHSCORE_CHECKPOINT` for a TorchFold checkpoint.

## Configure and start

Deploy this directory to `/lustre/grp/cmclab/leiy/Torchscore-api`, or update the
working directory in `torchscore-api.sbatch`. Review every deployment-specific
path and partition before submitting. In particular, the example model directory
must contain actual model weights; its existence alone does not validate them.

| Variable | Purpose |
| --- | --- |
| `TORCHSCORE_ROOT` | TorchX repository root |
| `TORCHSCORE_PYTHON` | Absolute Python executable used by API and worker |
| `TORCHSCORE_API_WORK` | Persistent metadata, input snapshots, logs, and results |
| `TORCHSCORE_ALLOWED_INPUT_ROOTS` | Colon-separated permitted input/weight roots |
| `TORCHSCORE_MODEL_DIR` | Official model parameters directory; exclusive with checkpoint |
| `TORCHSCORE_CHECKPOINT` | Trained checkpoint file; exclusive with model directory |
| `TORCHSCORE_PARTITION` | Candidate partitions for scoring jobs |
| `TORCHSCORE_GRES` | Scoring-job GPU request, normally `gpu:1` |
| `TORCHSCORE_CPUS` | Scoring-job CPU count, default 8 |
| `TORCHSCORE_MEM` | Scoring-job host memory, default 64G |
| `TORCHSCORE_TIME` | Scoring-job time limit, default 12 hours |

```bash
mkdir -p /lustre/grp/cmclab/leiy/logs
cd /lustre/grp/cmclab/leiy/Torchscore-api
TMPDIR=/tmp sbatch torchscore-api.sbatch
squeue -u "$USER" -o '%.18i %.25j %.10T %.25N'
```

The API listens on **8060**. The outer API allocation does not request a GPU;
its CPU/memory allocation is independent from the inference subjob. Prefer an
allowed CPU partition for the API if available. Scoring remains a separate Slurm
GPU allocation and may queue. This package does not implement a preallocated GPU pool.

When Harness runs on another node, use `http://API_NODE:8060`, not localhost.
The supplied Torch Optimization API also uses 8060; use 8061 for Optimization
if both services run on the same node, and update its startup command accordingly.

## Register the native model in Torch-Harness

Save the following as a services JSON file, replacing the hostname and input roots:

```json
{
  "model_services": {
    "torchscore": {
      "kind": "torchscore",
      "service": {
        "url": "http://API_NODE:8060",
        "health_path": "/health",
        "status_field": "status",
        "exit_code_field": "exit_code",
        "require_exit_code": true,
        "files_field": "files",
        "metrics_field": "metrics",
        "metric_fields": {"iptm": "iptm"}
      },
      "max_samples": 1,
      "allowed_input_roots": ["/lustre/grp/cmclab"]
    }
  }
}
```

Register it in the actual Harness configuration and restart Harness:

```bash
ddeharness protein-design models register --services-file /path/to/services.json
ddeharness protein-design models list --check-health
```

The chat tool is **`protein_model_torchscore`**, with the usual
`protein_model_status` and `protein_model_results` tools. It submits a request
without waiting for inference. A registered endpoint is not proof of successful
scientific inference. Cancellation is not implemented by this API.

If Harness should start the API on demand, add these fields to its `service`
profile instead of manually starting it first:

```yaml
sbatch_script: /lustre/grp/cmclab/leiy/Torchscore-api/torchscore-api.sbatch
sbatch_resolve_host: true
startup_timeout: 1800
```

Use a localhost placeholder with the correct port in this on-demand profile;
Harness resolves the API allocation's hostname. Do not configure both manual
startup and on-demand startup for the same service/port.

## Select TorchScore in initial-round

Merge this fragment into a complete initial-round YAML, retaining the configured
generators, TorchFold, DockQ interpreter, path mappings, and optional Rosetta:

```yaml
evaluation:
  scoring_backend: torchscore
  torchscore:
    url: http://API_NODE:8060
    health_path: /health
    status_field: status
    exit_code_field: exit_code
    require_exit_code: true
    files_field: files
    metrics_field: metrics
    metric_fields:
      iptm: iptm
  torchscore_options:
    run_data_pipeline: false
    run_inference: true
    init_guess: true
    num_samples: 1
    num_workers: 4
    seed: 10
    write_fold_input_json_file: false
  min_torchscore_iptm: 0.5
```

Model registrations in global config and the workflow YAML are separate:
registration exposes chat tools; initial-round uses its own selected service.
The worker stores `metrics.torchscore_iptm`, `torchscore_results`,
`torchscore_input/`, and `torchscore-metrics.json`. It filters strictly above
`min_torchscore_iptm`, then applies source-specific deduplication/refold limits.
Do not assume TorchScore and AF3Score thresholds are scientifically interchangeable.
Use a new run name/directory when switching scorers; do not alter a submitted
AF3Score checkpoint to resume it under TorchScore.

## HTTP requests

### Score PDB structures

```bash
curl -fsS http://API_NODE:8060/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"mode":"pdb","input_pdb_dir":"/lustre/grp/cmclab/leiy/test_pdbs","num_workers":4}'
```

The directory must contain `.pdb` files. Their basenames may contain letters,
digits, underscores, periods, and hyphens, and must start with a letter or digit.
The API copies each PDB into a task-specific input snapshot.

POST persists metadata and returns **202** with `task_id`, `status: SUBMITTING`,
and `slurm_job_id: null` before calling `sbatch` in a background task. Query status
later for the actual Slurm job ID. No node-allocation wait occurs inside POST.

### Extract metrics from existing TorchScore outputs

```bash
curl -fsS http://API_NODE:8060/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"mode":"metrics","input_pdb_dir":"/lustre/grp/cmclab/leiy/test_pdbs","af3score_output_dir":"/lustre/grp/cmclab/leiy/existing/torchscore_outputs"}'
```

`af3score_output_dir` is a historical wire-field name retained by the native
TorchScore request schema. Its contents must be **TorchScore** raw outputs.
Metrics-only jobs request CPU/memory but omit GPU GRES; they still use the
configured `TORCHSCORE_PARTITION`.

Supported controls are deliberately limited to the verified upstream protocol:

```text
mode: pdb or metrics
seed: 10
num_samples: 1
init_guess: true
run_data_pipeline: false
write_fold_input_json_file: false
num_workers: 1..64
```

PDB mode requires `run_inference=true`; metrics mode turns inference off.
Prepared mode, MSA/database searches, and pipeline-only execution are unsupported.
`run_data_pipeline=false` does not skip PDB-to-CIF/JSON/H5 preparation. Upstream
preparation builds single-sequence MSAs and templates from each input structure.

### Inspect status, metrics, results, and logs

```bash
curl -fsS http://API_NODE:8060/health
curl -fsS http://API_NODE:8060/v1/tasks/TASK_ID
curl -fsS http://API_NODE:8060/v1/tasks/TASK_ID/metrics
curl -fsS http://API_NODE:8060/v1/tasks/TASK_ID/results
curl -fsS 'http://API_NODE:8060/v1/tasks/TASK_ID/logs?tail=100'
```

Status includes `status`, `state`, `exit_code`, `slurm_job_id`, and errors.
Metrics/results return 409 until validated completion. Metrics are returned as
`{"metrics":[{"description":"candidate","iptm":0.8,"ptm":0.9,...}]}`.
One request may contain multiple structures; each row corresponds to a PDB stem.
Initial-round submits one candidate per scoring task, satisfying its scalar gate.

## Outputs and correctness checks

```text
$TORCHSCORE_API_WORK/<task_id>/
├── metadata.json
├── inputs/*.pdb
├── run.sh
├── single_chain_cif/, json/, single_seq.csv
├── tf_input_batch/{json,pdb,jax}/
├── torchscore_outputs/<name>/seed-10_sample-0/
│   ├── confidences.json
│   └── summary_confidences.json
├── torchscore_metrics.csv
├── result.json
├── exit_code.txt
├── worker.log
└── slurm-<job_id>.{out,err}
```

Metrics-only mode reads external raw outputs instead of creating a new
`torchscore_outputs` directory. It still saves its CSV/result/logs under its own task.

The worker calls upstream preparation, H5 conversion, inference, and metrics
extraction scripts directly with checked exit codes. It does not reuse the
original shell's unconditional completion markers. It checks input coverage,
required JSON/H5 outputs, exactly one metric row per input, finite ptm/iptm in
[0,1], and agreement between CSV values and actual summary confidences.
Optional nonfinite CSV fields become JSON null. Missing scores are never replaced
with fabricated zeros. Slurm completion alone is insufficient for success.
Inference is sequential within a single-GPU job and preserves Slurm's
`CUDA_VISIBLE_DEVICES`; CPU preprocessing is limited by the allocated CPU count.

## Retry and recovery

An optional `Idempotency-Key` header must be 32 lowercase hexadecimal characters.
It becomes the task ID. Reusing the same key and identical request body returns
the existing task without another Slurm submission; a conflicting body returns 409.

Background submission runs inside the API process, not a durable external queue.
If the process dies or sbatch times out during submission, metadata may remain
SUBMITTING/SUBMISSION_UNCERTAIN. Inspect Slurm before recovery; this API never
blindly resubmits uncertain tasks. Preserve job directories when restarting.

Current Harness automatic key/hash reconciliation applies to `/v1/designs`, not
this `/v1/tasks` endpoint. For an uncertain initial-round submission, identify the
existing API task and resume with its `task_id`, not the API allocation's Slurm ID:

```bash
ddeharness protein-design initial-round -c /path/to/original.yaml \
  --detach --attach-job 'score/CANDIDATE_ID=VERIFIED_TASK_ID'
```

Use the exact checkpoint key and unchanged original configuration, and stop the
old worker before attaching. Reconciliation of an uncertain *server-side sbatch*
must happen before resuming; attaching a task ID cannot repair its missing Slurm ID.

## Validation and deployment limits

Protocol tests use mocked Slurm/model processes, including native Harness
TorchScore request validation, input coverage, metric integrity, failures,
idempotency, and response-before-submission ordering. They do not establish GPU,
weights, CUDA/C++ extension, or scientific model compatibility on your cluster.
Run one real complex before launching a full workflow.

This API assumes a trusted cluster network and has no built-in authentication.
Do not expose it directly to the public Internet.
