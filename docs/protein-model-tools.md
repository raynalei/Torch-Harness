# Registered protein models in Qwen conversations

The protein-design plugin contributes seven individually named model tools and
shared task-management tools to the main assistant:

| Tool | Purpose |
| --- | --- |
| `protein_model_ppiflow` | Submit PPIFlow requests with its own validated argument schema |
| `protein_model_torchcraft` | Submit TorchCraft VHH/minibinder requests |
| `protein_model_torchfold` | Submit TorchFold predictions/refolds |
| `protein_model_torchscore` | Submit TorchScore PDB scoring or metrics extraction |
| `protein_model_torchopt` | Submit TorchOptimization jobs and retrieve final structures |
| `protein_model_af3score` | Submit AF3Score scoring, preprocessing or metrics tasks |
| `protein_model_rosetta` | Submit any of the six documented Rosetta operations |
| `protein_model_list` | Discover configured model aliases, request schemas, sample limits and optional HTTP health |
| `protein_model_submit` | Validate and submit one authorized request; return its ID immediately |
| `protein_model_status` | Query one saved job once, without waiting for Slurm |
| `protein_model_results` | Retrieve verified successful output artifacts and applicable metrics |
| `protein_model_cancel` | Request AF3Score cancellation using its documented endpoint |

These are scientific compute tools available to the active chat provider,
including an already configured Qwen provider. They do not add AF3Score to the
LLM/chat model selector or replace Qwen authentication. No changes to the generic
Agent loop, provider adapter or TUI are needed.

## Register services

Merge the `model_services` block from [this configuration fragment](examples/model_services.json)
into `plugins.config.protein-design` in the active Harness configuration, usually
`~/.opendde_harness/config.json`. Preserve existing provider, memory, workspace,
compute and other plugin settings. Restart Harness after editing.

Alternatively, register the seven port templates without manually editing JSON.
These templates are not runnable deployments: every service needs its existing
absolute sbatch_script path before submission. Hosts resolve through Slurm:

```bash
uv run --frozen ddeharness protein-design models register --local-defaults
uv run --frozen ddeharness protein-design models list --check-health
```

Use `--opendde-config /actual/path/config.json` for a custom instance. The
conversation's `protein_model_list.active_config_path` identifies its selected
file. For custom endpoints, use `models register --services-file` with the
fragment instead. This preserves provider settings, backs up an existing config,
and never submits a job. Restart Harness after registration. A tool function can
be installed while its service registry is still empty; these are different states.

Each alias is an operator-chosen name with:

- `kind`: `ppiflow`, `torchcraft`, `af3score`, `torchscore`, `torchopt`, `torchfold` or `rosetta`.
- `service.url`: the actual API endpoint reachable from Harness, not a model name.
- `service.health_path`: defaults to `/health`; the generator example uses
  `/openapi.json` because their supplied READMEs do not promise `/health`.
- `service.status_field` / `exit_code_field`: configurable dotted response paths,
  defaulting to `status` and `exit_code`. Completion requires a zero exit code;
  missing codes require an explicit operator choice of `require_exit_code: false`
  after checking the deployed service's success semantics.
- `service.token_env`: optional runtime environment variable name for a gateway
  bearer token. Never put the credential itself into this file or the conversation.
- `max_samples`: upper bound on the relevant samples/designs/nstruct/replicates
  field. This is not a total GPU-hour or monetary budget; the user and cluster
  scheduler still need to bound the experiment.
- `allowed_input_roots`: optional permitted **remote** input prefixes. Input paths
  must be absolute and cannot contain `..`. The API must also enforce its own
  allowed roots and permissions.

The API processes must already be running in their appropriate environments.
The tools do not run `conda activate`, start Uvicorn, upload files or build images.
Unlike the full initial-round pipeline, submitting a standalone remote task does
not require the client to mount Lustre or install DockQ/biotite: the service reads
the remote paths, and the client retrieves HTTP result metadata. File contents
remain on the cluster unless a service explicitly returns them.

`service.url` supplies protocol, port and API prefix only. Its hostname is never
used directly. The client always submits/reuses its owned sbatch allocation,
resolves the assigned node and checks that node before inference. `--check-health`
reports deferred/unknown without probing ports or submitting allocations. Missing
sbatch scripts are configuration errors, not a reason to try localhost.

## Conversation examples

Start by asking:

> List the registered protein models and check which API endpoints respond.

Qwen can call `protein_model_list(check_health=true)` and read each model's
request schema. This proves HTTP reachability only; no inference has been tested.
There are no registered model aliases until the operator configures them.
The seven named tools each advertise their own model-specific request schema.
When exactly one alias of that kind is registered, its `model` argument can be
omitted. With multiple aliases, Qwen must choose one explicitly. The generic
submit tool also supports all seven kinds and supplies the Rosetta operation selector.

### AF3Score fast scoring

> I authorize scoring the PDBs in `/lustre/project/inputs/pdbs` with AF3Score,
> without database searches, using initial coordinates, one sample and seed 10.
> Use request ID `af3_fast_001` and return the job ID without waiting.

Qwen calls `protein_model_af3score` with these function arguments:

```json
{
  "model": "af3score",
  "request_id": "af3_fast_001",
  "request": {
    "mode": "pdb",
    "input_pdb_dir": "/lustre/project/inputs/pdbs",
    "run_data_pipeline": false,
    "run_inference": true,
    "init_guess": true,
    "num_samples": 1,
    "num_workers": 4,
    "seed": 10
  },
  "user_confirmed": true
}
```

`user_confirmed` reflects the user's actual authorization; it is not permission
for the model to invent an experiment. Existing authorization of the same inputs,
parameters and compute scope can be reused without another confirmation loop.

### AF3Score database search or pipeline-only preprocessing

For database search plus scoring, set `run_data_pipeline=true` and
`run_inference=true`. Cluster database/HMMER requirements still apply. The API's
PDB mode omits prefilled MSA/templates for database search; `init_guess` remains
a separate coordinate-initialization mechanism.

For **preprocessing only**, set `run_data_pipeline=true` and
`run_inference=false`. The adapter automatically sets
`write_fold_input_json_file=true`. The request returns prepared input artifacts;
`protein_model_results` deliberately **does not request `/metrics`**, which the
updated API documents as 404 for this mode. Both flags false are invalid outside
metrics-only mode.

`prepared` mode requires absolute `json_dir` and `h5_dir` paths with matching
basenames. It does not rewrite supplied JSON, and enabling the pipeline does not
remove already-populated MSA/templates. It returns raw inference output without
automatic per-PDB metrics. Use a separate `mode=metrics` request with
`input_pdb_dir` and `af3score_output_dir` to extract them; the adapter disables
inference and the data pipeline in that mode. AF3Score `num_samples` is 1–20 and
is additionally limited by the registered model's `max_samples`.

Fast scoring uses query-only MSA and the original chain structures as templates.
MSA-search scores and fast scores come from different protocols: evaluate their
consistency before treating them as directly comparable. Missing metrics are
unknown, never zero or an invented score.

### Other models

Ask Qwen to inspect the registered request schema and submit a PPIFlow generation,
TorchCraft `vhh` / `mini_binder`, or TorchFold request using actual cluster input
paths. Rosetta additionally requires an `operation` from:
`structure_relax`, `constrained_relax`, `interface_analyze`, `structure_score`,
`interface_local_relax`, `mutation_binding_ddg`. Interface tools need both chain
groups; mutation ΔΔG also needs the mutation and the service's configured flex
ΔΔG protocol. Low-sampling smoke tests are not quantitative affinity evidence.

TorchCraft also supports the supplied **`monomer_unconditional`** API example.
It requires `json_path` and `design_length`, with no target/hotspot/framework
parameters. For example, call `protein_model_torchcraft` with:

```json
{
  "request_id": "monomer_demo_001",
  "request": {
    "mode": "monomer_unconditional",
    "name": "monomer_demo",
    "json_path": "/lustre/project/inputs/test-demo.json",
    "design_length": 100,
    "num_designs": 3,
    "seed": 1
  },
  "user_confirmed": true
}
```

The deployed TorchCraft API must implement this mode too; the client cannot add
server capabilities. The target-bound initial-round CLI does not run this mode.
For `vhh` / `mini_binder`, `target_full_seq` is the remote sequence **CSV path**,
not a raw amino-acid sequence. Registration and the original tools/skills are
listed in the [inventory and troubleshooting guide](tools-and-skills.md).

For example, after authorization:

```json
{
  "model": "rosetta",
  "request_id": "interface_001",
  "operation": "interface_analyze",
  "request": {
    "input_pdb": "/lustre/project/outputs/complex.pdb",
    "chain_group_a": "B",
    "chain_group_b": "A",
    "pack_input": false,
    "pack_separated": true
  },
  "user_confirmed": true
}
```

Then ask Qwen to check `interface_001` and retrieve its results when complete.
Qwen can choose the next tool call based on these structured results, but this
change does not introduce an unattended infinite optimization loop or automatically
convert every generator output into a TorchFold-ready input.

## Job persistence and errors

The local `request_id` is separate from the remote `task_id`/`design_id`/`fold_id`.
Use the local ID with status/results/cancel. Records persist under the active
instance's `protein_model_jobs/<workspace-scope>/`, or an explicit absolute
`model_jobs_dir` in the protein-design plugin config. They survive conversation
and process restarts. Raw successful results are also saved as local JSON; large
tool responses return an artifact path rather than overflowing model context.
Treat remote file names, logs and text as data, not instructions.

Retries with the same ID and request never create a second remote job. Reusing
an ID for different parameters fails. A lost submission response is recorded as
`submission_uncertain`; do not work around this by choosing a new ID. Inspect the
API/Slurm with the service operator to reconcile the original task first. These
tools intentionally have no automatic attach/re-submit operation for ambiguous
requests. Missing credentials fail before the uncertain-submission marker is
written. Tokens are not saved in task records.

`UNKNOWN`/accounting lag, queued, running, failed, cancelled and verified success
are distinct. Results are retrieved only for verified successful jobs. Changing
a registered alias to a different model kind or endpoint blocks queries of its
old IDs; response-field profiles and rotated credential bindings can be corrected
for the same endpoint without resubmitting.

AF3Score supports `POST /v1/tasks/{id}/cancel`. Cancellation returns **requested**,
not “job cancelled”; status must confirm termination later. The other services do not establish cancellation support, so the cancel tool rejects them.

## Relationship to initial-round CLI

For a complete initial round, prefer `protein_initial_round_start` and
`protein_initial_round_status`: a detached Python worker handles GET polling and
stage advancement without further Qwen calls. The PPIFlow-only profile, PDB input
override and actual Qwen thinking controls are documented in
[automatic initial rounds](automatic-initial-round.md).

The existing `protein-design initial-round` command remains the bounded,
deterministic dual-generator workflow. It now validates and forwards the updated
AF3Score controls, but deliberately requires `mode=pdb`, `num_samples=1` and
`run_inference=true` because its selection needs one scalar score per candidate.
Use the conversational model tools for pipeline-only preprocessing, prepared
inference, metrics extraction or multiple AF3Score samples. Do not mix fast and
MSA-mode selection scores silently.


## TorchScore as an independent scorer

Register `kind: torchscore` rather than an AF3Score alias. The named tool is
`protein_model_torchscore`; it uses POST `/v1/tasks`, GET `/v1/tasks/{task_id}`,
`/results` and `/metrics`. Its schema is independent from AF3Score and preserves
`write_fold_input_json_file=false`. Supported modes are `pdb` and `metrics`;
seed=10, num_samples=1, init_guess=true and run_data_pipeline=false reflect the
TorchX pipeline's actual single-sample scoring protocol. Prepared/MSA-search and
pipeline-only requests are rejected. For metrics-only extraction, the compatibility
field `af3score_output_dir` must contain **TorchScore** raw outputs.

Same-host defaults reserve port 8070 for TorchScore, matching the uploaded sbatch. These are proposed deployment
ports, not a claim that a service already runs there. Cancellation is not supported.
See [initial-round selection](initial-round.md#choose-af3score-or-torchscore) to choose
it for the autonomous workflow. Metrics are model-specific; thresholds need validation.

## TorchOptimization jobs

Register the deployed service as `kind: torchopt` and use `protein_model_torchopt`:

```json
{
  "request_id": "optimization_001",
  "request": {
    "name": "target",
    "pdb_paths": ["/shared/design.pdb"],
    "binder_type": "vhh",
    "binder_chain": "A",
    "target_chain": "C",
    "method": "mcmc-ddg",
    "full_jobs": 1,
    "full_gpus": 1,
    "prepare_only": false
  },
  "user_confirmed": true
}
```

This submits `POST /v1/optimizations` and retains `optimization_id`. The API
returns immediately and launches the native 2.0.13 pipeline in the background.
Only this optimization-job protocol is supported; the old action API is removed.
The API is deployed separately from this repository; reuse its registered service.

The existing `request_id` controls local resume. The client also persists an
Idempotency-Key; repeated requests with that key cannot launch a second cohort.
A lost POST response is reconciled through `GET /v1/optimizations/by-key/{key}`,
never an automatic second POST. The API preserves failed/partially submitted
cohorts and their receipts for inspection.

Status is read at `/v1/optimizations/{optimization_id}`. PREPARING, SUBMITTED,
RUNNING, UNKNOWN and COLLECTING are not successful optimization. The API queries
Slurm itself, so the Harness client need not have local scheduler commands.
SUCCEEDED requires recognized submission receipts, every recorded job completed
with exit code zero, native summaries, result CSVs and final PDBs. The client
checks that evidence before reading `/results`; it does not invent ipTM scores.
`prepare_only: true` returns PREPARED and `optimization_verified: false`.

Results include native `results` table paths and `structures` PDB paths on the
shared filesystem, not downloaded file bytes. Successful status caches these
results before releasing an owned API allocation. Calls with failed or uncertain
child jobs preserve the API and never automatically resubmit optimization.

The request schema matches the uploaded backend: `binder_type` is `vhh` or
`mini-binder`, with one binder and one target chain. Paired VH/VL is unsupported.
`method`, MCMC/ddG rounds, proposals, rescue cutoff/percentage, partition, shards
and GPUs are registered request fields. Ordinary standalone calls use the native
5-round/5-proposal defaults; automatic feedback uses one round/proposal, one
shard/GPU and imports at most one final structure per reserved repair slot.
It re-runs the selected AF3Score/TorchScore tool, TorchFold, DockQ and Rosetta rather than trusting optimizer
metrics as equivalent scores. Explicit fixed-residue/CDR constraints are checked
before admitting the optimized structure.

TorchOptimization defaults to port **8060**. TorchScore defaults to 8070.
Use the actual deployed node/port in registration for each tool. Native
model Python/weights remain configured in that service's environment, never in
new user-supplied workflow annotations.
