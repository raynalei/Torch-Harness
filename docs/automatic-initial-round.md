# Automatic PPIFlow initial round and shorter Qwen responses

The initial-round runner now supports the requested deterministic chain:

1. Submit PPIFlow generation and poll `GET /v1/designs/{id}`.
2. Read its returned structure manifest, score each structure with AF3Score,
   polling `GET /v1/tasks/{id}` and reading `/metrics` only after success.
3. Continue only candidates with **AF3Score ipTM strictly greater than**
   `min_af3score_iptm`.
4. Construct a TorchFold input from candidate chain sequences (or a supplied
   prepared template), submit refolding and poll `GET /v1/folds/{id}`.
5. Pair the chosen refold structure with its confidence output, verify chain
   sequences, and compute DockQ against the **PPIFlow-generated design pose**.
6. Export measurements and ranking. DockQ is design/refold consistency, not
   accuracy against an experimentally known complex or measured affinity.

The Python worker performs GET polling, exit-code checks and all transitions.
No Qwen calls or thinking tokens are consumed between stages. `PENDING`,
`RUNNING` and `UNKNOWN` remain waiting states; failures and nonzero exits stop
that job's advancement. Requests/results remain checkpointed so resuming a
submitted job uses its saved ID instead of POSTing it again. A lost submission
response remains uncertain and requires operator reconciliation, not blind retry.

TorchCraft is optional, and Rosetta runs only if `evaluation.rosetta` is present.
VHH/minibinder generator combinations remain supported by the runner; the only bundled deployment profile is the sbatch example. Every above-threshold
structure proceeds when `per_source_refold_limit: null` and
`deduplicate_sequences: false`; a top-K limit and sequence deduplication are
explicit optional policies. Below-threshold and exactly-equal candidates do not
submit TorchFold jobs. All-filtered runs finish with `no_hits`.

## Prepare once, then provide a PDB

Copy [the paired-antibody sbatch profile](examples/initial_round_sbatch.yaml). Replace
service endpoints/response fields, shared paths, chain IDs, framework and hotspot
policy with your already validated inputs. The example is for paired VH/VL antibody generation;
PPIFlow `binder` can be configured using its documented input/chain/length fields.
A PDB alone does not determine the binder format, framework or scientific epitope;
those decisions belong in this prepared profile. The PDB must be on the mounted
shared filesystem visible to the API and Slurm workers.

With `fold_template: null`, inputs use the AlphaFold3 JSON dialect, mapped chain
IDs and `fold_seeds`. By default `run_data_pipeline: false` supplies explicit empty
MSAs/templates, an intentional single-sequence refold; it is not equivalent to
MSA-supported prediction. With the pipeline enabled and no template, these fields
are omitted so the service can search its configured databases. A supplied
prepared template retains target features and is checked against the candidate's
target sequence; old binder features are cleared to avoid stale sequence MSAs.
Validate the generated JSON against the deployed TorchFold version before scaling.

```bash
uv run --frozen ddeharness protein-design initial-round \
  -c /actual/profiles/initial_round_sbatch.yaml \
  --pdb /lustre/project/inputs/my_target.pdb --name target_001 --check

uv run --frozen ddeharness protein-design initial-round \
  -c /actual/profiles/initial_round_sbatch.yaml \
  --pdb /lustre/project/inputs/my_target.pdb --name target_001 --detach
```

The first command submits no inference. The second returns `run_dir` and a worker
PID immediately. The API processes and worker host must remain available; the
worker survives the chat/TUI exiting, but cannot survive host shutdown or scheduler
termination of its own host allocation. Run it on a host permitted for persistent
lightweight orchestration. It submits compute through the APIs, not on the login
node itself.

Inspect saved progress (this does not make extra remote GET calls):

```bash
uv run --frozen ddeharness protein-design initial-round-status \
  --run-dir /lustre/project/harness/runs/target_001
```

Use the same start command to resume after an interrupted/failed worker. Existing
active/completed runs are returned rather than launched twice. Input content and
configuration changes require a new `--name`. A run lock also prevents a second
foreground/background runner from submitting the same stages concurrently.
Partial runs retain candidate errors and can be explicitly restarted to retry
result retrieval; persisted terminal failed remote jobs are not auto-resubmitted.
Logs are in `worker.log`; checkpoints, candidates, CSV and summary stay in the run
directory. Stopping the worker does not automatically cancel external Slurm jobs.
The launch host is recorded. Querying from a different login node reads the last
saved state without interpreting that node's unrelated PID table; investigate a
stalled worker on its recorded host before attempting recovery.

## Start from a Qwen conversation

Merge this property into `plugins.config.protein-design` in your active Harness
configuration, preserving the existing provider and service registrations:

```json
{
  "initial_round_config": "/actual/profiles/initial_round_sbatch.yaml"
}
```

Restart Harness. You can then say:

> Run the configured initial-round pipeline for `/lustre/project/inputs/my_target.pdb`,
> named `target_001`. I authorize the full configured generation, scoring, refold
> and DockQ workflow. Start it in the background and return the run directory.

Qwen calls `protein_initial_round_start` once. The worker advances independently;
`protein_initial_round_status` reads progress when requested. Do not have Qwen
submit each stage manually or poll in a loop. These are separate from the legacy
`protein_design_start/status` task controllers.

## Reduce actual Qwen thinking and output tokens

Hiding the thinking panel does not reduce model generation. The new per-model
fast profile writes the existing request controls and the proper Qwen payload
compatibility format, preserving API keys, addresses and other compatibility
settings:

```bash
uv run --frozen ddeharness provider model set YOUR_PROVIDER YOUR_MODEL --qwen-fast
```

Replace the two names with your configured provider ID and served model ID.
Default settings are `reasoning_effort: off`, `maxTokens: 2048`, and
`compat.thinkingFormat: qwen` for OpenAI-compatible Chat Completions endpoints.
`reasoning: true` records that this is a hybrid-capable model so the adapter
sends the explicit switch; it does not turn thinking on when effort is off.
The adapter sends **`enable_thinking: false`**. For a vLLM-style server:

```bash
uv run --frozen ddeharness provider model set YOUR_PROVIDER YOUR_MODEL \
  --qwen-fast --thinking-format qwen-chat-template --max-tokens 2048
```

That format sends `chat_template_kwargs.enable_thinking: false`. Restart Harness
for the changed model row to take effect. Output length can be adjusted with
`--max-tokens`; too small a ceiling can truncate tool arguments. Neither option
reduces already-sent input context. These switches require a hybrid Qwen model
and a server that implements the selected format; a reasoning-only deployment
cannot necessarily disable thinking. The local wire test verifies the flag and
ceiling are sent, not a vendor's compliance or a real model's latency.


## Natural-language launch without a YAML file

`protein_initial_round_start` accepts a complete structured `run_config` as an
alternative to `config_path` or a plugin `initial_round_config` profile. Describe
the task in chat; the agent assembles the object from existing registered service
metadata, known tool parameters and local Python results. You do not need to
write a YAML file. Chain/hotspot/framework and deployment inputs must still be
known; an API URL or framework path must not be invented. The usual initial
compute authorization applies; bounded follow-up rounds do not ask again.

For example: "Use the PDB, chains, hotspots and framework from the previous
request. Generate five candidates and run the scoring/refolding pipeline, with
at most three follow-up rounds under the existing sample budget."

The worker validates the object with the same `InitialRoundConfig` schema as a
profile and saves `worker-config.json` and `launch-inputs.json` for recovery.
An inline configuration and explicit `config_path` cannot be supplied together.
`protein_model_list` now includes safe service configuration metadata (credential
variable names only, never values), so the agent can reuse registered deployments.
YAML remains an optional reproducible CLI format.

## Release API allocations

Each sbatch-managed API allocation has a unique `harness-api-*` scheduler name
and an ownership record. A full workflow caches its scientific outputs and, when
all recorded inference jobs are terminal, requests `scancel` for only those API
allocations it owns. `api_cleanup` in the summary/checkpoint reports the outcome.
This stops the persistent API server allocation; it does not delete output files.

Standalone registered requests cache results/metrics when a successful status is
observed, then request release of their private API allocation. Later status and
results calls use the saved records without reconnecting to the stopped API.
There is no new independent polling daemon for standalone requests: cleanup is
triggered by status/results observation, while full workflows already poll in
their worker. Pending/uncertain inference, unverified completion, external APIs,
and unresolved TorchOptimization child jobs retain their APIs.

Cancellation is reported as `cancel_requested`; the scheduler may briefly show
`COMPLETING`. A scheduler-name mismatch prevents cancellation, even if a numeric
job ID has been reused. Startup records from older versions without ownership
names are not cancelled automatically. A hard-killed worker cannot execute its
finalizer; resume the same run to use its retained ownership records. Actual
Slurm/GPU deployment behavior must be verified on the cluster.
