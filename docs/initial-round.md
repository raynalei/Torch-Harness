# Dual-generator initial round

For the PPIFlow-only autonomous chain (PPIFlow -> AF3Score threshold -> TorchFold
-> DockQ), use the [automatic initial-round guide](automatic-initial-round.md).
TorchCraft and Rosetta are now optional; the dual-generator example below remains
supported. `fold_template: null` builds the fold input automatically, while a
prepared template can still be supplied. An unset/null refold limit admits every
above-threshold candidate; sequence deduplication remains configurable.

`ddeharness protein-design initial-round` runs a deterministic initial design
round without LLM credentials or a local GPU. It is separate from the existing
`protein-design start` evolutionary workflow and uses the supplied PPIFlow,
TorchCraft, AF3Score, TorchFold and Rosetta HTTP services.

The sequence is: generate with **both** sources, score each candidate with
AF3Score, select a bounded distinct-sequence shortlist **per source**, refold with
TorchFold, evaluate design/refold consistency with DockQ, run Rosetta interface
analysis, and export candidates and rankings. Candidates are processed
sequentially to avoid flooding Slurm. Large production batches need separately
measured quotas and throughput; begin with small samples.

## Installation

```bash
uv sync --locked --extra dev --extra initial-round
# DockQ 2.1.3 requires NumPy <2, while Harness requires NumPy >=2.
# Install it as an isolated tool, not inside the Harness environment.
UV_TOOL_DIR=/lustre/project/tools UV_TOOL_BIN_DIR=/lustre/project/tools/bin \
  uv tool install --python 3.12 'DockQ==2.1.3'
```

DockQ builds a native extension and needs a C compiler. If the Python toolchain
expects an unavailable clang and GCC is installed, prefix the tool installation
with `CC=gcc CXX=g++`. Set `evaluation.dockq_python` to the installed tool
environment's interpreter, `/lustre/project/tools/dockq/bin/python` in this
example. Neither CUDA nor the model dependencies belong in the client environment.

## Configuration and execution

Copy [the example](examples/initial_round.yaml) to a local, reviewed configuration.
Replace the service URLs, paths, output chain IDs and hotspot indices. The example
uses VHH: PPIFlow `nanobody` plus TorchCraft `vhh`. For minibinders use PPIFlow
`binder` with `input_pdb`, `target_chain`, `binder_chain`, `samples_min_length`,
`samples_max_length`; pair it with TorchCraft `mini_binder`, providing `pdb_path`,
`hotspot_indices`, `target_full_seq`, and `design_length`. Requests otherwise pass
through to the documented services. Paired antibodies and partial flow are not
part of this initial-round command.

```bash
uv run --frozen ddeharness protein-design initial-round -c initial.yaml --check
uv run --frozen ddeharness protein-design initial-round -c initial.yaml
# Resume using the exact same command/configuration.
```

The command runs in the foreground. Use a persistent terminal on the submission
host for long jobs. `--check` validates local inputs, the isolated DockQ import,
and API health responses without submitting inference jobs; it does not prove
GPU/model readiness. Exit codes: 0 completed/check passed, 1 configuration or
round-level error, 2 partial completion with candidate errors. Zero accepted
candidates is a legitimate completed `no_hits` outcome, not a fabricated success.

All services must be reachable from the runner. The README localhost addresses
refer to the API host, not to this cloud container. AF3Score and TorchFold both
use 8040 in the supplied READMEs: allocate different ports if colocated. Keep
these unauthenticated APIs on a trusted network or behind an authenticated
gateway; optional `service.token_env` reads a bearer credential from the named
runtime variable, never from YAML. Preserve the platform's networking policy.

### Shared artifacts

The runner must have a readable/writable mount of the shared HPC filesystem.
There is no implicit upload, SSH transfer or interpretation of remote paths as
local paths. `path_mappings` translate mounted local prefixes into API-host
prefixes, for example `/mnt/hpc/project` to `/lustre/project`. `work_dir` is a
**local absolute** path inside a mapping, and inputs in generator requests are
**remote absolute** paths. `fold_template` and `dockq_python` are local absolute
paths. Output paths outside the declared roots, including symlink escapes, fail.

Each run uses `work_dir/name`; do not share it between different experiments.
Configuration and input content hashes bind a submitted run. Changing model
parameters, templates, input files or thresholds after submission requires a new
name/directory. An API-health-only check with no submitted jobs can be reconfigured.

### Response profiles

The READMEs do not specify complete response schemas. Defaults below are explicit
adapter contracts, **not claims that the cluster already returns these shapes**.
Check the deployed OpenAPI and actual output samples before running a full batch.
`health_path` defaults to `/health`; the example uses `/openapi.json` for the two
generators because their READMEs do not document a health route. An OpenAPI probe
proves HTTP reachability, not model or scheduler readiness.
Each service can configure dotted `status_field`, `exit_code_field`,
`files_field`, `metrics_field` and `metric_fields` paths. For example,
`status_field: slurm.state` and `metrics_field: rows`.

- Submission IDs: PPIFlow/TorchCraft `design_id`, AF3Score/Rosetta `task_id`,
  TorchFold `fold_id`. Rosetta POSTs to `/v1/tools/interface_analyze` but polls
  `/v1/tasks/{id}`; the other status URLs follow their submission collections.
- Default status: `{"status":"COMPLETED","exit_code":"0:0"}`. `UNKNOWN`, queue
  and running states continue polling. Failed Slurm states and nonzero exit codes
  fail. Missing exit codes fail by default. A service that genuinely does not
  expose exit codes needs an explicit reviewed `require_exit_code: false`.
- Default `/results`: `{"files":["/lustre/.../model.pdb"]}`; entries may also be
  `{"path":"/lustre/.../model.pdb"}`. No directory-wide discovery of potentially
  stale results is performed. Generator files are filtered by `structure_glob`;
  configure it to exclude intermediate PDBs. Alternatively, `candidates_field`
  selects a manifest list whose entries have `structure_path`.
- AF3Score submits one isolated PDB directory and one sample per candidate, then
  reads `/v1/tasks/{id}/metrics`. Default: `{"metrics":[{"iptm":0.9}]}`. Exactly
  one scalar measurement is required; ambiguous rows are errors.
  `evaluation.af3score_options` accepts `run_data_pipeline`, `run_inference`,
  `init_guess`, `num_workers`, `seed` and `write_fold_input_json_file`. The initial
  round requires inference and one sample; pipeline-only/prepared/multi-sample
  tasks are available through the [registered model tools](protein-model-tools.md).
- TorchFold: select the highest-ipTM sample with paired `*summary_confidences.json`
  and `*model.cif`/`*model.pdb` in the same folder. Every selected file must appear
  in `/results`. For another output layout, configure `structure_field` to select
  a remote structure path and return one scalar ipTM in the configured metrics
  object. This avoids guessing which confidence score belongs to which structure.
- Rosetta: default `{"metrics":{"dG_separated":-12.0}}`, or a listed `*.sc`
  score table with one measurement. Configure `rosetta_scores_glob` to select a
  CSV instead if needed. Missing or ambiguous energy values fail.

Do not fabricate measurements when a service only reports file paths. Collect
actual output files, set the profile, or extend the service's result manifest.

### TorchFold inputs and scientific interpretation

Provide a valid prepared AlphaFold3-dialect JSON template for each generator.
Use one protein entity per chain with scalar `id`, and include seeds and the
version required by your TorchFold deployment. Its IDs must exactly match
`chains.refold_map` values. The mapping is **generated chain -> refold chain**.
The generated target sequence must match the prepared target; cropped or missing
residues need a matching template. No implicit residue renumbering is performed.

Binder sequences are replaced from generated structures; old binder MSAs and
structural templates are cleared. Target MSAs are retained. With
`run_data_pipeline: false`, prepare the target MSA/features beforehand (or supply
explicit empty MSAs for an intentionally single-sequence run). Enabling the
pipeline requires the cluster service's configured databases/binaries. Embedded
JSON `*Path` fields must already use absolute remote paths.

`af3score_iptm` and `torchfold_iptm` remain distinct. Refold chain sequences are
checked before comparing structures. DockQ evaluates only target/binder
interfaces, excluding intrabinder interfaces, using the generated pose as the
reference; the minimum scored target/binder interface DockQ is retained. This is
**design/refold consistency**, not agreement with an experimental structure.
Rosetta `dG_separated` is an energy score, not measured affinity.

AF3Score thresholding precedes selection. Within each source, candidates are
ordered by AF3Score ipTM and sequence duplicates removed until the source's
refold limit is reached. Accepted candidates pass both refold-ipTM and DockQ
thresholds and have a finite Rosetta interface score. Ranking is lexicographic:
refold ipTM descending, DockQ descending, Rosetta interface energy ascending;
no incompatible score scales are added together.

## Results and recovery

The run directory contains `run-config.json`, `checkpoint.json`, `candidates.json`, `ranking.csv`,
`summary.json`, `initial_candidates.json`, and per-candidate inputs/evidence.
The accepted export uses the existing `Candidate` fields; its `iptm` alias and
objective explicitly refer to TorchFold. Use it as reviewed input when preparing
a later evolutionary workflow; this command does not automatically launch one.

Job IDs and request hashes are persisted before consuming results. A dropped
submission response is uncertain: the remote job may already exist. Automatic
POST retries are deliberately disabled. Inspect the API/Slurm jobs and attach the
verified existing ID using the exact key from `checkpoint.json`:

```bash
uv run --frozen ddeharness protein-design initial-round -c initial.yaml \
  --attach-job generate/ppiflow=VERIFIED_DESIGN_ID
```

Attachment is an operator assertion about the lost request's identity; compare
its inputs and parameters on the service. Never attach an unrelated job. If no
job was actually created, use a new run directory after confirming that fact.
A persisted failed job is not resubmitted by resume. Timeouts leave jobs available
for subsequent polling. Ctrl-C stops the client and its DockQ subprocess, but
**does not cancel external Slurm jobs**: most supplied APIs do not document a
cancellation route. Cancel those through the service/cluster, then inspect the
checkpoint before restarting. An advisory run lock prevents concurrent runners.

## Paired-chain antibodies

PPIFlow-only initial rounds also accept `mode: antibody`. Use the same request
fields as `protein_model_ppiflow`: `name`, `antigen_pdb`, `antigen_chain`,
`framework_dir`, `heavy_chain`, `light_chain`, and `cdr_length`. Optional hotspots
and sampling controls are forwarded unchanged. Set `chains.binder` to both
heavy and light chain IDs, `chains.target` to the antigen chains, and include
every chain in `refold_map`. Both binder sequences are placed in TorchFold
inputs; template MSA/features are cleared separately for each binder chain.
DockQ reports the minimum score across target/binder interfaces, excluding the
heavy/light interface. Rosetta groups both mapped binder chains together.
Exported binder sequences are joined with `|` in `chains.binder` order.
TorchCraft has no registered paired-antibody mode, so do not add a TorchCraft
generator to this configuration. Existing VHH/minibinder pairings are unchanged.
