# Workflow configuration

The two configuration files serve different purposes and are loaded together.

| File | Purpose |
| --- | --- |
| Active `config.json` (normally `~/.opendde_harness/config.json`) | LLM/provider settings and `plugins.config.protein-design.model_services`: registered API kinds, ports, sbatch scripts and service protocol settings. |
| Task YAML, or the natural-language tool's `run_config` object | This experiment's PDB, hotspots, frameworks, generator requests, chain mappings, selected scorer and autonomous budget. |
| Run directory `worker-config.json` / `run-config.json` | Resolved configuration saved for launch/resume and execution. Existing runs retain their recorded settings. |

Missing service blocks in a task are filled from a unique registration with the corresponding `kind`. A complete service block explicitly supplied in the task takes precedence; blocks are not merged field by field. Multiple registrations of one kind require an explicit task service. The selected scorer and TorchFold are resolved for every run; autonomous runs also resolve Rosetta and optional TorchOptimization (`kind: torchopt`). Generator service settings are resolved only for generators declared in the task. Registration does not supply scientific inputs or enable an undeclared generator.

This resolution applies to the CLI, YAML-based natural-language launches and structured `run_config` launches. The worker uses the resolved snapshot; editing global settings does not change an already launched run.

## Examples

- [`initial_round_sbatch.yaml`](examples/initial_round_sbatch.yaml) is the original paired VH/VL **single-round** task with its own explicit service blocks. It uses PPIFlow, AF3Score, TorchFold and Rosetta. It has no autonomous section. TorchCraft and TorchOptimization currently support VHH/minibinder modes, not this paired VH/VL mode.
- [`autonomous_round_registered.yaml`](examples/autonomous_round_registered.yaml) is a VHH autonomous task using registered services. It declares both PPIFlow and TorchCraft, selects TorchScore, and allows TorchOptimization through its registration when the decision system finds a legal optimization action. AF3Score can instead be selected with `scoring_backend: af3score`; the two scorers are alternatives for a run.
- [`model_services.json`](examples/model_services.json) is a **configuration fragment**, not a replacement for your complete global config. Merge its `model_services` registrations into the active config and replace the example shared paths with your deployment paths. All seven services include sbatch script placeholders. Ports must match the scripts; these example defaults need not match the older task's explicitly configured ports. TorchOptimization uses port 8060.

After configuring the real shared paths and sbatch scripts:

```bash
ddeharness protein-design initial-round -c docs/examples/autonomous_round_registered.yaml
```

For natural-language launches the agent constructs the equivalent `run_config`; a handwritten YAML is optional. Required scientific inputs must already be provided or available from existing inputs. Every scientific API allocation still starts through sbatch, resolves its assigned compute host, and is health-checked there before inference requests.

## Existing deployment configuration

If your global config already registers PPIFlow (8010), TorchCraft (8020), AF3Score (8030), TorchFold (8040), Rosetta (8050) and TorchScore, keep those existing deployment settings. The generic registration example is not an instruction to change their ports. Add the missing `torchopt` registration using the deployed optimization sbatch script and port 8060. If TorchScore currently listens on 8060, choose a separate port such as 8070 and update both its registration and sbatch launch command. Replace any `/REPLACE_WITH_ACTUAL_PATH/...` value in `initial_round_config` with the real task YAML path; this field is the default profile for the natural-language tool. CLI `-c` and tool `config_path` explicitly select a profile; inline `run_config` supplies the task directly.

A configured `127.0.0.1` service URL is treated only as a port template by the sbatch transport, which replaces the host with the allocated compute node before HTTP requests. Prefer the explicit `slurm-api.invalid` placeholder in configuration to make that intent visible. Neither placeholder is an inference destination.

## TorchOptimization tool invocation and diagnostics

`protein_model_torchopt` is a submit-only named tool. It has no `action: import` parameter. A valid invocation uses the registered alias (optional if unique), a stable local request ID, and the native optimization request:

```json
{
  "model": "torchopt",
  "request_id": "vhh-opt-001",
  "request": {
    "pdb_paths": ["/lustre/project/inputs/complex.pdb"],
    "binder_type": "vhh",
    "method": "mcmc-ddg"
  },
  "user_confirmed": true
}
```

Use `user_confirmed: true` only within an existing user-authorized compute scope. The transport starts the registered API sbatch allocation and posts to `/v1/optimizations`, expecting `optimization_id` in the JSON response. It does not use a generic import endpoint. Query the local `request_id` using `protein_model_status` or `protein_model_results`.

A missing remote ID does not by itself prove an endpoint mapping error. Failed submissions now preserve `submission_diagnostic`: POST endpoint, failure stage, exception type, expected ID field and HTTP status when available, without response bodies or credentials. Status and same-ID retries expose that diagnostic. A healthy `/health` endpoint proves reachability, not successful job submission. If an ID is absent and by-key recovery cannot verify submission, do not repeat a direct POST or invent a new request ID: the remote job may already exist. Existing uncertain records created before this diagnostic was added cannot retrospectively recover the original error.
