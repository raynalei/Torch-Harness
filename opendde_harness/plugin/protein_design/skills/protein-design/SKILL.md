---
name: protein-design
description: Prepare, run and interpret antibody or minibinder workflows, including registered PPIFlow, TorchCraft, AF3Score, TorchScore, Torch Optimization, TorchFold and Rosetta model tools.
metadata:
  opendde:
    always: false
---

# Protein design

## Design scope

Use the Torch-Harness initial round and registered external model tools for antibody, VHH and minibinder design. Never relabel a mini binder as an antibody.

## Registered external model tools

### Initial-round preflight

Use `ddeharness protein-design initial-round -c <profile> --check` for preflight,
not ad hoc package imports or shell network probes. The runner executes the
bundled `initial_round/dockq.py --check` with the YAML's `evaluation.dockq_python`.
The Python module is **`DockQ.DockQ`**, case-sensitive; `import dockq` is not a
valid test. Do not use the active shell's python/python3 to judge this isolated
interpreter. If preflight fails, read the reported `dockq-check.log` and diagnose
the actual exception before proposing an installation. Do not grep the config
directory or repeatedly search for the package source to perform this check.

For any service with `sbatch_script`, preflight checks no local ports and submits
no Slurm job. It reports `reachable: null` and startup deferred. This is expected,
not an unhealthy-service failure. Do not run curl, socket probes, registry health
checks or a local-port scan to second-guess this result. After authorization,
launch the worker; each new run starts fresh API allocations for its stages, resolves
the Slurm node, and checks that node's API before its HTTP task. Never treat the
configured URL host as the allocated service address. No localhost probing or direct-HTTP fallback is allowed. Resolve real missing
script/input files, but do not claim sbatch-managed services must be running
before launch. 


For a complete authorized initial round, prefer `protein_initial_round_start`. Select PPIFlow or TorchCraft from the user prompt using `task.generators: ["ppiflow"]` or `["torchcraft"]`; neither is mandatory. If unspecified, choose a compatible configured generator from the design type and available inputs and state the choice. Do not switch a requested generator silently. Keep explicit dual-generator configurations supported. Prefer its compact `task` for new targets: pdb_path, design_type, hotspots and initial_samples. Collect the user’s target_chain and ordered binder_chains: one binder for minibinder, one heavy chain for VHH, or [heavy, light] for paired antibody. Ask for any missing chain IDs. A target-only PDB needs only the target chain; output chains are supplied by the user. PPIFlow binder does not accept framework_dir, fixed_positions, or num_timesteps. For PPIFlow minibinder tasks, pass samples_min_length and samples_max_length inside task.generator_overrides.ppiflow.request if the profile does not already supply the requested lengths. Use the compact task for this. Refold preserves the task IDs unless the user explicitly supplies refold_map. Persistent workflow_defaults/design_profiles and registered request_defaults supply fixed deployment settings. Frameworks and fixed/designable residue requirements belong to the current task and must not be persisted as defaults. Inspect protein_model_list for configured profile names. Missing deployment settings fall back only to the bundled cluster_deployment.yaml or configured deployment_config; do not repeatedly ask users for fixed source/weight/Conda paths. Never reuse an old target CSV, hotspots or fold template for a new protein. Accept natural-language task instructions: assemble its structured `run_config` from existing registered service metadata, tool arguments and Python-derived results. Do not require the user to write YAML. Use service-side defaults for optional model parameters; do not copy weights/config/seed/timestep/device values from old tasks. Never load historical initial_round_config automatically. An explicit user-requested config_path or run_config remains supported. Never invent missing framework paths, deployment addresses or scientific inputs. The detached Python worker handles generation, GET polling, scorer filtering, refold-input construction, TorchFold and design/reference DockQ. Optional autonomous feedback uses bounded decisions from the configured chat provider; GET polling itself never requires repeated chat calls. Report its run directory concisely; query saved progress with `protein_initial_round_status` when asked. Do not manually poll model APIs from repeated chat turns. PPIFlow-only profiles support paired VH/VL antibodies with `mode: antibody`, user-specified heavy and light chains, and an identity `refold_map` unless explicitly overridden; TorchCraft cannot be paired with this mode. Reuse known framework, chain, hotspot and shared-path parameters from the conversation or configuration; derive verifiable structure facts with Python. When using inline run_config, the worker saves its own JSON snapshot for recovery. Do not fabricate these inputs from a PDB alone.

Call `protein_model_list` to discover configured aliases, the active config path and request schemas. An empty models list means tool code is installed but no external API endpoints are configured in that instance; it does not mean the tools are missing. The named tools are `protein_model_ppiflow`, `protein_model_torchcraft`, `protein_model_torchfold`, `protein_model_af3score`, `protein_model_torchscore`, `protein_model_torchopt` and `protein_model_rosetta`; each exposes its own request schema. Use the documented `protein-design models register` command to register operator-supplied endpoints in the correct config, then restart Harness. Never invent a working endpoint. Registration is independent of chat-provider configuration . Use only the operations the registry advertises. A configured endpoint or health response is not evidence of successful model inference.

TorchCraft `monomer_unconditional` requires an existing remote `json_path` and a `design_length`; it does not require a target, hotspots or framework. TorchCraft `mini_binder` and `vhh` require PDB and hotspots; `target_full_seq` is optional and omitting it uses native PDB-sequence fallback. If provided, it is a remote CSV path with target/real_sequence columns, not sequence text; target matches the PDB filename stem and hotspot_indices uses PDB residue IDs. Do not apply target-binding  to unconditional monomer generation.

Resolve scientific inputs and the compute scope with the user. Reuse existing authorization for the same inputs and scope rather than repeatedly asking. Call `protein_model_submit` with a unique, stable `request_id` and `user_confirmed: true` only when authorized. Submission returns immediately. Preserve the request ID and query `protein_model_status` when needed; never poll in a tight loop. Read completed artifacts with `protein_model_results`. For ordinary successful model requests, status automatically caches artifacts/metrics before requesting cancellation of that request's private API allocation; later status/results reads use the retained data. Full workflows release owned APIs when their remote jobs are terminal. Externally managed APIs, unresolved submissions and unresolved TorchOptimization child jobs are not cancelled automatically. Cleanup reports cancellation requested, not confirmed scheduler termination. A lost submission response is uncertain: do not submit again under a new request ID; ask the service operator to reconcile it. `protein_model_cancel` currently supports AF3Score only, and reports cancellation requested rather than claiming the remote job stopped.

AF3Score `pdb` mode can perform fast scoring or database searches plus inference. Pipeline-only tasks (`run_data_pipeline=true`, `run_inference=false`) produce prepared inputs and no metrics. `prepared` mode does not rewrite supplied MSA/templates or automatically extract metrics; use a separate `metrics` task with the original PDB directory and AF3Score output directory when required. `init_guess` is independent of structural templates. Never fabricate scores or compare fast/MSA scoring as if their protocols were identical. Use absolute paths on the service's filesystem; these tools do not upload files or activate Conda interactively. Every scientific API must have an existing `service.sbatch_script`; direct HTTP is disabled. Slurm must allocate and resolve a non-loopback node before any HTTP request; environment activation belongs in that script.

For external-only requests, proceed through these tools. Predicted structures, confidence and Rosetta energies are not experimentally measured affinity.

## Scientific input preparation

Resolve target identity, species, sequence, domain boundaries, hotspot numbering and chain mapping using authoritative sources. Use PDB, UniProt and PubMed lookup skills where appropriate. Preserve supplied frameworks and fixed positions. Validate a supplied initial-round profile with `ddeharness protein-design initial-round -c <profile> --check`. Review the exact scientific inputs and resource scope before launch. Use the initial-round status and resume tools for existing runs. Never invent paths, scores or scientific inputs.

TorchScore is an independent `torchscore` model kind, not an AF3Score alias. For
the antibody and nanobody routes, use `evaluation.scoring_backend: torchscore`
and the registered `torchscore` service. Keep the scorer fixed when resuming a run. It supports PDB scoring and metrics extraction
with seed=10/one sample, not AF3Score's full preprocessing protocol.
TorchOptimization (`torchopt`) uses only `/v1/optimizations`: submit registered
`pdb_paths`, `binder_type` (`vhh` or `mini-binder`), chains and method. Reuse the
local request_id; the client persists an idempotency key. The API returns an ID
before launching and verifies Slurm receipts itself. PREPARING/SUBMITTED/RUNNING/
UNKNOWN/COLLECTING are not success. Only SUCCEEDED with completed scheduler jobs,
zero exit codes and native final artifacts establishes optimization completion;
PREPARED is preparation only. Read native CSV/PDB paths with the results tool,
then independently score final structures. Never claim affinity from optimizer
completion. Paired VH/VL is unsupported. The bundled deployment uses port 8060;
reuse the registered deployment and its model environment rather than inventing
new paths or asking for extra workflow annotations. Autonomous feedback can use
this registered service for low-confidence candidates without per-round approval.


PPIFlow `binder` and TorchCraft `mini_binder` finish after the configured TorchScore
or AF3Score confidence scorer, Rosetta interface energy, TorchFold and DockQ.
PPIFlow `antibody` uses the antibody route. PPIFlow `nanobody` and TorchCraft
`vhh` use the nanobody route. Both routes use TorchScore for every generated or
refined candidate and default to the existing follow-up policy:
initial ipTM < 0.5 terminates that candidate; both ipTM and pTM > 0.8 go directly to
TorchFold/DockQ. Other initial candidates use the uploaded interface-energy
algorithm on Rosetta ResResE logs: summed negative target–binder contact energies
below -5 select fixed binder residues for Partial Flow, with all other binder
residues mutable. Partial Flow output is rescored: both > 0.8 go directly to
TorchFold/DockQ, otherwise use registered TorchOptimization. Rescore optimized
output and refold only if both > 0.8; otherwise terminate. Do not substitute scalar
dG_separated for per-residue energies or route by hotspot contacts. Missing
scientific outputs require recovery, never invented values or automatic passes.
See docs/autonomous-feedback.md for the service output contract and limits.
The supplied TorchScore sbatch uses 8070; TorchOptimization remains 8060.


Feedback classifications are Python outputs: `hotspot_hit`, `classify`,
`repair_allowed`, `check_region_encoding`, `torchopt_supported` and `stop_reason`. Use measured values,
not prose guesses, to select legal options. The transport rejects loopback hosts,
absolute URL overrides, redirects and saved endpoints without allocation proof.
Registry health discovery is deferred and never probes a local port. Missing
sbatch deployment metadata is a configuration issue, not permission to run an API
locally or bypass Slurm. The language-provider connection is separate from these
scientific compute APIs.
