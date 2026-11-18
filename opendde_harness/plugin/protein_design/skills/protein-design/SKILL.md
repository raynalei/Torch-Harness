---
name: protein-design
description: Run Torch-Harness model workflows and native scientific evidence functions without an OpenDDE compute service.
metadata:
  opendde:
    always: false
---

# Torch-Harness protein design

## Start and recover

Prefer `protein_design_start(task=...)` with new pdb_path, target_chain, design_type,
explicit hotspots and initial_samples. Persistent workflow_defaults/design_profiles
and registered request_defaults supply fixed services, sbatch environment, source/
weight paths, shared mappings and output/DockQ settings. No per-task YAML is needed.
Choose task.generators from the prompt: ["ppiflow"] or ["torchcraft"]. Neither is
mandatory; paired VH/VL requires PPIFlow. Do not silently switch a requested model.
Framework and fixed/designable residue requirements belong to the current task;
never reuse old target CSVs, hotspots or prepared fold templates. Inspect configured
profiles with protein_model_list and report only missing one-time setup.

Call `protein_design_context` for configured model kinds and native capabilities.
Use `protein_model_list` for actual model aliases and request schemas. Assemble
run_config from known scientific inputs and registrations or reuse workflow_config.
Do not require the user to write YAML or choose an initial-round mode.
Call `protein_design_start` when inputs/model execution are authorized. Reuse the
same run_dir with `protein_design_status`, `protein_design_resume`,
`protein_design_list` and `protein_design_candidates` after interruption.
If status reports pending_regions, show the exact question and candidate PDB; use
`protein_design_regions` with the user-supplied CDR ranges and question_id to resume.
CLI equivalents are `ddeharness protein-design start/status/resume/list/candidates`.
The OpenDDE compute service, compute_url/worker pools, Docker setup and old OpenDDE
model checkpoints are not part of this workflow. Do not start them or fall back to
their endpoints. Model environment activation belongs in each registered sbatch.

## Qwen planner and fallback

New generic starts default to decision_policy=planner, planner_layout=two_stage: PPIFlow/TorchCraft generate initial candidates,
then TorchScore measures ipTM/pTM. First evaluate only TorchScore and Rosetta, then let Qwen stop, advance or select
one compatible optimization. Finish the batch first-stage decisions before stage two.
Second evaluate the current candidate using TorchScore, Rosetta and full native
evaluation skills; then let Qwen stop, optimize or request final TorchFold/DockQ.
The context builder loads stage-specific skills and metrics, history and constraints. Rule recommendations are advisory, not menu gates.
Qwen chooses stop (discard poor candidate), continue (TorchFold/DockQ then final
completion), analysis (one selected evaluation skill), partial_flow or torchopt.
Stage-one continue/analysis advances to full stage-two evaluation. Results are saved
and passed into the second scientific decision. A second-stage optimization child
is freshly TorchScored and finally validated without a third Qwen decision. Optimization children
are scored afresh before replanning; never carry parent scores as child measurements.

Before either optimizer call use protein_design_mutation_selector with explicit
one-based observed binder fixed_positions or mask_positions. Python validates
inherited fixed points, allowed regions, chain IDs, numbering and nonempty masks.
Partial Flow receives real encoded mask/fixed ranges; TorchOptimization lacks a
native mask argument, so the workflow validates returned structures against the
mask and rejects violations. Do not claim native masked optimization for that API.

Planner timeout, malformed or invalid choices use recorded staged rule fallback.
Set decision_policy=rules to run rules explicitly. Legacy agent profiles preserve
their earlier rule-constrained priority mode. Model execution failures pause the
planned action for recovery; they do not trigger another scientific submission.
There is no global round, sample, batch or patience cap; each surviving lineage has
two scientific decision stages and either optimizer may be selected when compatible.
Final acceptance requires real TorchFold/DockQ validation. Existing iterative planner
snapshots remain compatible; new generic starts use the two-stage layout.

Use dedicated `protein_model_*` tools for authorized standalone requests, preserving
request_id and uncertain remote IDs. A model registration is not proof of inference.
PPIFlow supports configured antibody/nanobody/binder modes; TorchCraft and the current
optimizer do not support paired VH/VL. Mini-binder Partial Flow is unavailable.
Inverse-folding and ESM2 skills remain capability-gated references, not executable
operations in the current registered model set. Do not mislabel another model as them.

## Native evidence tools

Use `protein_design_analyze` with one of these actions:

- `developability`: chemical PTM motifs, method-labelled PSH/PNC/PSC and EpiMatrix
  availability. Supply sequences and optional structure_path/binder_chains.
- `structure` / `epitope`: native heavy-atom contacts, explicit chain roles and
  true PDB hotspot residue IDs. Read confidence from real scoring artifacts.
- `evolution`: recorded parent/child lineage; no invented ancestry.
- `mutations`: apply validated one-based substitutions while preserving fixed points.
- `rank`: list complete measured confidence tuples without changing acceptance.
- `lookup`: read public PDB/UniProt/PubMed metadata by validated identifier.

These are backed by scientific_agent/backends, not a remote OpenDDE compute API.
Missing evidence stays unavailable, never fabricated by the LLM. EpiMatrix is a
reserved licensed adapter; it is not currently configured. Surface descriptors are
not official TAP calibration; preserve the algorithm and parameter labels.

## Decision references

Read [binder_failure_diagnosis.md](references/binder_failure_diagnosis.md),
[generation_strategy.md](references/generation_strategy.md) and
[continue_or_stop.md](references/continue_or_stop.md). The scientific context builder
extracts measured Candidate JSON, enriches with actual actions/history/constraints,
and compresses the stage-relevant evidence for Qwen. Python enforces actual capabilities and task constraints; the planner owns scientific
action choice. Rules remain advisory and available for fallback. Predicted outcomes need experimental validation.
