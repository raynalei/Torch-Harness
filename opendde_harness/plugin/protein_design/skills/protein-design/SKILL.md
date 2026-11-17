---
name: protein-design
description: Run Torch-Harness model workflows and native scientific evidence functions without an OpenDDE compute service.
metadata:
  opendde:
    always: false
---

# Torch-Harness protein design

## Start and recover

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

## Models and finite routing

PPIFlow/TorchCraft generate compatible candidates. TorchScore supplies ipTM/pTM.
Initial ipTM<0.5 terminates that candidate. Both ipTM>0.8 and pTM>0.8 go directly
to TorchFold/DockQ. Other initial candidates use Rosetta residue energies to fix
positions with energy<-5 for compatible Partial Flow. Rescore Partial Flow output:
both>0.8 refold, otherwise TorchOptimization if actually supported. Rescore optimized
output: both>0.8 refold, otherwise terminate. Thresholds come from the task config.
There is no global rounds, samples, batch or patience limit. Qwen prioritizes legal
actions and cannot invent another route. Unknown metrics/logs pause for recovery.

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
and compresses the stage-relevant evidence for Qwen. Evidence and skill reasoning
do not override the Python router. Predicted outcomes need experimental validation.
