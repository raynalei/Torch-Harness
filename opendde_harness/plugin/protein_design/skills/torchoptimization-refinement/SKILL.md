---
name: torchoptimization-refinement
description: Refine an existing single-chain VHH or mini-binder complex with TorchOptimization.
---

# Refine an existing single-chain VHH or mini-binder complex with TorchOptimization

TorchOptimization suits optimization of an existing VHH or mini-binder complex
with one target and one binder chain. The API has no native residue mask.
When the planner supplies a mask, validate the returned proposal against it.

Provide `pdb_paths`, `binder_type` (`vhh` or `mini-binder`) and distinct
`target_chain` and `binder_chain` values. Choose `mcmc`, `ddg` or
`mcmc-ddg` based on the requested search. `prepare_only` prepares a job
without establishing optimization completion. Rescore actual returned structures.

Use `protein_model_list` to inspect registered `torchopt` aliases and request fields.
Submit standalone requests with `protein_model_torchopt`.
Use the returned local `request_id` with `protein_model_status` and
`protein_model_results`. Reuse that ID when recovering an uncertain submission.
