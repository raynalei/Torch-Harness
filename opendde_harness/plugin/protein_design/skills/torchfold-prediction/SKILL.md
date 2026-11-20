---
name: torchfold-prediction
description: Predict or refold structures with TorchFold from a prepared fold-input JSON.
---

# Predict or refold structures with TorchFold from a prepared fold-input JSON

Submit a shared `json_path` with the sequences, chains and seeds for the intended
structure. Use `run_data_pipeline` and `num_diffusion_samples` as configured
for the task. When refolding a design, check the sequence for every mapped
chain and keep the actual design structure as the DockQ reference. TorchFold
confidence and design-versus-refold DockQ describe different outputs.

Use `protein_model_list` to inspect registered `torchfold` aliases and request fields.
Submit standalone requests with `protein_model_torchfold`.
Use the returned local `request_id` with `protein_model_status` and
`protein_model_results`. Reuse that ID when recovering an uncertain submission.
