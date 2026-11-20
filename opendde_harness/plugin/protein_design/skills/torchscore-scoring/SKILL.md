---
name: torchscore-scoring
description: Score candidate PDBs with TorchScore or extract metrics from existing TorchScore runs.
---

# Score candidate PDBs with TorchScore or extract metrics from existing TorchScore runs

Use measured TorchScore iPTM and pTM to assess candidate confidence. The
two-stage planner uses these measurements as evidence, with rule cutoffs
reserved for rule mode and fallback.

- `pdb`: supply `input_pdb_dir` and run inference.
- `metrics`: supply `input_pdb_dir` and the existing TorchScore output directory
  in the compatibility field `af3score_output_dir`.

This API uses one sample, seed 10, init guess and no data pipeline. Confidence
does not measure binding affinity.

Use `protein_model_list` to inspect registered `torchscore` aliases and request fields.
Submit standalone requests with `protein_model_torchscore`.
Use the returned local `request_id` with `protein_model_status` and
`protein_model_results`. Reuse that ID when recovering an uncertain submission.
