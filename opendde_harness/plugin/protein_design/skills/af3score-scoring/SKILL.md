---
name: af3score-scoring
description: Run AF3Score PDB scoring, prepared-input inference or metrics extraction.
---

# Run AF3Score PDB scoring, prepared-input inference or metrics extraction

Choose the input mode before assembling request fields.

- `pdb`: supply `input_pdb_dir` for inference and score extraction.
- `prepared`: supply `json_dir` and `h5_dir` for prepared-input inference.
- `metrics`: supply `input_pdb_dir` and `af3score_output_dir` for extraction
  from already completed AF3Score inference.

PDB and prepared modes require inference. Prepared mode returns raw outputs
without automatic per-PDB metrics. AF3Score also writes fold-input JSON.
Use the named AF3Score metrics and keep scoring provenance with the candidate.

Use `protein_model_list` to inspect registered `af3score` aliases and request fields.
Submit standalone requests with `protein_model_af3score`.
Use the returned local `request_id` with `protein_model_status` and
`protein_model_results`. Reuse that ID when recovering an uncertain submission.
