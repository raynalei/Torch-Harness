---
name: ppiflow-design
description: Generate antibodies, nanobodies or minibinders, or refine an existing antibody or nanobody complex with PPIFlow Partial Flow.
---

# Generate antibodies, nanobodies or minibinders, or refine an existing antibody or nanobody complex with PPIFlow Partial Flow

PPIFlow Partial Flow refines an existing antibody or nanobody complex while
preserving explicit fixed residues. Use compatible chain roles and verified
PDB numbering for the mutable region. Use full generation for a new design.

| Mode | Required scientific inputs |
| --- | --- |
| `binder` | `input_pdb`, `target_chain`, `binder_chain`, `samples_min_length`, `samples_max_length` |
| `nanobody` | `antigen_pdb`, `antigen_chain`, `framework_dir`, `heavy_chain` |
| `antibody` | The nanobody fields plus `light_chain` and `cdr_length` |
| `partial_nanobody` | `complex_pdb`, `antigen_chain`, `heavy_chain`, `fixed_positions`, `cdr_position`, `start_t` |
| `partial_antibody` | The partial_nanobody fields plus `light_chain` |

Specify chain IDs for the task. In binder mode a target-only input is supported,
with the binder chain ID naming the output chain. Keep binder requests free of
`framework_dir`, `fixed_positions` and `num_timesteps`. Check the result
manifest for the complete target–binder complex before scoring.

Use `protein_model_list` to inspect registered `ppiflow` aliases and request fields.
Submit standalone requests with `protein_model_ppiflow`.
Use the returned local `request_id` with `protein_model_status` and
`protein_model_results`. Reuse that ID when recovering an uncertain submission.
