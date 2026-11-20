---
name: torchcraft-design
description: Generate TorchCraft mini-binders, VHH designs or unconditional monomers.
---

# Generate TorchCraft mini-binders, VHH designs or unconditional monomers

Choose a TorchCraft mode based on the requested design and available inputs.

| Mode | Required scientific inputs |
| --- | --- |
| `mini_binder` | `pdb_path`, `hotspot_indices`, `design_length` |
| `vhh` | `pdb_path`, `hotspot_indices`, `framework_cif_path` |
| `monomer_unconditional` | `json_path`, `design_length` |

Supply `name` for every request. Use the deployed hotspot-index convention.
For complex designs, `target_full_seq` can point to the target sequence CSV
when needed. Preserve user-specified chain roles in the workflow. The monomer
mode is a standalone request and takes no target or framework fields.

Use `protein_model_list` to inspect registered `torchcraft` aliases and request fields.
Submit standalone requests with `protein_model_torchcraft`.
Use the returned local `request_id` with `protein_model_status` and
`protein_model_results`. Reuse that ID when recovering an uncertain submission.
