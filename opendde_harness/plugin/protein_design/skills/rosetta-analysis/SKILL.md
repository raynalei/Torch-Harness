---
name: rosetta-analysis
description: Measure PyRosetta interface energies and use registered Rosetta scoring, relaxation and ddG operations when available.
---

# Measure PyRosetta interface energies and use registered Rosetta scoring, relaxation and ddG operations when available

Direct Slurm PyRosetta supports interface analysis through the workflow.
Use the generated complex and its explicit chain groups for interface energy.
Keep `dG_separated` separate from ResResE residue energy evidence.

If a Rosetta HTTP service is registered, inspect its advertised operations
before using the standalone tool. The client supports `interface_analyze`,
`structure_score`, `structure_relax`, `constrained_relax`,
`interface_local_relax` and `mutation_binding_ddg`. Interface analysis,
interface-local relaxation and mutation ddG require both chain groups.
Mutation ddG also requires the actual mutation.

Use `protein_model_list` to inspect registered `rosetta` aliases and request fields.
Use `protein_design_start` and `protein_design_status` for direct Slurm workflow
evaluation. For a registered HTTP service, submit with `protein_model_rosetta`
and use its local `request_id` with `protein_model_status` and
`protein_model_results`. Reuse the ID when recovering an uncertain submission.
