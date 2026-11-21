---
name: rosetta-analysis
description: Evaluate protein interfaces with checkpointed direct Slurm PyRosetta; never use the retired Rosetta HTTP API.
---

# PyRosetta interface analysis (direct Slurm)

The autonomous workflow uses the bundled `rosetta_slurm.py` and `rosetta_worker.py`.
They launch a Slurm job via `sbatch`, importing PyRosetta under the configured
absolute interpreter. No Rosetta API server, HTTP URL, port, or model allocation
is required.

The supported operation is `interface_analyze`: generate measured
`dG_separated` and a residue-pair energy log from an actual PDB complex.
Supply distinct target/binder chain groups and the input PDB; retain PDB
numbering and validate output hashes and Slurm ExitCode=0:0.

Use `protein_design_start`, `protein_design_status`,
`protein_design_resume` and `protein_design_candidates`.
The standalone `protein_model_rosetta` submit tool and all six former HTTP
operations have been retired. The current direct worker does not implement
`structure_relax`, `constrained_relax`, `interface_local_relax`,
`structure_score`, or `mutation_binding_ddg`; do not claim those are
PyRosetta-capable until native workers are added and tested.
