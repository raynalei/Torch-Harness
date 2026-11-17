---
name: epitope-analysis
description: Measure target contacts and hotspot coverage in true PDB numbering.
---

# epitope-analysis

Use registered Torch-Harness model APIs and native evidence tools. Do not call
an OpenDDE compute service, install its Docker image or request a compute URL.
Call `protein_design_context` to inspect actual capabilities. Missing evidence is
unknown; never replace a measurement with an LLM-generated score.

Call native action `epitope` with structure_path, explicit binder/target chains
and hotspots as objects {chain, residue_id, optional insertion_code}. These are
actual PDB residue IDs, not zero-based or assumed sequence offsets. The backend
validates mappings and reports minimum heavy-atom distance (<5 Angstrom default).
Missing hotspots are unknown, not a miss. Generated and refolded structures may
have different contacts; record the structure basis. Hotspot evidence diagnoses
off-epitope poses but does not override the autonomous confidence router.
