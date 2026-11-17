---
name: structure-analysis
description: Analyze materialized model structures without a compute-service dependency.
---

# structure-analysis

Use registered Torch-Harness model APIs and native evidence tools. Do not call
an OpenDDE compute service, install its Docker image or request a compute URL.
Call `protein_design_context` to inspect actual capabilities. Missing evidence is
unknown; never replace a measurement with an LLM-generated score.

Call `protein_design_analyze` with action `structure`, structure_path and
explicit binder_chains/target_chains. The native backend reports chain sequences
and minimum-heavy-atom contact evidence. Read ipTM/pTM/pLDDT/PAE from the actual
TorchScore/TorchFold artifacts; PDB B factors are not assumed confidence. The
contact backend does not claim PLIP hydrogen-bond or energetic classifications.
Use the registered Rosetta tool for energy measurements, not inferred contacts.
