---
name: pdb-lookup
description: Retrieve public PDB metadata by verified accession.
---

# pdb-lookup

Use registered Torch-Harness model APIs and native evidence tools. Do not call
an OpenDDE compute service, install its Docker image or request a compute URL.
Call `protein_design_context` to inspect actual capabilities. Missing evidence is
unknown; never replace a measurement with an LLM-generated score.

Call native action `lookup`, source `pdb`, identifier as a four-character PDB
accession. The backend reads the fixed RCSB metadata endpoint. Verify chain,
organism and experimental provenance; do not claim a coordinate file was downloaded
or a biological target identified from a metadata-only result. Read actual files
before configuring model inputs.
