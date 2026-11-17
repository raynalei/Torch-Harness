---
name: uniprot-lookup
description: Retrieve public UniProt records by accession.
---

# uniprot-lookup

Use registered Torch-Harness model APIs and native evidence tools. Do not call
an OpenDDE compute service, install its Docker image or request a compute URL.
Call `protein_design_context` to inspect actual capabilities. Missing evidence is
unknown; never replace a measurement with an LLM-generated score.

Call native action `lookup`, source `uniprot`, identifier as the real UniProt
accession (including isoform suffix if relevant). Verify organism, isoform,
sequence length and domains. Never assume UniProt positions equal cropped target
PDB numbering. The backend is read-only and does not require a compute service.
