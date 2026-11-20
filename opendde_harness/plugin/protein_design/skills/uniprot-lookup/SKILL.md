---
name: uniprot-lookup
description: Retrieve public UniProt records by accession.
---

# uniprot-lookup

Call `protein_design_analyze` through the configured login-node public lookup
API on port 8100. Check `protein_design_context.public_lookup_configured`
first. If the API is unavailable, report the error. Keep database requests
on the login node.

Use action `lookup`, source `uniprot`, identifier as the real UniProt
accession (including isoform suffix if relevant). Verify organism, isoform,
sequence length and domains. Never assume UniProt positions equal cropped target
PDB numbering. The backend is read-only and does not require a compute service.
