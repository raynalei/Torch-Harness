---
name: pdb-lookup
description: Retrieve public PDB metadata by verified accession.
---

# pdb-lookup

Call `protein_design_analyze` through the configured login-node public lookup
API on port 8100. Check `protein_design_context.public_lookup_configured`
first. If the API is unavailable, report the error. Keep database requests
on the login node.

Use action `lookup`, source `pdb`, identifier as a four-character PDB
accession. The backend reads the fixed RCSB metadata endpoint. Verify chain,
organism and experimental provenance; do not claim a coordinate file was downloaded
or a biological target identified from a metadata-only result. Read actual files
before configuring model inputs.
