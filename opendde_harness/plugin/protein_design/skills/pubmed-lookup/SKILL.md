---
name: pubmed-lookup
description: Retrieve cited PubMed records by PMID.
---

# pubmed-lookup

Call `protein_design_analyze` through the configured login-node public lookup
API on port 8100. Check `protein_design_context.public_lookup_configured`
first. If the API is unavailable, report the error. Keep database requests
on the login node.

Use action `lookup`, source `pubmed`, identifier as a numeric PMID.
Report the source and distinguish publication metadata/abstract claims from
experimental evidence for the actual candidate. The backend retrieves metadata,
not guaranteed full text. Do not invent titles, citations or findings.
