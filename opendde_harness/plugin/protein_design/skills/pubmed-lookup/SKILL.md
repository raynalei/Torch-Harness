---
name: pubmed-lookup
description: Retrieve cited PubMed records by PMID.
---

# pubmed-lookup

Use registered Torch-Harness model APIs and native evidence tools. Do not call
an OpenDDE compute service, install its Docker image or request a compute URL.
Call `protein_design_context` to inspect actual capabilities. Missing evidence is
unknown; never replace a measurement with an LLM-generated score.

Call native action `lookup`, source `pubmed`, identifier as a numeric PMID.
Report the source and distinguish publication metadata/abstract claims from
experimental evidence for the actual candidate. The backend retrieves metadata,
not guaranteed full text. Do not invent titles, citations or findings.
