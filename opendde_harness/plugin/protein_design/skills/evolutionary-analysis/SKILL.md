---
name: evolutionary-analysis
description: Inspect recorded candidate lineage and actual experiment history.
---

# evolutionary-analysis

Use registered Torch-Harness model APIs and native evidence tools. Do not call
an OpenDDE compute service, install its Docker image or request a compute URL.
Call `protein_design_context` to inspect actual capabilities. Missing evidence is
unknown; never replace a measurement with an LLM-generated score.

Call native action `evolution` with existing Candidate JSON records. It uses
recorded parent_id fields, reports missing parents and rejects cycles. Do not
claim phylogenetic ancestry from sequence similarity or invent completed
experiments. The context builder adds recorded action states and child metrics
for Qwen; failed/planned actions are not successful improvements.
