---
name: post-filter
description: Interpret final measured confidence and developability evidence.
---

# post-filter

Use registered Torch-Harness model APIs and native evidence tools. Do not call
an OpenDDE compute service, install its Docker image or request a compute URL.
Call `protein_design_context` to inspect actual capabilities. Missing evidence is
unknown; never replace a measurement with an LLM-generated score.

Use native action `rank` with Candidate JSON for a reproducible listing of
complete TorchScore ipTM/pTM and design-refold DockQ tuples. It does not change
router acceptance; candidates with missing metrics are explicitly unranked.
Explain trade-offs using actual PTM motif flags, method-labelled PSH/PNC/PSC,
hotspot contacts and available EpiMatrix results. Missing immunogenicity is unknown,
not low risk. Do not fabricate a weighted affinity score or silently reorder
saved acceptance decisions. New scientific ranking policies require explicit task
constraints, not an invented replacement for the current router.
