---
name: cdr-point-mutation
description: Propose and validate fixed-position-aware antibody substitutions.
---

# cdr-point-mutation

Use registered Torch-Harness model APIs and native evidence tools. Do not call
an OpenDDE compute service, install its Docker image or request a compute URL.
Call `protein_design_context` to inspect actual capabilities. Missing evidence is
unknown; never replace a measurement with an LLM-generated score.

Use the measured parent interface, chemical PTM flags and surface patch evidence
to propose explicit one-based chain sequence substitutions. Preserve framework,
selected fixed points and target sequence. Validate proposed changes using
`protein_design_analyze` with action `mutations`, sequences, mutations (chain,
position, amino_acid) and fixed_positions. This backend applies substitutions,
not an affinity prediction. Materialize/rescore through authorized registered
TorchFold/TorchScore requests before accepting any result. The autonomous router
uses TorchOptimization only where its actual mode/chain capability permits it.
