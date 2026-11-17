---
name: minibinder-point-mutation
description: Validate substitutions in an existing mini-binder sequence.
---

# minibinder-point-mutation

Use registered Torch-Harness model APIs and native evidence tools. Do not call
an OpenDDE compute service, install its Docker image or request a compute URL.
Call `protein_design_context` to inspect actual capabilities. Missing evidence is
unknown; never replace a measurement with an LLM-generated score.

There are no antibody CDR or framework assumptions for mini-binders. Keep the
provided binder length, fixed residues and target. Use native action `mutations`
to validate explicit one-based substitutions. Use compatible registered
TorchOptimization for actual optimization, with new TorchScore evaluation of
its outputs. Never reinterpret a sequence plausibility hypothesis as binding.
