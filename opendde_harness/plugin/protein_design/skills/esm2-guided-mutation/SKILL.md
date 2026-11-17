---
name: esm2-guided-mutation
description: Capability-gated ESM2 likelihood guidance; not an affinity predictor.
---

# esm2-guided-mutation

Use registered Torch-Harness model APIs and native evidence tools. Do not call
an OpenDDE compute service, install its Docker image or request a compute URL.
Call `protein_design_context` to inspect actual capabilities. Missing evidence is
unknown; never replace a measurement with an LLM-generated score.

The current registered models do not include an ESM2 scoring/DMS backend.
Do not invoke the retired compute endpoint or invent LLR/PLL values. Native
capability discovery reports ESM2 unavailable until a real backend and weights
are configured. A validated native substitution is not an ESM2 inference result.
Use existing measured evidence for supported TorchOptimization requests instead,
without relabeling them as ESM2-guided sampling.
