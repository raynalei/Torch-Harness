---
name: antibody-inverse-folding
description: Capability-gated reference for backbone-conditioned antibody sequence design.
---

# antibody-inverse-folding

Use registered Torch-Harness model APIs and native evidence tools. Do not call
an OpenDDE compute service, install its Docker image or request a compute URL.
Call `protein_design_context` to inspect actual capabilities. Missing evidence is
unknown; never replace a measurement with an LLM-generated score.

True inverse folding requires a corresponding registered backend. The current
PPIFlow/TorchCraft/TorchOptimization contracts do not expose a fixed-backbone
SolubleMPNN operation. This skill is therefore unavailable as an executable action
in the current native capability menu. Do not import an OpenDDE compute client,
pretend PPIFlow is inverse folding, or fabricate sequences as model output.
A separately authorized compatible Partial Flow request is a different operation.
