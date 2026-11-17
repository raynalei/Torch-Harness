---
name: minibinder-full-redesign
description: Generate mini-binders with compatible registered PPIFlow/TorchCraft modes.
---

# minibinder-full-redesign

Use registered Torch-Harness model APIs and native evidence tools. Do not call
an OpenDDE compute service, install its Docker image or request a compute URL.
Call `protein_design_context` to inspect actual capabilities. Missing evidence is
unknown; never replace a measurement with an LLM-generated score.

Use registered PPIFlow binder or TorchCraft mini_binder with real target,
hotspot and length inputs. TorchCraft target_full_seq is the target CSV path.
Do not use antibody frameworks/CDR assumptions. These integrations do not expose
mini-binder Partial Flow. Unsupported actions remain unavailable. Score actual
structures using TorchScore; generation completion is not successful binding.
