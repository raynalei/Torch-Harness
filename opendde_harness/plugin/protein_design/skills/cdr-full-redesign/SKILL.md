---
name: cdr-full-redesign
description: Select compatible antibody generation or energy-fixed Partial Flow.
---

# cdr-full-redesign

Use registered Torch-Harness model APIs and native evidence tools. Do not call
an OpenDDE compute service, install its Docker image or request a compute URL.
Call `protein_design_context` to inspect actual capabilities. Missing evidence is
unknown; never replace a measurement with an LLM-generated score.

Use PPIFlow nanobody/antibody for authorized initial generation. TorchCraft VHH
is available only for its supported single-chain mode; not paired VH/VL. For
an existing pose, use compatible PPIFlow Partial Flow with Rosetta-selected fixed
points. Follow the rule/capability-filtered menu, preserve target and fixed
positions, and rescore with TorchScore. Do not substitute guessed CDR ranges for
measured fixed-point evidence. Standalone redesign is a separate authorized model
request; it does not add regeneration to the autonomous menu.
