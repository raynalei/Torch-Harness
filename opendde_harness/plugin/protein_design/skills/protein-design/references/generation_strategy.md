# Generation strategy

In default planner mode, choose from the capability-filtered menu using scientific context. Rule score thresholds are recommendations and fallback, not action gates. A registered model alone is not an executable action: mode, chain mapping, input artifacts and parent lineage must also be compatible. Do not invent an alternative action when one is unavailable.

## Initial generation

PPIFlow supports configured nanobody, paired-antibody and binder initial modes.
TorchCraft supports the registered VHH and mini-binder target-conditioned modes;
paired VH/VL is not supported by this integration. Inputs, framework paths,
hotspots and requested sample counts come from the authorized task configuration.
RFD4 supports native binder-generation inputs supplied by the task as rfd4_inputs,
using an existing absolute .json/.cif file or a path relative to RFD4_INPUTS. Do not invent a binder spec from a target
PDB or reuse an old target spec. Its output chain IDs must match the workflow
roles explicitly supplied by the user.
The output target sequence must match the current task.
The current autonomous menu does not regenerate or expand initial batches.
Standalone monomer generation is a separate tool capability, not target binding.

## Partial redesign

Use mutation_selector to choose fixed points or masks from measured interface,
hotspot, residue hydropathy/SASA and chemical risk evidence. Rosetta energies below
-5 can recommend favorable fixed points but are not mandatory in planner mode. The current API
supports compatible partial_nanobody/partial_antibody modes, not mini-binder
Partial Flow. Preserve target sequence, binder length and selected fixed points.
Do not invent CDR numbering or scores; the selector preserves inherited task constraints.
In rules fallback, energy < -5 selects the historical fixed-point partition.

## TorchOptimization

The planner can choose TorchOptimization for a compatible initial or optimized
candidate using all measured evidence. It needs a registered optimizer and compatible single target
and binder chains. Paired VH/VL is unsupported. Keep inherited residue/target
constraints. Score the actual optimized child anew; optimizer completion itself
is not evidence of improved confidence or binding. The current API does not accept native residue masks: validate all outputs against
the selector partition. The rules fallback retains its historical finite stage order.


## RFD4 Partial Flow

Use rfd4_partial for an existing complex with a registered RFD4 service and shared
rfd4_input_root. Harness exports the current parent as real CIF under RFD4_INPUTS,
then requests inference_base with a partial trajectory (default fraction 0.2,
overridable in evaluation.rfd4_options). The API has no native per-residue mask:
mutation_selector defines sequence constraints enforced on the returned structures.
This does not freeze coordinates or guarantee that RFD4 can satisfy a narrow mask.
Target sequence, chain IDs, binder length and all positions outside the selected
mask must be preserved; otherwise the action fails without a fabricated child.

## ProteinMPNN

Use proteinmpnn for sequence redesign on existing binder chains. The native API
returns FASTA, not coordinates, and exposes chain-level design only. Use the
mutation selector, exclude the target chains from design_chains, and reject samples
that alter fixed residues or positions outside the mask. Narrow masks may produce
no admissible samples. There is no native residue-mask guarantee.
Harness selects one admitted designed sequence (not the native FASTA record),
realizes its complex with TorchFold, restores the task chain IDs, and only then
runs fresh TorchScore. This preliminary fold is structure realization, not a
Qwen decision or successful final validation. Later DockQ compares this realized
complex with its validation refold; it does not measure retention of the parent's
pre-mutation geometry. Source/weights/default tuning remain server-managed.

After any first-stage optimization, evaluate its child in stage two. After any
second-stage optimization, rerun TorchScore and proceed directly to TorchFold/DockQ.
There is no third Qwen decision. Rules fallback keeps its existing route.
