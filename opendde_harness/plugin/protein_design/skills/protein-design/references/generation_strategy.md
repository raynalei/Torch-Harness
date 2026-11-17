# Generation strategy

In default planner mode, choose from the capability-filtered menu using scientific context. Rule score thresholds are recommendations and fallback, not action gates. A registered model alone is not an executable action: mode, chain mapping, input artifacts and parent lineage must also be compatible. Do not invent an alternative action when one is unavailable.

## Initial generation

PPIFlow supports configured nanobody, paired-antibody and binder initial modes.
TorchCraft supports the registered VHH and mini-binder target-conditioned modes;
paired VH/VL is not supported by this integration. Inputs, framework paths,
hotspots and requested sample counts come from the authorized task configuration.
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
