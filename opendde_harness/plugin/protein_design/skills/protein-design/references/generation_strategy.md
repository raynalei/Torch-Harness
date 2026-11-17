# Generation strategy

Use only the capability-filtered menu. A registered model alone is not an executable action: mode, chain mapping, input artifacts and parent lineage must also be compatible. Do not invent an alternative action when one is unavailable.

## Initial generation

PPIFlow supports configured nanobody, paired-antibody and binder initial modes.
TorchCraft supports the registered VHH and mini-binder target-conditioned modes;
paired VH/VL is not supported by this integration. Inputs, framework paths,
hotspots and requested sample counts come from the authorized task configuration.
The current autonomous menu does not regenerate or expand initial batches.
Standalone monomer generation is a separate tool capability, not target binding.

## Partial redesign

For an initial candidate outside the direct-refold gate, use measured Rosetta
residue energies. Fix binder residues below the configured energy threshold
(default strictly < -5); other binder residues are mutable. The current API
supports compatible partial_nanobody/partial_antibody modes, not mini-binder
Partial Flow. Preserve target sequence, binder length and selected fixed points.
Do not replace energy-selected positions with guessed CDRs or invent scores.

## TorchOptimization

Only a Partial Flow child that misses the joint high-confidence gate can enter
TorchOptimization. It needs a registered optimizer and compatible single target
and binder chains. Paired VH/VL is unsupported. Keep inherited residue/target
constraints. Score the actual optimized child anew; optimizer completion itself
is not evidence of improved confidence or binding. Never optimize an optimized
child again under the current finite stage policy.
