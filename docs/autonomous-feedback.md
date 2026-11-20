# Two-stage Qwen scientific planner

New generic workflow starts default to `decision_policy: planner` with
`planner_layout: two_stage`. Each continuing candidate lineage gets two scientific
decision stages; a first-stage rejection ends early. This is workflow structure,
not the removed global round/sample/batch budget. Qwen still judges candidates
using scientific context, while rules remain fallback.

```yaml
autonomous:
  decision_policy: planner
  planner_layout: two_stage
  # planner_model: your-provider/your-qwen-model
  decision_context_tokens: 16384
  decision_timeout: 120
```

PPIFlow `binder` and TorchCraft `mini_binder` use the same score-only route.
The generated complex goes through the configured confidence scorer, Rosetta
interface scoring, then TorchFold and DockQ. The candidate is exported as
`scored_only` and skips both Qwen decision stages and follow-up optimization.

## Stage 1: TorchScore and Rosetta only

PPIFlow/TorchCraft generates initial candidates. Every candidate is measured by
TorchScore (ipTM/pTM) and Rosetta (interface residue energies) before Qwen judges it.
Even high- or low-confidence candidates get both evaluations; there is no automatic
confidence gate before Qwen. Native developability, contacts and lineage analyses
are not performed in this stage or included as first-stage evidence.

Qwen can stop a poor candidate, advance without optimization (`continue` or
`analysis`), or choose compatible Partial Flow/TorchOptimization via the mutation
selector. All first-stage decisions in the initial batch finish before stage-two
evaluation begins. Fixed points and masks use actual sequence/PDB position mapping.

## Stage 2: full evaluation, then final validation

Evaluate the current candidate: the initial candidate if stage 1 skipped optimization,
or the actual scored optimization child otherwise. Stage 2 uses TorchScore,
Rosetta and the installed developability/structure/epitope/evolution/post-filter
backends. Same-structure cached measurements can be reused; parent measurements
are not relabelled as child evidence. EpiMatrix remains uncomputed; IgLM, RFdiffusion
and MPNN are not silently added as backends.

Qwen makes the second scientific decision: stop, continue to final validation, or
perform one selected mask/fixed-point optimization. A stage-two optimization child
gets its own TorchScore and then TorchFold/DockQ; no third Qwen decision is opened.
Final acceptance uses the existing measured final-validation checks. A child does
not inherit the parent's developability evidence as if recomputed on the child.
These are computational checks, not measured affinity or experimental validation.

Counters distinguish `scientific_decision_stages=2`, `followup_rounds` (stages
actually reached) and `planner_decisions` (individual per-candidate decisions).
Two surviving candidates can therefore generate four Qwen choices across two stages.

## Existing runs and explicit rule mode

Existing immutable run snapshots keep their original policy. The internal layout
default remains `iterative` and is omitted from legacy serialization so existing
checkpoint hashes remain compatible. Generic `protein_design_start` / CLI `start`
inserts `two_stage` for new planner tasks; explicit `planner_layout: iterative`
retains the preceding planner behavior. Use a new run to change a saved policy.
`decision_policy: rules` preserves the deterministic staged workflow; legacy
`agent` preserves its historical rule-constrained prioritization.

## Evaluation skills and scientific context

Before stage-one decisions the builder loads only TorchScore/Rosetta evaluation
guidance and mutation-selector instructions. Before stage-two decisions it reads
trusted installed guidance for developability-filter, structure-analysis,
epitope-analysis, evolutionary-analysis and post-filter. It supplies available analysis choices, rule recommendations,
mutation constraints, history, measured confidence, interface/hotspot evidence and
compressed analysis results. Skill text is compressed; the context budget bounds
the serialized evidence. Oversized essential evidence causes a recorded rule
fallback rather than an invented measurement or unbounded prompt.

Stage-two evaluation runs these available backends before the second decision:

- Developability: chemical PTM motifs, surface PSH/PNC/PSC and residue-level
  hydropathy/SASA. EpiMatrix stays unavailable until connected.
- Structure/epitope: heavy-atom contacts plus mapped hotspot diagnostics. Missing
  hotspot mapping is unknown, not an off-epitope conclusion.
- Evolution: actual recorded parent IDs.
- Post-filter: measured confidence ranking; incomplete tuples remain unranked.
- Rosetta interface: required registered ResResE energy analysis in the two-stage layout. Its per-residue
  energies and < -5 fixed-point suggestions inform Qwen, but do not force its mask.

Results are persisted in Candidate `analysis_results`, developability in
`metadata.developability_evidence`, and per-skill JSON under the candidate directory.
The second scientific decision sees updated evidence. New children do not inherit their
parent's measured scores or developability as if measured on the child.

## Mutation selector

`protein_design_mutation_selector` partitions explicit **one-based observed binder
sequence positions** into fixed points and a mutation mask. Example selection:

```json
{"fixed_positions":{"B":[1,8]},"mask_positions":{"B":[3,4,5]}}
```

Either field may be omitted. With only fixed points, other authorized positions
are mutable; with a mask, its complement is fixed. Reject overlap, nonexistent or
target chains, duplicate/out-of-range positions, empty mutable sets, and attempts
to alter inherited task-fixed positions or exceed authorized redesign regions.
The workflow independently reads saved task constraints, rather than trusting
caller-supplied replacements. It records tool inputs, the validated partition and
optimizer requests in the decision checkpoint. PDB numbering and insertion codes
are distinct from sequence indices; native Partial Flow range encoding validates
the actual structure before submission.

Partial Flow requires both fixed and mutable residues and currently supports
compatible nanobody/antibody modes, not mini-binders. TorchOptimization currently
supports compatible single target/binder chains, not paired VH/VL. **Its API has
no native residue-mask argument.** The mask is enforced on returned structures:
changes outside the mask, changed target sequences or lengths are rejected. This
can discard proposals; the Harness does not claim the optimizer searched only the
requested mask. Native mask support needs a future TorchOptimization API update.

## Rules and recovery

Planner timeout, invalid JSON/menu choices or invalid mutation selections trigger
a recorded deterministic fallback. Model execution errors do not trigger fallback
submissions; the planned action and remote ID remain available for recovery. If
neither planner nor rule route can execute (for example a required fallback service
is missing), pause with an error instead of reporting a scientific rejection.

The retained rule route uses strict comparisons: initial ipTM < 0.5 terminates;
both ipTM > 0.8 and pTM > 0.8 refold; other initial candidates use Rosetta and
energy < -5 fixed points for Partial Flow; other partial children optimize;
other optimized children terminate. Task-configured thresholds override defaults.
Rules still require the Rosetta service. The ResResE contract and uploaded energy
algorithm remain unchanged (negative target–binder pair sums within 12 Angstrom,
CB/CA fallback). Scalar interface energies do not replace the residue table.

Completed decisions, analysis and model jobs are reused on resume. Uncertain
submissions retain their IDs; never blindly repost. Use generic workflow status,
resume and candidates tools. No OpenDDE compute service participates.
