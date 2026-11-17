# Qwen scientific planner

The default `autonomous.decision_policy` is `planner`. PPIFlow/TorchCraft first
generate candidates, then TorchScore measures complex ipTM and pTM. No confidence
threshold silently rejects or routes a candidate before Qwen considers it.

```yaml
autonomous:
  decision_policy: planner
  # Optional qualified model ID; otherwise use the configured chat model (configure Qwen).
  # planner_model: your-provider/your-qwen-model
  decision_context_tokens: 16384
  decision_timeout: 120
```

Existing immutable run snapshots keep their original policy. Start a new run to
change a saved scientific policy. Set `decision_policy: rules` to explicitly use
the deterministic workflow. The legacy `agent` setting retains its historical
rule-constrained prioritization behavior for checkpoint compatibility.

## Planner actions

| Action | Behavior |
| --- | --- |
| `stop` | Discard the selected poor candidate; other candidates continue |
| `continue` | Obtain real TorchFold/DockQ if missing, then reconsider; a later `continue` completes and accepts that measured candidate |
| `analysis` | Run a selected evaluation skill, save its evidence, and ask Qwen again |
| `partial_flow` | Call the validated mutation selector, encode fixed/masked positions for compatible PPIFlow Partial Flow, generate a child, then TorchScore |
| `torchopt` | Call the mutation selector, run a compatible TorchOptimization job, reject out-of-mask outputs, then TorchScore the admitted child |

Qwen chooses acceptance using the full measured evidence; the configured confidence
and DockQ thresholds are recommendations in planner mode. Accepted means a
computational planner decision, not measured affinity or experimental validation.
There is no global round/sample/batch cap or mandatory Partial Flow → optimization
order. A parent is closed after a scored optimization child is produced; its child
gets a fresh decision. Exact unchanged sequence/structure children are duplicates.

## Evaluation skills and scientific context

Before every planner decision the context builder reads trusted installed guidance
for developability-filter, structure-analysis, epitope-analysis, evolutionary-analysis
and post-filter. It supplies available analysis choices, rule recommendations,
mutation constraints, history, measured confidence, interface/hotspot evidence and
compressed analysis results. Skill text is compressed; the context budget bounds
the serialized evidence. Oversized essential evidence causes a recorded rule
fallback rather than an invented measurement or unbounded prompt.

Analysis is chosen explicitly, not automatically run in full on every candidate:

- Developability: chemical PTM motifs, surface PSH/PNC/PSC and residue-level
  hydropathy/SASA. EpiMatrix stays unavailable until connected.
- Structure/epitope: heavy-atom contacts plus mapped hotspot diagnostics. Missing
  hotspot mapping is unknown, not an off-epitope conclusion.
- Evolution: actual recorded parent IDs.
- Post-filter: measured confidence ranking; incomplete tuples remain unranked.
- Rosetta interface: optional registered ResResE energy analysis. Its per-residue
  energies and < -5 fixed-point suggestions inform Qwen, but do not force its mask.

Results are persisted in Candidate `analysis_results`, developability in
`metadata.developability_evidence`, and per-skill JSON under the candidate directory.
Every subsequent decision sees updated evidence. New children do not inherit their
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
