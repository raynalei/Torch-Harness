# Scientific-agent evidence and skills

New generic starts use `decision_policy: planner`, `planner_layout: two_stage`.
Stage one evaluates only TorchScore/Rosetta before Qwen; stage two evaluates the
current candidate with full evaluation skills before the second Qwen decision. Python validates
real model capabilities, task constraints and returned artifacts. Rule thresholds
are advice and fallback; see [the planner guide](autonomous-feedback.md).

## Skill inventory

The bundled catalog retains protein-design and 15 specialists. Executable
analysis skills are developability-filter, structure-analysis, epitope-analysis,
evolutionary-analysis and post-filter. Before the second-stage decision, the context builder reads their trusted installed
SKILL.md files. First-stage guidance contains only TorchScore/Rosetta evaluation
and mutation-selector instructions. The two-stage path completes available native evaluation before stage-two judgment;
legacy iterative snapshots retain their explicit analysis selection.

Generation and mutation skills use registered Torch model APIs and the native
mutation selector. Public PDB/UniProt/PubMed lookups remain independent tools.
Antibody/mini-binder inverse folding and ESM2 remain explicitly unavailable reference
skills because there is no registered backend. EpiMatrix is a reserved interface,
never replaced by a fabricated immunogenicity score. No OpenDDE compute is used.

## Extract → enrich → compress

`scientific_agent/context_builder.py` projects Candidate JSON:

1. Extract measured structural confidence, interface energies, mapped hotspots,
   developability and saved evaluation results. Parent scores are not measurements
   of a child. Missing/nonfinite values stay unknown. Chemical PTM is distinct from
   structural pTM. Surface descriptors keep their named scales and method.
2. Enrich with advisory rule recommendations, actual executable actions, inherited
   fixed/mutable task constraints, chain lengths, experiment history and installed
   evaluation skills. `protein_design_mutation_selector` independently validates
   planner fixed points or masks against saved task constraints before submission.
3. Compress to the configured context budget. Complete flat finite metrics and
   the current parent's constraints accompany its visible options. Residue evidence
   prioritizes chemical-liability positions, then exposed high-hydropathy residues,
   and explicitly reports omissions. Contacts/lineage/ranking are summarized;
   original results remain under the candidate directory and in checkpoints.

```yaml
autonomous:
  decision_policy: planner
  planner_layout: two_stage
  decision_context_tokens: 16384
  # planner_model: your-provider/your-qwen-model
```

The budget limits serialized user evidence, not scientific rounds or samples.
UTF-8 bytes conservatively bound byte-tokenized payload size. The fixed system
prompt and separate 2200-token response allowance are additional. Options retain
their original IDs when paginated. Planner essential evidence and evaluation
skill guidance are not silently stripped to force a decision into an impossible
budget; that failure invokes the recorded rule fallback. API credentials and raw
submission requests are never passed to Qwen.

## Decisions and recovery

Planner actions are stop, continue, analysis, partial_flow and torchopt. Analysis
results return to the context before the next decision. Optimization records the
mutation selector arguments, validated partition and actual model request. Native
Partial Flow receives encoded ranges; TorchOptimization masks currently apply to
output validation only. A first-stage optimization child gets fresh stage-two evaluation. A second-stage
optimization child gets fresh TorchScore and final TorchFold/DockQ, without a third
Qwen decision or inherited parent analysis.

Planner response/schema/selection errors use deterministic fallback. Execution
errors retain the planned action and remote IDs for recovery, without selecting
another submission. Exact unchanged sequence/structure children are duplicates.
`rules` explicitly runs the retained staged policy; legacy `agent` snapshots keep
their historical rule-constrained prioritization and context behavior.

The `initial_round` directory owns shared service adapters, executor and checkpoint
formats. It remains an internal compatibility package, not a public mode or
required initial-round CLI/tool option. Generic protein_design_start/status/resume
entry points drive the planner through that shared executor.


Rosetta interface ranking uses the API's `dG_separated`, stored as
`metrics.rosetta_interface_dg`, separately from the summed residue-pair interface
energy. When measured, `ranking_score = 100 * torchscore_iptm - rosetta_interface_dg`.
Unmeasured dG remains unknown/unranked and is exported with a null objective; it is
never inferred from residue energies or treated as zero. This changes ordering,
not planner stages, acceptance thresholds or optimizer constraints. The context
builder includes both values before each Qwen decision. Rosetta remains a registered
API backend: Harness calls `/v1/tools/interface_analyze`; the cluster service runs
PyRosetta and returns residue logs and interface metrics/scorefiles.
