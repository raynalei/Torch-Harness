# Scientific-agent evidence and skills

Default `decision_policy: planner` lets Qwen choose scientific actions after
TorchScore and after each subsequent evaluation or optimization. Python validates
real model capabilities, task constraints and returned artifacts. Rule thresholds
are advice and fallback; see [the planner guide](autonomous-feedback.md).

## Skill inventory

The bundled catalog retains protein-design and 15 specialists. Executable
analysis skills are developability-filter, structure-analysis, epitope-analysis,
evolutionary-analysis and post-filter. Before every planner decision, the context
builder reads their trusted installed SKILL.md files and includes compressed
guidance plus Rosetta interface and mutation-selector guidance. Qwen chooses which
missing evaluation to execute; it does not automatically run every skill.

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
output validation only. A new child gets fresh TorchScore and analysis evidence.

Planner response/schema/selection errors use deterministic fallback. Execution
errors retain the planned action and remote IDs for recovery, without selecting
another submission. Exact unchanged sequence/structure children are duplicates.
`rules` explicitly runs the retained staged policy; legacy `agent` snapshots keep
their historical rule-constrained prioritization and context behavior.

The `initial_round` directory owns shared service adapters, executor and checkpoint
formats. It remains an internal compatibility package, not a public mode or
required initial-round CLI/tool option. Generic protein_design_start/status/resume
entry points drive the planner through that shared executor.
