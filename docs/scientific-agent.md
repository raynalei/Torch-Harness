# Scientific-agent evidence and skills

The scientific agent prioritizes **existing executable options**. Python retains
ownership of routing, thresholds, capability validation and execution. The builder
does not run a model, measure a new property or add a regeneration action.

## Current skill inventory

The plugin exposes the general `protein-design` skill and 15 specialist skills:

| Category | Skills |
| --- | --- |
| Workflow | `protein-design` |
| Antibody design | `cdr-point-mutation`, `cdr-full-redesign`, `antibody-inverse-folding`, `esm2-guided-mutation` |
| Mini-binder design | `minibinder-full-redesign`, `minibinder-point-mutation`, `minibinder-inverse-folding` |
| Analysis and filtering | `evolutionary-analysis`, `epitope-analysis`, `structure-analysis`, `developability-filter`, `post-filter` |
| Public-source lookup | `pdb-lookup`, `uniprot-lookup`, `pubmed-lookup` |

Three decision guides are references of the general skill, not extra executable
model tools:

- [binder_failure_diagnosis.md](../opendde_harness/plugin/protein_design/skills/protein-design/references/binder_failure_diagnosis.md)
  distinguishes off-epitope evidence from low confidence and missing evidence.
- [generation_strategy.md](../opendde_harness/plugin/protein_design/skills/protein-design/references/generation_strategy.md)
  explains initial PPIFlow/TorchCraft, energy-fixed Partial Flow and optimization.
- [continue_or_stop.md](../opendde_harness/plugin/protein_design/skills/protein-design/references/continue_or_stop.md)
  explains finite lineages, stopping, recovery and why new exploration needs a
  separate authorized request under the current menu.

The context builder reads their trusted summaries, plus strategy sections relevant
to the current menu. Full references remain available through `protein-design`.

## Extract → enrich → compress

`opendde_harness/plugin/protein_design/scientific_agent/context_builder.py` exposes
three projections and `build_context` / `context_for_runner` entry points:

1. **Extract** reads existing Candidate JSON. Autonomous dictionaries and serialized
   core Candidates expose structural confidence, interface metrics, hotspot evidence
   and computed developability fields. Paths identify structures; coordinates,
   full sequences and raw model responses are not sent. Metric units are retained
   when supplied. Missing/nonfinite TorchScore values stay unknown; generic ipTM
   and AF3Score are never relabeled as TorchScore. Legacy `metadata.fold`,
   `gate_evidence` and `developability_evidence` are supported. No objective
   developability computation is fabricated when absent.
2. **Enrich** recomputes the rule recommendation from measured metrics and task
   thresholds, separately reports capability-filtered actions, and includes inherited
   fixed/mutable positions, chain roles, task constraints and checkpoint experiment
   history. History includes action state and actual child result metrics. Planned
   or failed actions are not described as completed improvements. Availability means
   configured, validated capability, not proof of remote inference success.
3. **Compress** includes only candidates behind visible options and bounds detailed
   residue evidence and recent history. Raw API requests, service credentials,
   complete energy tables and unrelated metadata never enter the context. Fixed-point
   summaries explicitly report omitted positions. Strategy detail is stage-specific;
   optimization decisions do not receive irrelevant per-residue energy rankings.

```yaml
autonomous:
  decision_policy: agent
  decision_context_tokens: 4096
```

This is a **per-prompt context limit**, not a round/sample/batch budget. It excludes
the fixed system message and the separate 1500-token response allowance. Serialized
UTF-8 bytes conservatively bound byte-tokenized Qwen context without fetching any
tokenizer vocabulary. Full records remain in the checkpoint.

Menus can be paginated to fit: `option_id` remains the original executable index,
and omitted counts are reported. Unshown candidates remain available on later
iterations. Qwen may not select an index absent from the context it received.
Optional history/guidance/detail is reduced before losing essential confidence,
lineage constraints, rule recommendations and executable identities. An impossible
budget, invalid response or unavailable provider records a fallback and uses the
existing deterministic legal option. `decision_policy: rules` does not call the
builder or Qwen.

## Why initial_round remains

The autonomous runtime is not independent of the `initial_round` package:

```text
CLI / protein_initial_round_start
  → automation / worker
  → InitialRoundRunner.run
  → generate → run_autonomous → score / Rosetta / refold / optimize
  → checkpoint and export
```

The shared package owns input validation, service allocation, path mapping,
TorchScore, TorchFold/DockQ, candidate lineage and recovery. Deleting it would
break autonomous execution. Its single-round branch and existing CLI/tool names
are retained as compatible entry points; they do not replace the staged autonomous
router. Removing the package would require a separately planned migration of all
imports, configuration snapshots, tools and saved workers. The new builder has
its own scientific_agent namespace while deliberately reusing the current router.
