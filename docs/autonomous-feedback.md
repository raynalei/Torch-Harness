# Staged autonomous rounds

Enable `autonomous` for checkpointed follow-ups after PPIFlow/TorchCraft generation.
Autonomous runs require **TorchScore** for both complex ipTM and pTM. The separate
single-round workflow still supports AF3Score and TorchScore.

```yaml
evaluation:
  scoring_backend: torchscore
  rosetta_residue_glob: '*.out'
  # Other service connections and dockq_python as in the registered example.
autonomous:
  min_score: 0.5
  min_refold_score: 0.8
  min_ptm: 0.8
  interface_energy_threshold: -5.0
  interface_distance: 12.0
  min_dockq: 0.49
  max_rounds: 3
  max_samples: 100
  patience: 2
  decision_policy: agent
```

## Executable routes

| Stage | TorchScore result | Next operation |
| --- | --- | --- |
| Initial PPIFlow/TorchCraft | ipTM < 0.5 | Terminate this candidate |
| Initial PPIFlow/TorchCraft | ipTM > 0.8 and pTM > 0.8 | TorchFold → DockQ |
| Initial PPIFlow/TorchCraft | All other valid scores | Rosetta residue interface energies → fixed points → Partial Flow |
| Partial Flow output | ipTM > 0.8 and pTM > 0.8 | TorchFold → DockQ |
| Partial Flow output | All other valid scores | TorchOptimization |
| Optimization output | ipTM > 0.8 and pTM > 0.8 | TorchFold → DockQ |
| Optimization output | All other valid scores | Terminate this candidate |

The initial lower cutoff is strict: ipTM **equal to 0.5** proceeds to interface
scoring. Both high-confidence cutoffs are strict: **equal to 0.8** does not refold.
The initial 0.5 termination gate is not reapplied to Partial Flow output: low
Partial Flow scores go to optimization. Missing, nonfinite, out-of-range or
ambiguous scores are evaluation failures, not scientific rejections.

Hotspot contacts are still measured and recorded for inspection, but do not route
a candidate or block a high-confidence refold. DockQ compares the refolded complex
with its generated design pose, taking the minimum over target–binder interfaces.
It is not an experimental binding score. Final acceptance retains the configured
`evaluation.min_refold_iptm` and `evaluation.min_dockq` checks, and the autonomous
DockQ cutoff. A failed final check does not start another Partial Flow cycle.

## Rosetta interface energies and fixed points

`initial_round/interface_energy.py` adapts the computational functions in the
uploaded **1get_interface_energy.py**. It uses the generated complex and:

1. Reads `ResResE` records from Rosetta output, including `Res1`, `Res2`, `total`.
   The uploaded format encodes an uppercase chain as ASCII before the residue
   number: for example `ALA_661` is chain B, PDB residue 1.
2. Selects target–binder residue pairs whose CB distance is **≤12 Å**, using CA
   when CB is absent. Only negative pair energies contribute.
3. Sums those energies per binder residue, retaining the binder chain ID.
4. Fixes residues whose summed energy is **< -5**. Exactly -5 is not fixed.
   Every other binder residue is mutable in Partial Flow. The existing API's
   `cdr_position` field carries this mutable complement; no CDR alignment is needed.

The adapter supports multiple binder/target chains and reversed pair orientation;
it avoids double-counting repeated pair rows and rejects contradictory duplicates,
nonfinite energies and unsupported/ambiguous residue numbering. Plotting and the
uploaded script's fixed 32-process pool are unnecessary for workflow scoring.

**Deployment requirement:** the registered Rosetta service's existing
`/v1/tools/interface_analyze` operation must return exactly one readable shared-file
log containing the residue-pair `ResResE` table, selected by
`evaluation.rosetta_residue_glob` (default `*.out`). Configure the deployed Rosetta
protocol to emit this output if needed. A scalar `dG_separated` or ordinary score
file is insufficient and is never substituted. A native Conda/PyRosetta server
that implements this output contract is provided in
[deploy/rosetta](../deploy/rosetta/README.md); it requires no SIF or Rosetta CLI.

The worker saves `interface_residue_energies`, `interface_fixed_positions`, the log
path/hash and `interface-residue-energies.json`. `rosetta_interface_score` records
the sum for inspection. Autonomous candidates also save InterfaceAnalyzer's
`dG_separated` as `metrics.rosetta_interface_dg` and expose it in Qwen's decision
context. This interface dG is distinct from the filtered residue-energy sum;
fixed-point selection still uses per-residue energies below the configured threshold.
The value is read from API metrics or the returned scorefile, never inferred from
residue energies. Existing cached interface jobs can backfill the metric when
`score_interface` is revisited, without submitting another Rosetta job. Completed
workflows are not automatically reopened solely to backfill old candidate JSON.
High-confidence candidates, Partial Flow children and optimization children skip
Rosetta entirely. No selected fixed points terminates a candidate; a fully fixed
binder has no legal Partial Flow action. Unsupported modes/chain mappings are
reported as unavailable. The current registered Partial Flow supports nanobody
and antibody modes; minibinders have no compatible Partial Flow mode.

## Decisions, budgets and recovery

Qwen can choose only executable `partial_flow`, `torchopt` or `stop` menu entries.
An invalid/unavailable response falls back to the first legal option and is
recorded. Stop is offered only for an accepted candidate. There is no regeneration
or expansion action under this staged policy. Terminating one candidate does not
terminate other candidates in its batch.

Partial Flow and optimization reserve one imported candidate each. Initial
requested samples count toward `max_samples`. Round, sample and no-improvement
limits still bound execution; they can stop a run before every candidate finishes
its route. Each parent is refined at most once. Optimization output must preserve
the target and inherited fixed-residue constraints. Every child receives a fresh
TorchScore evaluation on its own structure.

Ranking uses TorchScore ipTM, then pTM and DockQ, descending; optional Rosetta
scores are not required for direct-refold candidates. `ranking_score` stores
100×TorchScore ipTM. Successful operations are checkpointed and reused after
restart. Uncertain submissions retain their IDs and are never silently reposted.
Infrastructure errors and missing residue logs pause for recovery using the same
run directory. Use a **new run name** for this changed scientific policy; do not
edit old checkpoint hashes to reuse decisions made under the former routing rules.
All scientific services still start through the registered sbatch allocations.
