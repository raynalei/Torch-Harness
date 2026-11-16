# Four-metric autonomous feedback

Enable `autonomous` to run initial generation followed by bounded Qwen decisions.
Without it, the existing single-round workflow remains available.

```yaml
autonomous:
  min_score: 0.5
  min_ptm: 0.8
  min_dockq: 0.49
  contact_distance: 5.0
  max_rounds: 3
  max_samples: 100
  batch_size: 5
  max_batch_size: 20
  patience: 2
  decision_policy: agent
```

## Executable decision table

Rows have the following priority for each candidate. Comparisons are strict `>`;
equality belongs to the failing branch. The selected API supplies complex ipTM/pTM.
No maximum over individual chain scores is taken by Harness.

| Hotspot | ipTM > 0.5 | pTM > 0.8 | DockQ > 0.49 | Rule menu |
| --- | --- | --- | --- | --- |
| No | Any | Any | Any | Regenerate |
| Yes | No | Any | Any | Regenerate |
| Yes | Yes | No | Any | TorchOptimization |
| Yes | Yes | Yes | No | Partial Flow |
| Yes | Yes | Yes | Yes | Partial Flow / Expand / Stop |

`policy.rule_actions` implements the table. Required unknown/nonfinite evidence does
not pass a row. Earlier rows do not require later metrics: a low ipTM/pTM candidate
does not need refolding before regeneration/optimization. High-score candidates are
refolded and evaluated with DockQ. Infrastructure/metric extraction failures pause
execution for checkpointed recovery rather than causing a scientific decision.

Each batch is scored candidate by candidate and deduplicated within its source.
Hotspot contacts are measured on the generated complex before refolding; after a
refold is available, contacts are evaluated on that structure. The recorded
`hotspot_structure` identifies the basis. Mapping uses the explicit PPIFlow hotspot
PDB IDs, unique target sequence matching and chain mappings. At least one specified
hotspot must have a non-hydrogen atom within 5 Å of a binder non-hydrogen atom.
Missing/ambiguous mappings stay unknown. DockQ compares refold vs the generated
design pose, taking the minimum over target–binder interfaces, not binder–binder.
It is not an experimental binding score.

## Qwen menu and automatic execution

`options_for` combines the executable entries supported by evaluated candidates.
It removes missing services, unsupported modes and unencodable residue selections.
For example, paired VH/VL cannot use the current TorchOptimization interface, and
mini-binders have no registered Partial Flow mode. A table-required action that is
unavailable is recorded; no unrelated action is substituted for that candidate.
If all candidates lack an executable action the run stops with `no_legal_actions`.

Qwen uses the existing configured chat provider and chooses one menu index/reason.
Hotspot, ipTM, pTM and DockQ determine the permissible action categories.
Rosetta interface_dg and the recorded ranking are also sent to Qwen for comparisons
within the legal menu. Lower interface energy is preferred; there is no additional
energy cutoff that changes the four-metric rule rows. Resource/capability limits
constrain execution; flexibility and local RMSD are not used. Invalid/unavailable provider responses fall back to the first executable
entry and record the fallback. `Stop` appears only for a fully passing candidate.

No general per-action approval is required. Refinements execute within the run
budget. The only new user interaction is a selected Partial Flow whose automatic
CDR identification failed: it pauses for a concrete region answer. Each decision
and request is checkpointed before POST.

## Automatic Partial Flow positions

`regions.infer_cdr_positions` aligns a binder sequence to built-in heavy/light
variable-domain framework references with Biotite BLOSUM62 affine-gap alignment.
Reference CDR positions and query insertions map to mutable positions. Acceptance
requires framework coverage >=90%, framework identity >=70%, conserved framework
anchors and coverage of all three CDRs. These are **Kabat-like reference-alignment
estimates**, not certified IMGT numbering; divergent, truncated, tandem or ambiguous
PDB domains may have no reliable automatic annotation. No external prediction model
is used. If Qwen selects Partial Flow and alignment is uncertain, the worker saves
`pending-regions.json` and status `awaiting_regions`, then exits without submitting
Partial Flow or reserving its sample budget. Normal resume cannot bypass the question.

`python_partial_positions` translates these sequence positions to actual candidate
PDB IDs, including numbering offsets. `cdr_position` contains the mutable CDRs;
`fixed_positions` is the complement over all binder chains. Missing C-alpha atoms,
insertion-code ambiguity or unsupported numbering prevent submission and are reported.
Unsupported model modes/chain mappings still remove the action; a manual answer
cannot make an unsupported backend mode valid. A refinement
output must retain target sequences and must not alter binder sequences outside
the computed CDRs. Candidates from TorchCraft can use the PPIFlow partial API only
when their generated chain IDs match its configured chain roles.

Each candidate is refined at most once. A new child may itself be refined in a
later round; global round/sample caps prevent unbounded recursion.

## Ranking, rounds and recovery

Autonomous ranking uses **100*selected_ipTM - rosetta_interface_dg**, descending,
then ipTM/pTM/DockQ for ties. Rosetta is a required registered service. Every evaluated
candidate receives interface scoring: on the generated complex for early low-score
branches, or the refold complex after successful refolding. `rosetta_structure` records
which structure was used and is exposed to Qwen. Missing energy is an evaluation
failure, not an assumed zero. Rosetta failures use the same failed-step recovery.

## Answering a CDR identification question

Status/list expose the pending candidate, source PDB, failure reason, metrics and
`question_id`. The agent must ask the user rather than inventing a CDR answer. The
user provides `cdr_position` in the candidate's real chain/residue numbering;
`fixed_positions` can be omitted and Python fixes every other binder residue.
If both are supplied they must form an exact, disjoint binder partition. Inputs
are validated against the unchanged candidate structure; stale question IDs fail.
Use `protein_initial_round_regions` with the actual user answer, or:

```bash
ddeharness protein-design initial-round-regions --run-dir /existing/run \
  --question-id <current-question-id> --cdr-position B98-110
```

The validated answer is saved per candidate and the original run resumes. Prior
successful generation/scoring/refolding/energy results are reused. No answer leaves
the run paused; no scientific request is submitted while waiting. A detached worker
does not inject a new chat turn: status/list expose the question when the agent
tracks the run or reconnects.

`max_rounds=3` means up to three additional actions after the initial batch. Initial
requested samples count toward `max_samples=100`. Expand requests twice batch_size;
regenerate requests batch_size, capped by max_batch_size and remaining budget.
Partial Flow/optimization reserve one imported candidate each. Their internal GPU
operations are not a GPU-hour budget. Stop reserves zero samples and ends immediately.
Two consecutive follow-ups without a new accepted sequence or a better metric tuple
stop with `no_improvement`. Other stops are max_rounds, sample_budget and agent_stop.

The existing saved run directory, worker, task index, attempt history and failed-step
recovery remain in use. Unknown submissions keep their remote IDs and never silently
POST again. All scientific API allocations still start through sbatch on allocated
compute nodes. See [workflow configuration](workflow-configuration.md).
