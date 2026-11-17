# Continue, explore or stop

Default Qwen planner uses measured scientific context, history and evaluation
skills to choose an actual executable action. No fixed score gate hides candidates
before that judgment. Rule recommendations remain available for fallback.

## Continue

In the default two-stage layout, stage one evaluates only TorchScore and Rosetta.
First-stage continue/analysis advances to full second-stage evaluation; either
compatible optimizer can instead create a stage-two child. Stage-two continue
requests final TorchFold/DockQ. A second-stage optimization child is freshly
TorchScored and sent directly to final validation, without a third Qwen decision.
Final validation retains measured acceptance checks. Stop discards a poor candidate
at either stage. EpiMatrix remains uncomputed.

Use mutation_selector before either optimizer. Fixed points and masks preserve
inherited task constraints. There are two scientific stages per continuing lineage,
not a global sample/batch cap. Existing iterative planner snapshots retain their
previous analysis/continue loop and explicit rules mode retains its staged routing.

## Explore a new candidate

Other existing candidate lineages remain available after one is discarded. Initial
regeneration is not a planner action: changed target/framework/objectives are a
separate scientific task. Do not invent an unregistered model or affinity prediction.

## Stop and recover

Stop discards only the selected poor candidate. Do not confuse unknown confidence,
missing services or failed jobs with poor structural quality. Execution failures
pause their saved action and remote job ID. Planner failure uses deterministic
staged rules when executable, otherwise pauses. The explicit rules mode preserves
its strict historical confidence/interface-energy routing and acceptance gates.
