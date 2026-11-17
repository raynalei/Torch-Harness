# Continue, explore or stop

Finish executable candidate lineages before stopping. Each lineage has at most one Partial Flow then one optimization; there is no global round, sample, batch or patience cap. New-candidate exploration is not in the autonomous menu.

## Continue

Python owns the strict thresholds. Initial ipTM < 0.5 terminates that candidate.
Joint ipTM > 0.8 and pTM > 0.8 goes directly to TorchFold/DockQ. Other initial
candidates use Rosetta fixed points and Partial Flow. Other Partial Flow children
use TorchOptimization. Other optimization children terminate. These are defaults;
use the supplied task thresholds, including exact boundary comparisons.
Qwen chooses priority among the actual legal options; it cannot move a candidate
to another branch or relax task constraints.

## Explore a new candidate

Continue other existing candidates when one fails. If every existing lineage is
finished or scientifically rejected, recommend a new authorized generation task
with a revised seed/framework or reviewed target/epitope inputs when the evidence
supports that change; do not claim it will rescue affinity. Do not add regeneration,
expansion, new seeds, a new design objective or another backend to the legal menu.
A genuinely new generation request is a separate user-authorized task with its
own inputs, not an invented rescue action for a stopped lineage.

## Stop and recover

Accept only after the configured TorchFold confidence and DockQ checks pass.
Stop is available only after no executable follow-up remains. A missing service
or unsupported mode is an unavailable action, not permission to substitute one.
Missing/nonfinite metrics or logs and uncertain submissions are infrastructure
failures: pause and recover the existing checkpoint/remote ID, never infer zero
or blindly repost. Historical planned/failed actions are not completed experiments.
