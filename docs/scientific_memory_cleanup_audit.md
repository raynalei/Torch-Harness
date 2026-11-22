# Scientific memory and cleanup audit (feat/autonomous_score_routing)

## Verified active paths

- `scientific_agent/context_builder.py::context_for_runner` supplies scientific metrics, legal options, packaged evaluation/optimization guidance, checkpoint decision history, and up to four retrieved cross-run episodes.
- `scientific_agent/episodic_memory.py` stores completed decisions as deduplicated JSONL under `autonomous.episodic_memory_path` or `work_dir/episodic_memory.jsonl`. It filters by current scientific mode, legal actions and run origin. Memory is untrusted evidence, never an executable command.
- `scientific_agent/planner.py`, `scientific_agent/two_stage.py`, and `initial_round/autonomous.py` call the context builder and write completed decisions. The planner uses a bounded legal-option menu and a fallback.
- `agents/skills.py::MODEL_SKILLS` maps active model integrations to packaged skill names; `context_builder.py` loads specific relevant skill guidance according to the stage/action. This is a **static skill guidance selector**, not a dynamic semantic retrieval router for learned skills.
- The new episodic IO guard makes offline or corrupted memory advisory, rather than a reason to fail a completed design step.

## Native OpenDDE components: deletion is **not safe yet**

The scientific decision path above does not instantiate `DesignMemory` or the native long-term backend. However these shared modules have independent callers:
- `agent/loop/factory.py` builds the host memory backend.
- `context_engine/factory.py` and `context_engine/segments/memory.py` render/retrieve long-term memory for general conversations.
- `plugin/protein_design/servers/worker.py` constructs `DesignMemory` and passes it to the legacy design orchestrator.
- `agents/skills.py` imports `memory_engine.skill_forge.documents` for the skill catalog.

Therefore deleting `opendde_harness/memory_engine/`, `opendde_harness/plugin/memory/` or `core/memory.py` alone breaks supported import/runtime paths. Removal should be a separate, atomic migration deleting or replacing every caller, updating CLI/TUI/plugin discovery and tests, not an unreviewed cleanup of the scientific workflow.

## Skills and models: retain until dependencies are proved unused

The scientific planner references PPIFlow, TorchCraft, RFD4, TorchOptimization, ProteinMPNN, TorchScore, TorchFold, Rosetta, and evaluation skills (developability, structure, epitope, evolutionary, post-filter). Other bundled catalog skills may still serve the legacy worker, and the catalog validates that required skill files exist. A skill's absence from one planner prompt is **not** evidence that its directory can be removed safely. AF3Score and other configured legacy model backends may remain necessary for checkpoints and backward-compatible configurations.

## Verification needed before destructive pruning

1. Inventory all files/imports on this branch, all CLI/TUI/plugin entrypoints, dynamic plugin names, checkpoint/config compatibility and tests.
2. Identify each specific model/skill to deprecate, update registries and validators, then delete files in the same reviewed change.
3. Run unit/regression tests for scientific planner (iterative + two-stage), initial-round score-only, CLI/TUI, and legacy protein-design worker.
4. Perform a real Qwen + Slurm dry run, including unavailable-memory and resumed-checkpoint cases.

This audit makes no claim that the HPC integration or end-to-end routes have been tested.
