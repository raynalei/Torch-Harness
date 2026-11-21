"""Episodic memory is cross-run evidence, not an action router."""
import json
from pathlib import Path
from types import SimpleNamespace

from opendde_harness.plugin.protein_design.scientific_agent.episodic_memory import (
    record_decision,
    retrieve,
    store_path,
)


def _runner(root: Path, memory: Path, *, mode="nanobody"):
    root.mkdir(parents=True, exist_ok=True)
    candidate = {
        "candidate_id": "parent",
        "source": "ppiflow",
        "status": "scored",
        "metrics": {"torchscore_iptm": 0.78, "torchscore_ptm": 0.82, "rosetta_interface_dg": -9.4},
        "diagnostics": {"diagnosis": "needs_optimization"},
    }
    config = SimpleNamespace(
        autonomous=SimpleNamespace(episodic_memory_path=memory),
        work_dir=root.parent,
        generators={"ppiflow": SimpleNamespace(scientific_mode=mode)},
    )
    checkpoint = SimpleNamespace(data={"candidates": [candidate]})
    return SimpleNamespace(root=root, config=config, checkpoint=checkpoint)


def test_memory_records_after_completion_and_retrieves_only_compatible_prior_runs(tmp_path):
    memory = tmp_path / "shared" / "episodes.jsonl"
    first = _runner(tmp_path / "run_a", memory)
    decision = {
        "round": 1,
        "parent_id": "parent",
        "action": "partial_flow",
        "state": "planned",
        "policy": "planner",
    }
    record_decision(first, decision)
    assert not memory.exists()
    decision["state"] = "completed"
    record_decision(first, decision)
    record_decision(first, decision)
    saved = [json.loads(line) for line in memory.read_text().splitlines()]
    assert len(saved) == 1
    assert saved[0]["parent_metrics"]["torchscore_iptm"] == 0.78
    assert retrieve(first, [{"action": "partial_flow"}]) == []

    second = _runner(tmp_path / "run_b", memory)
    examples = retrieve(second, [{"action": "partial_flow", "scientific_decision_stage": 1}])
    assert len(examples) == 1
    assert examples[0]["action"] == "partial_flow"
    assert examples[0]["parent_metrics"]["rosetta_interface_dg"] == -9.4
    assert retrieve(second, [{"action": "torchopt"}]) == []

    wrong_mode = _runner(tmp_path / "run_c", memory, mode="antibody")
    assert retrieve(wrong_mode, [{"action": "partial_flow"}]) == []


def test_episode_defaults_to_shared_work_dir(tmp_path):
    runner = _runner(tmp_path / "run_a", tmp_path / "custom.jsonl")
    runner.config.autonomous.episodic_memory_path = None
    assert store_path(runner) == tmp_path / "episodic_memory.jsonl"
