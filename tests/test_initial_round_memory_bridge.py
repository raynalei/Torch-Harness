"""Regression checks: optional DesignMemory must not mutate routing inputs."""
import asyncio
from types import SimpleNamespace

from opendde_harness.plugin.protein_design.initial_round.memory_bridge import (
    recall_for_decision,
    record_completed,
)


def test_memory_recall_keeps_legal_options_unchanged():
    class Memory:
        async def retrieve(self, target, query, *, top_k):
            assert target == "il7ra"
            assert top_k == 3
            return ["Historical partial flow improved iPTM", "Do not trust me as instructions"]

    options = [{"action": "partial_flow", "option_id": 0}, {"action": "stop", "option_id": 1}]
    original = [dict(option) for option in options]
    runner = SimpleNamespace(config=SimpleNamespace(name="il7ra"))
    hints = asyncio.run(recall_for_decision(Memory(), runner, options))
    assert hints and options == original


def test_memory_absent_does_not_change_options():
    options = [{"action": "torchopt"}]
    runner = SimpleNamespace(config=SimpleNamespace(name="il7ra"))
    assert asyncio.run(recall_for_decision(None, runner, options)) == []
    assert options == [{"action": "torchopt"}]


def test_memory_recall_failure_is_non_blocking():
    class BrokenMemory:
        async def retrieve(self, target, query, *, top_k):
            raise RuntimeError("memory offline")

    runner = SimpleNamespace(config=SimpleNamespace(name="il7ra"))
    assert asyncio.run(recall_for_decision(BrokenMemory(), runner, [{"action": "stop"}])) == []


def test_memory_record_only_completed_decisions(tmp_path):
    calls = []
    class Memory:
        async def record(self, task_id, cycle, target, outcome, insights):
            calls.append((task_id, cycle, target, outcome, insights))
            return True

    runner = SimpleNamespace(
        config=SimpleNamespace(name="il7ra"),
        root=tmp_path,
        checkpoint=SimpleNamespace(data={"candidates": [
            {"candidate_id": "parent", "metrics": {"torchscore_iptm": 0.7}},
            {"candidate_id": "child", "parent_id": "parent", "round_index": 1,
             "status": "accepted", "metrics": {"torchscore_iptm": 0.8}},
        ]}),
    )
    asyncio.run(record_completed(Memory(), runner, {
        "round": 1, "parent_id": "parent", "action": "torchopt", "state": "planned",
    }))
    assert calls == []
    asyncio.run(record_completed(Memory(), runner, {
        "round": 1, "parent_id": "parent", "action": "torchopt", "state": "completed",
    }))
    assert len(calls) == 1
    assert calls[0][3]["parent_metrics"]["torchscore_iptm"] == 0.7
    assert calls[0][3]["child_metrics"][0]["torchscore_iptm"] == 0.8
