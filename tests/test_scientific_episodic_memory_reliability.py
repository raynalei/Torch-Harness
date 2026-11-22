"""Regression coverage for the advisory scientific episodic-memory lane."""
from pathlib import Path
from types import SimpleNamespace

from opendde_harness.plugin.protein_design.scientific_agent import episodic_memory


def runner(tmp_path):
    generator = SimpleNamespace(scientific_mode="nanobody")
    return SimpleNamespace(
        root=tmp_path / "run1",
        config=SimpleNamespace(
            work_dir=tmp_path,
            autonomous=SimpleNamespace(episodic_memory_path=None),
            generators={"ppiflow": generator},
        ),
        checkpoint=SimpleNamespace(data={
            "candidates": [
                {"candidate_id": "p1", "source": "ppiflow", "metrics": {"torchscore_iptm": 0.6}},
                {"candidate_id": "c1", "parent_id": "p1", "round_index": 1,
                 "source": "ppiflow", "status": "accepted", "metrics": {"torchscore_iptm": 0.75}},
            ]
        }),
    )


def test_round_trip_and_duplicate_suppression(tmp_path):
    r = runner(tmp_path)
    decision = {"state": "completed", "round": 1, "parent_id": "p1",
                "action": "partial_flow", "policy": "planner"}
    episodic_memory.record_decision(r, decision)
    episodic_memory.record_decision(r, decision)
    path = episodic_memory.store_path(r)
    assert len(path.read_text().splitlines()) == 1
    # Cross-run history only, so current run's completed episode is excluded.
    assert episodic_memory.retrieve(r, [{"action": "partial_flow"}]) == []
    r.root = tmp_path / "run2"
    hits = episodic_memory.retrieve(r, [{"action": "partial_flow"}])
    assert len(hits) == 1
    assert hits[0]["outcome"]["accepted"] == 1


def test_corrupt_entries_do_not_break_recall(tmp_path):
    r = runner(tmp_path)
    path = episodic_memory.store_path(r)
    path.write_text('bad-json\n{"outcome": null}\n{"outcome": []}\n')
    assert episodic_memory.retrieve(r, [{"action": "partial_flow"}]) == []


def test_memory_read_failure_does_not_break_routing(monkeypatch, tmp_path):
    r = runner(tmp_path)
    path = episodic_memory.store_path(r)
    path.write_text("x")
    original = Path.read_text

    def broken_read(self, *args, **kwargs):
        if self == path:
            raise PermissionError("memory temporarily unavailable")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", broken_read)
    assert episodic_memory.retrieve(r, [{"action": "partial_flow"}]) == []


def test_memory_write_failure_does_not_fail_completed_decision(monkeypatch, tmp_path):
    r = runner(tmp_path)
    path = episodic_memory.store_path(r)

    def broken_mkdir(self, *args, **kwargs):
        if self == path.parent:
            raise PermissionError("readonly shared mount")
        return None

    monkeypatch.setattr(Path, "mkdir", broken_mkdir)
    episodic_memory.record_decision(r, {
        "state": "completed", "round": 1, "parent_id": "p1",
        "action": "partial_flow", "policy": "planner",
    })
    assert not path.exists()
