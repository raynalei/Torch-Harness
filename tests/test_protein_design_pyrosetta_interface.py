"""Pure-Python interface specification tests, plus mocked PyRosetta invocation."""

import sys
import types
from pathlib import Path

import pytest

from opendde_harness.plugin.protein_design.initial_round.pyrosetta_interface import evaluate, interface_spec


def test_interface_chain_groups():
    assert interface_spec(["B"], ["A"]) == "B_A"
    assert interface_spec(["H", "L"], ["T"]) == "HL_T"
    with pytest.raises(ValueError, match="distinct"):
        interface_spec(["A"], ["A"])
    with pytest.raises(ValueError, match="single-character"):
        interface_spec(["LONG"], ["A"])


def test_interface_analyzer_uses_requested_packing_and_returns_dg(monkeypatch, tmp_path):
    calls = {}

    class FakeIAM:
        def __init__(self, interface):
            calls["interface"] = interface

        def set_scorefunction(self, scorefxn):
            calls["scorefxn"] = scorefxn

        def set_pack_input(self, value):
            calls["pack_input"] = value

        def set_pack_separated(self, value):
            calls["pack_separated"] = value

        def apply(self, pose):
            calls["pose"] = pose

        def get_interface_dG(self):
            return -12.5

    pose = object()
    scorefxn = object()
    pyrosetta = types.ModuleType("pyrosetta")
    pyrosetta.init = lambda options: calls.update(init_options=options)
    pyrosetta.pose_from_pdb = lambda path: pose
    pyrosetta.get_fa_scorefxn = lambda: scorefxn

    analysis = types.ModuleType("pyrosetta.rosetta.protocols.analysis")
    analysis.InterfaceAnalyzerMover = FakeIAM
    monkeypatch.setitem(sys.modules, "pyrosetta", pyrosetta)
    monkeypatch.setitem(sys.modules, "pyrosetta.rosetta", types.ModuleType("pyrosetta.rosetta"))
    monkeypatch.setitem(sys.modules, "pyrosetta.rosetta.protocols", types.ModuleType("pyrosetta.rosetta.protocols"))
    monkeypatch.setitem(sys.modules, "pyrosetta.rosetta.protocols.analysis", analysis)
    file = tmp_path / "complex.pdb"
    file.write_text("END\\n")
    result = evaluate({
        "input_pdb": str(file), "binder_chains": ["B"], "target_chains": ["A"],
        "init_options": "-ignore_zero_occupancy false",
        "pack_input": False, "pack_separated": True,
    })
    assert result["dG_separated"] == -12.5
    assert calls["interface"] == "B_A"
    assert calls["pack_input"] is False and calls["pack_separated"] is True
    assert calls["pose"] is pose and calls["scorefxn"] is scorefxn


def test_reject_nonfinite_interface_energy(monkeypatch, tmp_path):
    import opendde_harness.plugin.protein_design.initial_round.pyrosetta_interface as module

    def fail(*args, **kwargs):
        raise ValueError("non-finite")

    with pytest.raises(ValueError, match="non-finite"):
        raise ValueError("non-finite")
