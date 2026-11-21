"""Pure-Python interface specification tests, plus mocked PyRosetta invocation."""

import sys
import types
from types import SimpleNamespace
from pathlib import Path

import pytest

from opendde_harness.plugin.protein_design.initial_round.pyrosetta_interface import (
    evaluate,
    interface_spec,
    make_interface_analyzer,
)


def test_interface_chain_groups():
    assert interface_spec(["B"], ["A"]) == "B_A"
    assert interface_spec(["H", "L"], ["T"]) == "HL_T"
    with pytest.raises(ValueError, match="distinct"):
        interface_spec(["A"], ["A"])
    with pytest.raises(ValueError, match="single-character"):
        interface_spec(["LONG"], ["A"])


def test_interface_analyzer_uses_requested_packing_and_returns_dg(monkeypatch, tmp_path):
    calls = {}

    class FakePartners:
        @staticmethod
        def docking_partners_from_string(spec):
            calls["raw_spec"] = spec
            return ("docking_partners", spec)

    class FakeIAM:
        def __init__(self, *args):
            if args:
                raise TypeError("PyRosetta 2026.29 rejects string constructor")
            calls["constructed_without_args"] = True

        def set_interface(self, interface):
            if isinstance(interface, str):
                raise TypeError("PyRosetta 2026.29 requires DockingPartners")
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
    core = types.ModuleType("pyrosetta.rosetta.core")
    pose_module = types.ModuleType("pyrosetta.rosetta.core.pose")
    pose_module.DockingPartners = FakePartners
    core.pose = pose_module
    rosetta_module = types.ModuleType("pyrosetta.rosetta")
    rosetta_module.core = core
    pyrosetta.rosetta = rosetta_module
    monkeypatch.setitem(sys.modules, "pyrosetta.rosetta", rosetta_module)
    monkeypatch.setitem(sys.modules, "pyrosetta.rosetta.core", core)
    monkeypatch.setitem(sys.modules, "pyrosetta.rosetta.core.pose", pose_module)
    monkeypatch.setitem(sys.modules, "pyrosetta.rosetta.protocols", types.ModuleType("pyrosetta.rosetta.protocols"))
    monkeypatch.setitem(sys.modules, "pyrosetta.rosetta.protocols.analysis", analysis)
    file = tmp_path / "complex.pdb"
    file.write_text("END\n")
    result = evaluate({
        "input_pdb": str(file), "binder_chains": ["B"], "target_chains": ["A"],
        "init_options": "-ignore_zero_occupancy false",
        "pack_input": False, "pack_separated": True,
    })
    assert result["dG_separated"] == -12.5
    assert calls["constructed_without_args"] is True
    assert calls["raw_spec"] == "B_A"
    assert calls["interface"] == ("docking_partners", "B_A")
    assert calls["pack_input"] is False and calls["pack_separated"] is True
    assert calls["pose"] is pose and calls["scorefxn"] is scorefxn




def test_recent_pyrosetta_string_setter_fallback(monkeypatch):
    calls = {}

    class IAM:
        def set_interface(self, value):
            calls["interface"] = value

    class DockingPartners:
        @staticmethod
        def docking_partners_from_string(spec):
            raise TypeError("unsupported DockingPartners conversion")

    analysis = types.ModuleType("pyrosetta.rosetta.protocols.analysis")
    analysis.InterfaceAnalyzerMover = IAM
    monkeypatch.setitem(sys.modules, "pyrosetta.rosetta.protocols.analysis", analysis)
    pyrosetta = SimpleNamespace(rosetta=SimpleNamespace(core=SimpleNamespace(pose=SimpleNamespace(DockingPartners=DockingPartners))))
    iam = make_interface_analyzer(pyrosetta, "B_A")
    assert isinstance(iam, IAM)
    assert calls["interface"] == "B_A"


def test_interface_analyzer_refuses_implicit_jump_selection(monkeypatch):
    class IAM:
        def set_interface(self, value):
            raise TypeError("unsupported")

    analysis = types.ModuleType("pyrosetta.rosetta.protocols.analysis")
    analysis.InterfaceAnalyzerMover = IAM
    monkeypatch.setitem(sys.modules, "pyrosetta.rosetta.protocols.analysis", analysis)
    pyrosetta = SimpleNamespace(rosetta=SimpleNamespace(core=SimpleNamespace(pose=SimpleNamespace())))
    with pytest.raises(RuntimeError, match="cannot configure interface"):
        make_interface_analyzer(pyrosetta, "A_B")
