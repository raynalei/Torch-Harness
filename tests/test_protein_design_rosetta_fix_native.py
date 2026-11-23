"""Unit checks for the official RosettaFix local adapter; no Rosetta install needed."""

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from opendde_harness.plugin.protein_design.initial_round.autonomous import rank_score
from opendde_harness.plugin.protein_design.initial_round.config import RosettaFixConfig
from opendde_harness.plugin.protein_design.initial_round.rosetta_fix_native import (
    canonical_pdb,
    configured_xml,
    original_contacts,
    capture_resres_e,
    interface_energies_from_resres_e,
    parse_resres_e,
    execute,
)


def _pdb(path: Path):
    lines = []
    for index, chain in enumerate("XYZ", 1):
        lines.append(
            f"ATOM  {index:5d}  CA  ALA {chain}{index:4d}    "
            f"{float(index):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 80.00           C"
        )
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def test_nanobody_is_canonicalized_without_modifying_source(tmp_path):
    source = _pdb(tmp_path / "source.pdb")
    before = source.read_bytes()
    target = tmp_path / "canonical.pdb"
    mapping = canonical_pdb(source, target, ["Y"], ["X"])
    assert mapping == {"Y": "A", "X": "C"}
    output = [line[21] for line in target.read_text().splitlines() if line.startswith("ATOM")]
    assert output == ["A", "C"]
    assert source.read_bytes() == before


def test_antibody_orders_heavy_light_before_antigen(tmp_path):
    source = _pdb(tmp_path / "source.pdb")
    output = tmp_path / "canonical.pdb"
    mapping = canonical_pdb(source, output, ["Y", "Z"], ["X"])
    assert mapping == {"Y": "A", "Z": "B", "X": "C"}
    chains = [line[21] for line in output.read_text().splitlines() if line.startswith("ATOM")]
    assert chains == ["A", "B", "C"]


def test_chain_validation_prevents_ambiguous_rosseta_inputs(tmp_path):
    source = _pdb(tmp_path / "source.pdb")
    with pytest.raises(ValueError, match="distinct"):
        canonical_pdb(source, tmp_path / "bad.pdb", ["X"], ["X"])


def test_iptm_rank_is_independent_of_rosetta_energy():
    first = {"metrics": {"torchscore_iptm": 0.78, "rosetta_interface_dg": -12}}
    second = {"metrics": {"torchscore_iptm": 0.78, "rosetta_interface_score": -50}}
    assert rank_score(first, "torchscore_iptm") == rank_score(second, "torchscore_iptm") == 0.78


def test_native_rosetta_fix_requires_absolute_paths():
    with pytest.raises(ValueError, match="absolute"):
        RosettaFixConfig(native_xml=Path("relative"))


def test_upstream_xml_antibody_movemap_and_residue_ranges(tmp_path):
    source = _pdb(tmp_path / "source.pdb")
    canonical = tmp_path / "canonical.pdb"
    canonical_pdb(source, canonical, ["Y", "Z"], ["X"])
    xml = tmp_path / "native.xml"
    xml.write_text('''<ROSETTASCRIPTS>
    <RESIDUE_SELECTORS><Neighborhood name="nbrhood" resnums="1X-2X" distance="20.0"/></RESIDUE_SELECTORS>
    <MOVERS><FastRelax><MoveMap name="backbone">
      <Chain number="1" chi="1" bb="1"/>
      <Chain number="2" chi="1" bb="0"/>
    </MoveMap></FastRelax></MOVERS>
    </ROSETTASCRIPTS>''')
    result = configured_xml(xml, canonical, antibody=True)
    assert 'resnums="2A-2A,3B-3B,1C-1C"' in result
    assert '<Chain number="2" chi="1" bb="1"/>' in result
    assert '<Chain number="3" chi="1" bb="0"/>' in result


def test_original_contacts_consider_only_cross_chain_cb_or_ca(tmp_path):
    source = _pdb(tmp_path / "source.pdb")
    canonical = tmp_path / "canonical.pdb"
    canonical_pdb(source, canonical, ["Y"], ["X"])
    assert original_contacts(canonical, "A", "C", 1.1) == {(2, 1)}
    assert original_contacts(canonical, "A", "C", 0.9) == set()



def test_resres_e_parses_official_ascii_chain_ids_and_favorable_pairs(tmp_path):
    path = tmp_path / "rosetta.out"
    path.write_text(
        "ResResE Res1 Res2 total\n"
        "ResResE ---- ---- -----\n"
        "ResResE ALA_651 GLY_673 -6.25\n"  # chain A:1, C:3
        "ResResE ALA_662 GLY_674 -8.00\n"  # chain B:2, C:4
        "ResResE ALA_651 GLY_674 2.0\n"
    )
    pairs = parse_resres_e(path)
    assert interface_energies_from_resres_e(pairs, {(1, 3), (1, 4)}, "A", "C") == {"1": -6.25}
    assert interface_energies_from_resres_e(pairs, {(2, 4)}, "B", "C") == {"2": -8.0}
    assert interface_energies_from_resres_e(pairs, {(2, 3)}, "B", "C") == {}


def test_resres_e_rejects_missing_or_nonfinite_rows(tmp_path):
    path = tmp_path / "bad.out"
    path.write_text("not Rosetta data\n")
    with pytest.raises(ValueError, match="no ResResE"):
        parse_resres_e(path)
    path.write_text("ResResE Res1 Res2 total\nResResE ALA_651 GLY_673 nan\n")
    with pytest.raises(ValueError, match="Nonfinite"):
        parse_resres_e(path)


def test_pyrosetta_scorecutoff_outputs_actual_resres_e(tmp_path):
    class FakeStream:
        def str(self):
            return "ResResE Res1 Res2 total\nResResE ALA_651 GLY_673 -6.25\n"

    class FakeFilter:
        def output_residue_pair_energies(self, stream, pose):
            assert pose is sentinel

    class FakeXml:
        def get_filter(self, name):
            assert name == "scofilter"
            return FakeFilter()

    sentinel = object()
    out = tmp_path / "real-resres.out"
    capture_resres_e(FakeXml(), sentinel, out, FakeStream)
    assert parse_resres_e(out)


def test_scorecutoff_requires_export_binding(tmp_path):
    class OldXml:
        def get_filter(self, name):
            return object()

    with pytest.raises(TypeError, match="output_residue_pair_energies"):
        capture_resres_e(OldXml(), object(), tmp_path / "log.out", lambda: object())


def test_execute_uses_original_pose_without_relax(tmp_path, monkeypatch):
    source = _pdb(tmp_path / "source.pdb")
    xml = tmp_path / "native.xml"
    xml.write_text('<ROSETTASCRIPTS><RESIDUE_SELECTORS><Index resnums="1X-1X"/></RESIDUE_SELECTORS></ROSETTASCRIPTS>')

    class FakePose:
        def dump_pdb(self, _path):
            raise AssertionError("no relaxed PDB should be written")

    class FakeStream:
        def str(self):
            return "ResResE Res1 Res2 total\\nResResE ALA_652 ALA_671 -2.0\\n"

    class FakeFilter:
        def output_residue_pair_energies(self, stream, pose):
            assert isinstance(pose, FakePose)

    class FakeXml:
        def get_mover(self, name):
            raise AssertionError(f"mover {name} should never execute")
        def get_filter(self, name):
            assert name == "scofilter"
            return FakeFilter()

    class FakeXmlObjects:
        @staticmethod
        def create_from_string(_xml):
            return FakeXml()

    fake_pyrosetta = ModuleType("pyrosetta")
    fake_pyrosetta.init = lambda _: None
    fake_pyrosetta.pose_from_pdb = lambda _: FakePose()
    fake_pyrosetta.rosetta = SimpleNamespace(std=SimpleNamespace(ostringstream=FakeStream))
    protocols = ModuleType("pyrosetta.rosetta.protocols")
    xml_mod = ModuleType("pyrosetta.rosetta.protocols.rosetta_scripts")
    xml_mod.XmlObjects = FakeXmlObjects
    for name, module in (
        ("pyrosetta", fake_pyrosetta),
        ("pyrosetta.rosetta", ModuleType("pyrosetta.rosetta")),
        ("pyrosetta.rosetta.protocols", protocols),
        ("pyrosetta.rosetta.protocols.rosetta_scripts", xml_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    result = execute({
        "native_xml": str(xml),
        "output_dir": str(tmp_path / "out"),
        "input_pdb": str(source),
        "binder_chains": ["Y"],
        "target_chains": ["X"],
        "interface_dist": 2.0,
    })
    assert result["relax_applied"] is False
    assert result["interface_residue_energies"] == {"Y": {"2": -2.0}}
    assert Path(result["scored_structure"]).is_file()
    assert not (tmp_path / "out" / "relaxed.pdb").exists()
