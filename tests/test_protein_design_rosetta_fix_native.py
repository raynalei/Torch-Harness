"""Unit checks for the official RosettaFix local adapter; no Rosetta install needed."""

from pathlib import Path

import pytest

from opendde_harness.plugin.protein_design.initial_round.autonomous import rank_score
from opendde_harness.plugin.protein_design.initial_round.config import RosettaFixConfig
from opendde_harness.plugin.protein_design.initial_round.rosetta_fix_native import (
    canonical_pdb,
    configured_xml,
    original_contacts,
    weighted_interface_energy,
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


def test_energy_graph_only_counts_favorable_pairs():
    class Edge:
        def __init__(self, value):
            self.value = value

        def dot(self, weights):
            assert weights == "weights"
            return self.value

    class Graph:
        def find_energy_edge(self, left, right):
            return {(1, 2): Edge(-4.5), (1, 3): Edge(2.0)}.get((left, right))

    class Energies:
        def energy_graph(self):
            return Graph()

    class PdbInfo:
        def chain(self, index):
            return {1: "A", 2: "C", 3: "C"}[index]

        def number(self, index):
            return {1: 5, 2: 6, 3: 7}[index]

    class Pose:
        def energies(self):
            return Energies()

        def pdb_info(self):
            return PdbInfo()

        def total_residue(self):
            return 3

    class Score:
        def __call__(self, pose):
            assert isinstance(pose, Pose)

        def weights(self):
            return "weights"

    energies = weighted_interface_energy(Pose(), Score(), {(5, 6), (5, 7)}, "A", "C")
    assert energies == {"5": -4.5}
