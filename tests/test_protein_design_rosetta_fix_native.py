"""Unit checks for the official RosettaFix local adapter; no Rosetta install needed."""

from pathlib import Path

import pytest

from opendde_harness.plugin.protein_design.initial_round.autonomous import rank_score
from opendde_harness.plugin.protein_design.initial_round.config import RosettaFixConfig
from opendde_harness.plugin.protein_design.initial_round.rosetta_fix_native import canonical_pdb


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
