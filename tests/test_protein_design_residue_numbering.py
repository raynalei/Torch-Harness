"""Residue coordinates cross the user boundary once, not once per backend."""

import json

import pytest


@pytest.mark.parametrize("index,label", [(0, "B:1:A>G"), (3, "B:4:A>G")])
def test_evolution_mutation_labels_are_one_based(index, label):
    from opendde_harness.plugin.protein_design.analysis.evolution_analysis import _mutations

    candidate = {"metadata": {"mutations": [{"chain_id": "B", "position": index, "from_aa": "A", "to_aa": "G"}]}}
    assert _mutations(candidate) == [label]


def test_standalone_epitope_yaml_parser_uses_same_one_based_boundary():
    from opendde_harness.plugin.protein_design.analysis.epitope_analysis import _parse_yaml_positions

    assert _parse_yaml_positions("1:4") == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        _parse_yaml_positions([0])


def test_epitope_cli_json_displays_one_based_cdr_positions(monkeypatch, capsys):
    from opendde_harness.plugin.protein_design.analysis import epitope_analysis

    monkeypatch.setattr(
        "sys.argv",
        [
            "epitope_analysis",
            "--structure_file",
            "test.pdb",
            "--antibody_chains",
            "B",
            "--antigen_chains",
            "A",
            "--config_yaml",
            "test.yaml",
        ],
    )
    monkeypatch.setattr(epitope_analysis, "load_cdr_regions_from_yaml", lambda *a, **k: {"B": {"CDR1": {0}}})
    monkeypatch.setattr(epitope_analysis, "analyze_epitope", lambda **k: {"cdr_regions": {"B": {"CDR1": [0]}}})
    epitope_analysis.main()
    assert json.loads(capsys.readouterr().out)["cdr_regions"] == {"B": {"CDR1": [1]}}
