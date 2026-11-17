"""Native evidence uses measured inputs and preserves unavailable domains."""

import httpx
import pytest

from opendde_harness.plugin.protein_design.scientific_agent.backends.analysis import (
    analyze_evolution,
    analyze_structure,
    apply_mutations,
    rank_candidates,
)
from opendde_harness.plugin.protein_design.scientific_agent.backends.developability import (
    assess_developability,
    chemical_modification_risks,
    epimatrix,
    surface_patches,
)
from opendde_harness.plugin.protein_design.scientific_agent.backends.public_lookup import lookup_public


def test_chemical_motifs_include_overlaps_and_one_based_positions():
    result = chemical_modification_risks({"B": "NNSTMWC"})
    assert result["count"] > 0
    assert result["position_semantics"] == "one-based chain sequence"
    assert "not PTM occupancy" in result["interpretation"]


def test_missing_structure_and_epimatrix_are_unknown():
    result = assess_developability({"B": "ACDE"})
    assert result["ptm_risks"]["count"] > 0
    for key in ("surface_patches", "immunogenicity"):
        assert result[key]["available"] is False
        assert result[key]["scores"] is None


@pytest.mark.parametrize("sequence", ["", "ABC", "acde", "A C"])
def test_noncanonical_sequences_rejected(sequence):
    with pytest.raises(ValueError):
        assess_developability({"B": sequence})


def test_epimatrix_is_only_supplied_by_trusted_adapter():
    class Adapter:
        def predict(self, sequences):
            assert sequences == {"B": "ACDE"}
            return {"method": "EpiMatrix", "available": True, "scores": [1.0], "version": "test"}

    assert epimatrix({"B": "ACDE"}, Adapter())["scores"] == [1.0]


def test_epimatrix_rejects_fabricated_method_and_nonfinite_results():
    class Adapter:
        def predict(self, sequences):
            return {"method": "other", "scores": [0]}

    with pytest.raises(ValueError):
        epimatrix({"B": "ACDE"}, Adapter())


@pytest.fixture
def complex_pdb(tmp_path):
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    atoms = struc.AtomArray(6)
    atoms.chain_id = ["B"] * 4 + ["A"] * 2
    atoms.res_id = [1, 1, 2, 2, 10, 10]
    atoms.res_name = ["LYS"] * 4 + ["ALA"] * 2
    atoms.atom_name = ["CA", "CB"] * 3
    atoms.element = ["C"] * 6
    import numpy as np

    atoms.coord = np.array([[0, 0, 0], [0, 1, 0], [3, 0, 0], [3, 1, 0], [0, 3, 0], [0, 4, 0]], dtype=float)
    pdb = PDBFile()
    pdb.set_structure(atoms)
    path = tmp_path / "complex.pdb"
    pdb.write(path)
    return path


def test_real_sasa_surface_and_sequence_mapping(complex_pdb):
    result = surface_patches(complex_pdb, ["B"])
    assert result["sequences"] == {"B": "KK"}
    assert result["scores"]["PSC"] > 0
    assert result["scores"]["PNC"] == 0
    assert "not the official TAP" in result["calibration"]
    assess_developability({"B": "KK"}, structure_path=complex_pdb, binder_chains=["B"])
    with pytest.raises(ValueError, match="do not match"):
        assess_developability({"B": "AA"}, structure_path=complex_pdb, binder_chains=["B"])


def test_contacts_use_target_pdb_ids_without_assuming_confidence(complex_pdb):
    result = analyze_structure(complex_pdb, ["B"], ["A"], hotspots=[{"chain": "A", "residue_id": 10}])
    assert result["epitope_hit"] is True
    assert result["confidence"]["available"] is False
    with pytest.raises(ValueError, match="not mapped"):
        analyze_structure(complex_pdb, ["B"], ["A"], hotspots=[{"chain": "A", "residue_id": 1}])


def test_mutations_enforce_fixed_positions_and_actual_changes():
    assert apply_mutations({"B": "ACD"}, [{"chain": "B", "position": 2, "amino_acid": "K"}])["sequences"] == {
        "B": "AKD"
    }
    with pytest.raises(ValueError, match="fixed"):
        apply_mutations({"B": "ACD"}, [{"chain": "B", "position": 2, "amino_acid": "K"}], {"B": [2]})


def test_lineage_uses_real_parents_and_detects_cycles():
    result = analyze_evolution([{"candidate_id": "child", "parent_id": "absent"}])
    assert result["missing_parents"] == ["absent"]
    with pytest.raises(ValueError, match="cyclic"):
        analyze_evolution([{"candidate_id": "a", "parent_id": "b"}, {"candidate_id": "b", "parent_id": "a"}])


def test_ranking_keeps_missing_metrics_unranked():
    result = rank_candidates([{"candidate_id": "missing"}])
    assert result["ranking"] == []
    assert result["unranked_missing_metrics"] == ["missing"]


@pytest.mark.parametrize(
    "source,identifier,host",
    [
        ("pdb", "1abc", "data.rcsb.org"),
        ("uniprot", "P12345", "rest.uniprot.org"),
        ("pubmed", "123", "eutils.ncbi.nlm.nih.gov"),
    ],
)
def test_public_lookup_only_calls_fixed_source_hosts(source, identifier, host):
    def respond(request):
        assert request.url.host == host
        return httpx.Response(200, json={"id": identifier})

    assert lookup_public(source, identifier, transport=httpx.MockTransport(respond))["data"] == {"id": identifier}
    with pytest.raises(ValueError):
        lookup_public(source, "../../secret")


def test_residue_hydropathy_preserves_pdb_numbering_and_buried_residues(complex_pdb, monkeypatch):
    import biotite.structure as struc
    import numpy as np
    from biotite.structure.io.pdb import PDBFile

    pdb = PDBFile.read(complex_pdb)
    atoms = pdb.get_structure(model=1)
    atoms.res_id[:4] = [42, 42, 57, 57]
    atoms.ins_code[:4] = ["A", "A", "", ""]
    atoms.res_name[:2] = "LEU"
    pdb = PDBFile()
    pdb.set_structure(atoms)
    pdb.write(complex_pdb)
    monkeypatch.setattr(struc, "sasa", lambda atoms, **kwargs: np.array([30.0, 30.0, 0.0, 0.0]))
    output = assess_developability({"B": "LK"}, structure_path=complex_pdb, binder_chains=["B"])
    surface = output["surface_patches"]
    hydrophobic, hydrophilic = surface["residues"]
    assert (
        hydrophobic["chain"],
        hydrophobic["residue_id"],
        hydrophobic["insertion_code"],
        hydrophobic["sequence_position"],
    ) == ("B", 42, "A", 1)
    assert hydrophobic["hydropathy"] == 3.8
    assert hydrophobic["hydropathy_class"] == "hydrophobic"
    assert hydrophobic["sasa_angstrom2"] == 60.0
    assert hydrophobic["relative_asa"] == pytest.approx(60.0 / 201)
    assert hydrophobic["surface_exposed"] is True
    assert hydrophilic["sequence_position"] == 2
    assert hydrophilic["hydropathy"] == -3.9
    assert hydrophilic["hydropathy_class"] == "hydrophilic"
    assert hydrophilic["charge_class"] == "positive"
    assert hydrophilic["surface_exposed"] is False
    assert surface["scores"] == {"PSH": 0, "PNC": 0, "PSC": 0}
