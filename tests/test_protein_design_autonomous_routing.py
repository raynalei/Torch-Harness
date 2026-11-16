"""Staged score routing, interface fixed points and checkpoint recovery."""

from __future__ import annotations

import copy
import json

import httpx
import pytest

from opendde_harness.plugin.protein_design.initial_round.artifacts import structure
from opendde_harness.plugin.protein_design.initial_round.config import Chains
from opendde_harness.plugin.protein_design.initial_round.interface_energy import (
    interface_energies,
    partial_positions,
    residue_pairs,
)
from opendde_harness.plugin.protein_design.initial_round.policy import rule_actions
from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner
from tests._slurm_api import mock_slurm_apis  # noqa: F401
from tests.test_protein_design_initial_round import autonomous_config, fake_services, pdb, round_config  # noqa: F401

pytestmark = [pytest.mark.in_memory_transport, pytest.mark.usefixtures("mock_slurm_apis")]


@pytest.mark.parametrize(
    "stage,iptm,ptm,fixed,expected",
    [
        ("initial", 0.4999, 0.99, None, "terminate"),
        ("initial", 0.5, 0.99, None, "score_interface"),
        ("initial", 0.8, 0.99, None, "score_interface"),
        ("initial", 0.99, 0.8, None, "score_interface"),
        ("initial", 0.8001, 0.8001, None, "refold"),
        ("initial", 0.7, 0.7, {"B": [1]}, "partial_flow"),
        ("initial", 0.7, 0.7, {"B": []}, "terminate"),
        ("partial_flow", 0.4, 0.7, None, "torchopt"),
        ("partial_flow", 0.8, 0.9, None, "torchopt"),
        ("partial_flow", 0.9, 0.9, None, "refold"),
        ("torchopt", 0.9, 0.8, None, "terminate"),
        ("torchopt", 0.4, 0.9, None, "terminate"),
        ("torchopt", 0.9, 0.9, None, "refold"),
    ],
)
def test_stage_and_strict_thresholds_control_routing(stage, iptm, ptm, fixed, expected):
    for hotspot in (False, None, True):
        assert rule_actions(
            {"torchscore_iptm": iptm, "torchscore_ptm": ptm},
            hotspot,
            stage=stage,
            fixed_positions=fixed,
        ) == [expected]


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True, -0.01, 1.01])
def test_invalid_scientific_metrics_never_route(bad):
    assert not rule_actions({"torchscore_iptm": bad, "torchscore_ptm": 0.9})
    assert not rule_actions({"torchscore_iptm": 0.9, "torchscore_ptm": bad})


def test_supplied_energy_algorithm_sums_only_negative_contact_pairs(tmp_path):
    model = pdb(tmp_path / "model.pdb")
    chains = Chains(target=["A"], binder=["B"], refold_map={"A": "A", "B": "B"})
    log = tmp_path / "rosetta.out"
    log.write_text(
        "irrelevant log line\nResResE Res1 Res2 total\nResResE ---- ---- -----\n"
        "ResResE ALA_661 GLY_651 -3\n"
        "ResResE GLY_652 ALA_661 -2.5\n"  # Reverse orientation, same binder residue.
        "ResResE ALA_661 GLY_653 7\n"  # Positive energies are excluded.
        "ResResE ALA_661 GLY_656 -100\n"  # Outside 12 Angstrom contact radius.
        "ResResE ALA_662 GLY_652 -5\n"
        "ResResE ALA_661 ALA_662 -50\n"  # Intrachain energies are excluded.
    )
    energies = interface_energies(log, model, chains)
    assert energies == {"B": {"1": -5.5, "2": -5.0}}
    fixed = {chain: [int(n) for n, value in values.items() if value < -5] for chain, values in energies.items()}
    positions = partial_positions(structure(model), chains, fixed)
    assert positions["fixed_positions"] == "B1"
    assert positions["cdr_position"] == "B2-6"
    assert positions["mutable_positions"] == {"B": [2, 3, 4, 5, 6]}


@pytest.mark.parametrize(
    "body",
    [
        "",
        "SCORE: total_score -5",
        "ResResE Res1 Res2 total\nResResE A_661 G_651 nan",
        "ResResE Res1 Res2 total\nResResE A_bad G_651 -6",
    ],
)
def test_missing_or_malformed_residue_logs_fail_explicitly(tmp_path, body):
    log = tmp_path / "bad.out"
    log.write_text(body)
    with pytest.raises(ValueError):
        residue_pairs(log)


@pytest.mark.parametrize(
    "iptm,ptm,folds,energies,terminated", [(0.49, 0.99, 0, 0, True), (0.9, 0.9, 2, 0, False), (0.5, 0.8, 0, 2, False)]
)
async def test_initial_stage_submits_only_the_required_services(
    round_config, fake_services, iptm, ptm, folds, energies, terminated
):
    config = autonomous_config(round_config, max_samples=2)
    base, calls = fake_services

    def handler(request):
        response = base.handler(request)
        if request.url.host == "torchscore.test" and request.url.path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": [{"iptm": iptm, "ptm": ptm}]})
        return response

    runner = InitialRoundRunner(config, transport=httpx.MockTransport(handler))
    await runner.run()
    assert calls[("torchfold", "POST")] == folds
    assert calls[("rosetta", "POST")] == energies
    assert calls[("torchscore", "POST")] == 2
    for candidate in runner.checkpoint.data["candidates"]:
        assert (candidate["status"] == "terminated") is terminated
        if energies:
            assert candidate["interface_fixed_positions"] == {"B": [1]}
            assert candidate["metrics"]["rosetta_interface_score"] == -6
        else:
            assert "rosetta_interface_score" not in candidate["metrics"]


async def test_partial_flow_is_rescored_and_directly_refolded_without_second_rosetta(round_config, fake_services):
    config = autonomous_config(round_config, max_samples=2, max_rounds=1)
    config.generators.pop("torchcraft")
    base, calls = fake_services

    def handler(request):
        response = base.handler(request)
        if request.url.host == "torchscore.test" and request.url.path.endswith("/metrics"):
            return httpx.Response(
                200, json={"metrics": [{"iptm": 0.7 if calls[("torchscore", "POST")] == 1 else 0.9, "ptm": 0.9}]}
            )
        if request.url.host == "ppiflow.test" and request.method == "POST":
            body = json.loads(request.content)
            if body["mode"].startswith("partial_"):
                assert body["fixed_positions"] == "B1"
                assert body["cdr_position"] == "B2-6"
        return response

    transport = httpx.MockTransport(handler)
    runner = InitialRoundRunner(config, transport=transport)
    result = await runner.run()
    parent, child = runner.checkpoint.data["candidates"]
    assert result["accepted"] == 1
    assert child["parent_id"] == parent["candidate_id"]
    assert child["diagnostics"]["stage"] == "partial_flow"
    assert calls[("torchscore", "POST")] == 2
    assert calls[("rosetta", "POST")] == calls[("torchfold", "POST")] == 1
    before = copy.deepcopy(calls)
    assert await InitialRoundRunner(config, transport=transport).run() == result
    assert {k: v for k, v in calls.items() if k[1] == "POST"} == {k: v for k, v in before.items() if k[1] == "POST"}


@pytest.mark.parametrize("energy,has_fixed", [(-5.0, False), (-5.0001, True)])
async def test_runner_applies_strict_residue_energy_threshold(round_config, fake_services, energy, has_fixed):
    config = autonomous_config(round_config, max_samples=2)
    base, calls = fake_services

    def handler(request):
        response = base.handler(request)
        if request.url.host == "torchscore.test" and request.url.path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": [{"iptm": 0.6, "ptm": 0.9}]})
        if request.url.host == "rosetta.test" and request.url.path.endswith("/results"):
            log = config.work_dir / response.json()["files"][0].removeprefix("/shared/")
            log.write_text(f"ResResE Res1 Res2 total\nResResE ALA_661 GLY_651 {energy}\n")
        return response

    runner = InitialRoundRunner(config, transport=httpx.MockTransport(handler))
    await runner.run()
    assert all(
        c["interface_fixed_positions"] == {"B": [1] if has_fixed else []} for c in runner.checkpoint.data["candidates"]
    )
    assert all((c["status"] == "terminated") is not has_fixed for c in runner.checkpoint.data["candidates"])
    assert calls[("torchfold", "POST")] == 0


async def test_partial_output_cannot_mutate_an_energy_selected_fixed_point(round_config, fake_services):
    config = autonomous_config(round_config, max_samples=2, max_rounds=1)
    config.generators.pop("torchcraft")
    base, calls = fake_services

    def handler(request):
        response = base.handler(request)
        if request.url.host == "torchscore.test" and request.url.path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": [{"iptm": 0.6, "ptm": 0.9}]})
        if (
            request.url.host == "ppiflow.test"
            and request.method == "POST"
            and json.loads(request.content)["mode"].startswith("partial_")
        ):
            pdb(config.work_dir / "ppiflow/generated.pdb", binder="VAAAAA")
        return response

    runner = InitialRoundRunner(config, transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="changed residues outside the selected region"):
        await runner.run()
    assert calls[("torchscore", "POST")] == 1  # The invalid child never reaches scoring.
    assert len(runner.checkpoint.data["candidates"]) == 1


async def test_retained_rosetta_log_cannot_change_during_resume(round_config, fake_services):
    config = autonomous_config(round_config, max_samples=2)
    base, _ = fake_services

    def handler(request):
        response = base.handler(request)
        if request.url.host == "torchscore.test" and request.url.path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": [{"iptm": 0.7, "ptm": 0.9}]})
        return response

    transport = httpx.MockTransport(handler)
    runner = InitialRoundRunner(config, transport=transport)
    await runner.run()
    candidate = runner.checkpoint.data["candidates"][0]
    runner.files.local(candidate["rosetta_energy_log"]).write_text("changed evidence")
    with pytest.raises(ValueError, match="residue-energy log changed"):
        await InitialRoundRunner(config, transport=transport).run()


def test_multiple_binder_chains_keep_separate_energy_sums_and_real_numbering(tmp_path):
    from biotite.structure.io import save_structure

    model = pdb(tmp_path / "complex.pdb", light="L")
    atoms = structure(model)
    atoms.res_id[atoms.chain_id == "B"] += 100
    atoms.res_id[atoms.chain_id == "L"] += 200
    save_structure(str(model), atoms)
    chains = Chains(target=["A"], binder=["B", "L"], refold_map={"A": "X", "B": "Y", "L": "Z"})
    log = tmp_path / "pairs.out"
    log.write_text("ResResE Res1 Res2 total\nResResE ALA_66101 GLY_651 -6\nResResE VAL_76201 GLY_651 -7\n")
    assert interface_energies(log, model, chains) == {"B": {"101": -6.0}, "L": {"201": -7.0}}
    positions = partial_positions(atoms, chains, {"B": [101], "L": [201]})
    assert positions["fixed_positions"] == "B101,L201"
    assert positions["cdr_position"] == "B102-106,L202-206"


def test_missing_calpha_cannot_shift_partial_flow_fixed_indices(tmp_path):
    model = pdb(tmp_path / "candidate.pdb")
    atoms = structure(model)
    atoms = atoms[~((atoms.chain_id == "B") & (atoms.res_id == 2) & (atoms.atom_name == "CA"))]
    chains = Chains(target=["A"], binder=["B"], refold_map={"A": "A", "B": "B"})
    with pytest.raises(ValueError, match="complete C-alpha"):
        partial_positions(atoms, chains, {"B": [1]})
