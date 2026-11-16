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
    config = autonomous_config(round_config)
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
    assert calls[("torchscore", "POST")] == (4 if energies else 2)
    for candidate in [c for c in runner.checkpoint.data["candidates"] if not c.get("parent_id")]:
        assert (candidate["status"] == "terminated") is terminated
        if energies:
            assert candidate["interface_fixed_positions"] == {"B": [1]}
            assert candidate["metrics"]["rosetta_interface_score"] == -6
        else:
            assert "rosetta_interface_score" not in candidate["metrics"]


async def test_partial_flow_is_rescored_and_directly_refolded_without_second_rosetta(round_config, fake_services):
    config = autonomous_config(round_config)
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
    config = autonomous_config(round_config)
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
        c["interface_fixed_positions"] == {"B": [1] if has_fixed else []}
        for c in runner.checkpoint.data["candidates"]
        if not c.get("parent_id")
    )
    assert all(
        (c["status"] == "terminated") is not has_fixed
        for c in runner.checkpoint.data["candidates"]
        if not c.get("parent_id")
    )
    assert calls[("torchfold", "POST")] == 0


async def test_partial_output_cannot_mutate_an_energy_selected_fixed_point(round_config, fake_services):
    config = autonomous_config(round_config)
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
    config = autonomous_config(round_config)
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


@pytest.mark.parametrize("legacy_stop", [None, "max_rounds", "sample_budget", "no_improvement"])
async def test_all_lineages_finish_without_round_sample_or_improvement_caps(
    round_config, fake_services, monkeypatch, legacy_stop
):
    from opendde_harness.plugin.protein_design.initial_round import autonomous
    from opendde_harness.plugin.protein_design.initial_round.config import Service

    config = autonomous_config(round_config)
    config.generators.pop("torchcraft")
    # This request alone exceeds the old 100-sample total cap. Four actual
    # candidates each need both stages, exceeding the old 3-action cap too.
    config.generators["ppiflow"].request["samples_per_target"] = 105
    config.evaluation.torchopt = Service(url="https://torchopt.test")
    paths = [pdb(config.work_dir / "seeds" / f"{i}.pdb", binder=aa + "AAAAA") for i, aa in enumerate("ACDE")]
    base, calls = fake_services
    outputs, optimized = {}, []

    def handler(request):
        response = base.handler(request)
        if request.url.host == "ppiflow.test" and request.method == "POST":
            payload = json.loads(request.content)
            outputs[response.json()["design_id"]] = {
                "files": [payload["complex_pdb"]]
                if payload["mode"].startswith("partial_")
                else [str(path).replace(str(config.work_dir), "/shared", 1) for path in paths]
            }
        if request.url.host == "ppiflow.test" and request.url.path.endswith("/results"):
            return httpx.Response(200, json=outputs[request.url.path.split("/")[-2]])
        if request.url.host == "torchscore.test" and request.url.path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": [{"iptm": 0.7, "ptm": 0.7}]})
        return response

    async def optimize(self, decision, round_index):
        parent = next(c for c in self.checkpoint.data["candidates"] if c["candidate_id"] == decision["parent_id"])
        child = copy.deepcopy(parent)
        child.update(
            candidate_id="optimized-" + parent["candidate_id"],
            parent_id=parent["candidate_id"],
            optimized_by="torchopt",
            round_index=round_index,
            generation_stage=f"round-{round_index}/torchopt",
            metrics={},
            status="generated",
        )
        optimized.append(parent["candidate_id"])
        self.checkpoint.data["candidates"].append(child)
        self.save_candidates()
        return [child]

    monkeypatch.setattr(InitialRoundRunner, "optimize", optimize)
    if legacy_stop:
        original = autonomous.run_autonomous

        async def legacy_state(runner):
            runner.checkpoint.data["autonomous"] = {
                "decisions": [],
                "used_samples": 105,
                "repaired": [],
                "stop_reason": legacy_stop,
                "stale_rounds": 2,
            }
            return await original(runner)

        monkeypatch.setattr(autonomous, "run_autonomous", legacy_state)
    transport = httpx.MockTransport(handler)
    runner = InitialRoundRunner(config, transport=transport)
    result = await runner.run()
    decisions = runner.checkpoint.data["autonomous"]["decisions"]
    assert len(decisions) == 8
    assert [d["action"] for d in decisions].count("partial_flow") == 4
    assert [d["action"] for d in decisions].count("torchopt") == 4
    assert len(set(d["parent_id"] for d in decisions)) == 8
    assert len(optimized) == 4
    assert result["used_samples"] == 113  # Counter only; never a limit.
    assert result["stop_reason"] == "no_legal_actions"
    candidates = runner.checkpoint.data["candidates"]
    assert len(candidates) == 12
    assert all(c["status"] == "terminated" for c in candidates if c.get("optimized_by"))
    if not legacy_stop:
        before = copy.deepcopy(calls)
        assert await InitialRoundRunner(config, transport=transport).run() == result
        assert {k: v for k, v in calls.items() if k[1] == "POST"} == {k: v for k, v in before.items() if k[1] == "POST"}


async def test_agent_cannot_stop_while_another_candidate_needs_a_followup(round_config, fake_services, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.autonomous import Choice

    config = autonomous_config(round_config, decision_policy="agent")
    base, calls = fake_services
    seen_actions = []

    def handler(request):
        response = base.handler(request)
        if request.url.host == "torchscore.test" and request.url.path.endswith("/metrics"):
            # One high-scoring initial candidate, one that needs Partial Flow.
            score = 0.7 if calls[("torchscore", "POST")] == 2 else 0.9
            return httpx.Response(200, json={"metrics": [{"iptm": score, "ptm": 0.9}]})
        return response

    async def choose(context):
        actions = [o["action"] for o in context["options"]]
        seen_actions.append(actions)
        assert "remaining_samples" not in context
        if "partial_flow" in actions:
            assert "stop" not in actions
        return Choice(option_id=0, reason="finish the pending lineage before stopping")

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.autonomous.agent_choice", choose)
    result = await InitialRoundRunner(config, transport=httpx.MockTransport(handler)).run()
    assert result["accepted"] == 2
    assert seen_actions == [["partial_flow"], ["stop"]]


def test_saved_budget_settings_do_not_block_checkpoint_recovery(round_config, tmp_path):
    from opendde_harness.plugin.protein_design.initial_round.storage import Checkpoint

    current = {"config": autonomous_config(round_config).model_dump(mode="json"), "inputs": {}}
    previous = copy.deepcopy(current)
    previous["config"]["autonomous"].update(max_rounds=3, max_samples=100, batch_size=5, max_batch_size=20, patience=2)
    root = tmp_path / "legacy-checkpoint"
    root.mkdir()
    checkpoint = Checkpoint(root, previous)
    checkpoint.data["jobs"]["existing"] = {"id": "original-remote-id", "state": "completed"}
    checkpoint.save()
    resumed = Checkpoint(root, current)
    assert resumed.data["jobs"]["existing"]["id"] == "original-remote-id"
    assert resumed.data["config_snapshot"] == current
