"""Autonomous interface dG is measured separately from residue contact sums."""

import copy

import httpx
import pytest

from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner
from opendde_harness.plugin.protein_design.initial_round.storage import Checkpoint
from tests._slurm_api import mock_slurm_apis  # noqa: F401
from tests.test_protein_design_initial_round import autonomous_config, fake_services, round_config  # noqa: F401

pytestmark = [pytest.mark.in_memory_transport, pytest.mark.usefixtures("mock_slurm_apis")]


@pytest.fixture
async def interface_case(round_config, fake_services):
    transport, calls = fake_services
    runner = InitialRoundRunner(autonomous_config(round_config), transport=transport)
    runner.root.mkdir(parents=True, exist_ok=True)
    runner.checkpoint = Checkpoint(runner.root, {})
    try:
        await runner.check()
        candidates = await runner.generate()
        yield runner, candidates[0], calls, transport
    finally:
        for service in runner.services.values():
            await service.close()


async def test_autonomous_records_dg_and_backfills_cached_jobs(interface_case):
    runner, candidate, calls, _ = interface_case
    await runner.score_interface(candidate)
    assert candidate["metrics"]["rosetta_interface_dg"] == -12
    assert candidate["metrics"]["rosetta_interface_score"] == -6
    assert candidate["interface_fixed_positions"] == {"B": [1]}
    before = copy.deepcopy(candidate["interface_residue_energies"])
    posts = calls[("rosetta", "POST")]
    candidate["metrics"].pop("rosetta_interface_dg")
    await runner.score_interface(candidate)
    assert candidate["metrics"]["rosetta_interface_dg"] == -12
    assert candidate["interface_residue_energies"] == before
    assert calls[("rosetta", "POST")] == posts
    await runner.score_interface(candidate)
    assert calls[("rosetta", "POST")] == posts


async def test_autonomous_reads_dg_from_scorefile(interface_case):
    runner, candidate, calls, transport = interface_case
    base = transport.handler
    score = runner.root / "interface.sc"
    score.write_text("SCORE: dG_separated total_score description\nSCORE: -8.25 -30 input\n")

    def handler(request):
        response = base(request)
        if request.url.host == "rosetta.test" and request.url.path.endswith("/results"):
            result = response.json()
            result.pop("metrics", None)
            result["files"].append(runner.files.remote(score))
            return httpx.Response(200, json=result)
        return response

    transport.handler = handler
    await runner.score_interface(candidate)
    assert candidate["metrics"]["rosetta_interface_dg"] == -8.25
    assert candidate["metrics"]["rosetta_interface_score"] == -6


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
async def test_autonomous_does_not_substitute_residue_sum_for_missing_dg(interface_case, value):
    runner, candidate, _, transport = interface_case
    base = transport.handler

    def handler(request):
        response = base(request)
        if request.url.host == "rosetta.test" and request.url.path.endswith("/results"):
            result = response.json()
            result["metrics"] = {} if value is None else {"dG_separated": str(value)}
            return httpx.Response(200, json=result)
        return response

    transport.handler = handler
    with pytest.raises(ValueError):
        await runner.score_interface(candidate)
    assert "rosetta_interface_dg" not in candidate["metrics"]


async def test_autonomous_qwen_context_includes_measured_interface_dg(round_config, fake_services, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.autonomous import Choice

    seen = []

    async def choose(context):
        scored = [c for c in context["candidates"] if c["interface_fixed_positions"] is not None]
        assert scored
        assert all(c["metrics"]["rosetta_interface_dg"] == -12 for c in scored)
        assert all(c["metrics"]["rosetta_interface_score"] == -6 for c in scored)
        seen.append(context)
        return Choice(option_id=context["options"][0]["option_id"], reason="compare measured interface evidence")

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.autonomous.agent_choice", choose)
    config = autonomous_config(round_config, decision_policy="agent", min_refold_score=0.99)
    await InitialRoundRunner(config, transport=fake_services[0]).run()
    assert seen
