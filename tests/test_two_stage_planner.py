"""Two scientific decisions separate screening from full evaluation."""

import json

import pytest

from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner
from opendde_harness.plugin.protein_design.scientific_agent import planner
from tests._slurm_api import mock_slurm_apis  # noqa: F401
from tests.test_protein_design_initial_round import autonomous_config, fake_services, round_config  # noqa: F401
from tests.test_scientific_planner import choose

pytestmark = [pytest.mark.in_memory_transport, pytest.mark.usefixtures("mock_slurm_apis")]


def config_for_stages(config):
    return autonomous_config(config, decision_policy="planner", planner_layout="two_stage")


async def test_two_stages_have_distinct_evaluations_and_no_third_decision(round_config, fake_services, monkeypatch):
    transport, calls = fake_services
    stages = []

    async def decide(context, **kwargs):
        stage = context["scientific_decision_stage"]
        stages.append(stage)
        assert context["scientific_decision_stages"] == 2
        if stage == 1:
            assert set(context["evaluation_skills"]) == {"torchscore", "rosetta-interface", "mutation_selector"}
            assert calls[("rosetta", "POST")] == 2
            assert calls[("torchfold", "POST")] == 0
            for record in context["candidates"]:
                assert "rosetta_interface_score" in record["measured_metrics"]
                assert record["completed_analysis"] == []
                assert record["developability"]["available"] is False
                assert record["hotspot"]["available"] is False
            return choose(context, "analysis", phase="advance")
        for record in context["candidates"]:
            assert record["developability"]["ptm_risks"]["available"] is True
            assert record["developability"]["immunogenicity"]["available"] is False
            assert set(record["completed_analysis"]) == set(planner.EVALUATION_SKILLS)
        return choose(context, "continue", phase="final_validation")

    monkeypatch.setattr(planner, "planner_choice", decide)
    config = config_for_stages(round_config)
    result = await InitialRoundRunner(config, transport=transport).run()
    assert stages == [1, 1, 2, 2]
    assert result["accepted"] == 2
    assert result["followup_rounds"] == result["scientific_decision_stages"] == 2
    assert result["planner_decisions"] == 4
    before = dict(calls)
    await InitialRoundRunner(config, transport=transport).run()
    assert dict(calls) == before
    assert stages == [1, 1, 2, 2]


async def test_second_stage_optimization_is_rescored_and_finalized_without_third_qwen(
    round_config, fake_services, monkeypatch
):
    transport, calls = fake_services
    stages = []

    async def decide(context, **kwargs):
        stage = context["scientific_decision_stage"]
        stages.append(stage)
        if stage == 1:
            return choose(context, "continue")
        choice = choose(context, "partial_flow")
        return choice.model_copy(update={"selection": planner.MutationSelection(fixed_positions={"B": [1]})})

    monkeypatch.setattr(planner, "planner_choice", decide)
    config = config_for_stages(round_config)
    result = await InitialRoundRunner(config, transport=transport).run()
    assert stages == [1, 1, 2, 2]
    assert result["accepted"] == 2
    assert calls[("torchscore", "POST")] == 4
    assert calls[("torchfold", "POST")] == 2
    candidates = json.loads((config.work_dir / config.name / "candidates.json").read_text())
    children = [c for c in candidates if c.get("parent_id")]
    assert all(c["scientific_decision_stage"] == 2 and c["final_validation"]["decision_stage"] == 2 for c in children)
    assert all("developability_evidence" not in c.get("metadata", {}) for c in children)


async def test_first_stage_can_stop_without_running_developability_or_fold(round_config, fake_services, monkeypatch):
    transport, calls = fake_services
    stages = []

    async def decide(context, **kwargs):
        stages.append(context["scientific_decision_stage"])
        return choose(context, "stop")

    monkeypatch.setattr(planner, "planner_choice", decide)
    config = config_for_stages(round_config)
    result = await InitialRoundRunner(config, transport=transport).run()
    assert stages == [1, 1]
    assert calls[("rosetta", "POST")] == 2
    assert calls[("torchfold", "POST")] == 0
    assert result["accepted"] == 0
    candidates = json.loads((config.work_dir / config.name / "candidates.json").read_text())
    assert all("analysis_results" not in c and "2" not in c["scientific_evaluations"] for c in candidates)


async def test_stage_fallback_still_respects_two_decision_stages(round_config, fake_services, monkeypatch):
    transport, calls = fake_services

    async def unavailable(*args, **kwargs):
        raise TimeoutError("planner offline")

    monkeypatch.setattr(planner, "planner_choice", unavailable)
    config = config_for_stages(round_config)
    result = await InitialRoundRunner(config, transport=transport).run()
    state = json.loads((config.work_dir / config.name / "decisions.json").read_text())
    assert [d["scientific_decision_stage"] for d in state["decisions"]] == [1, 1, 2, 2]
    assert all(d["policy"] == "rules_fallback" for d in state["decisions"])
    assert calls[("rosetta", "POST")] == 2
    assert result["accepted"] == 2


async def test_first_optimization_children_receive_second_stage_evaluation(round_config, fake_services, monkeypatch):
    transport, calls = fake_services
    stages = []

    async def decide(context, **kwargs):
        stage = context["scientific_decision_stage"]
        stages.append(stage)
        if stage == 1:
            choice = choose(context, "partial_flow")
            return choice.model_copy(update={"selection": planner.MutationSelection(fixed_positions={"B": [1]})})
        assert all(c["parent_id"] for c in context["candidates"])
        assert all(c["developability"]["ptm_risks"]["available"] for c in context["candidates"])
        return choose(context, "stop")

    monkeypatch.setattr(planner, "planner_choice", decide)
    config = config_for_stages(round_config)
    await InitialRoundRunner(config, transport=transport).run()
    assert stages == [1, 1, 2, 2]
    assert calls[("torchscore", "POST")] == calls[("rosetta", "POST")] == 4
    assert calls[("torchfold", "POST")] == 0


async def test_final_validation_failure_resumes_without_third_decision(round_config, fake_services, monkeypatch):
    transport, calls = fake_services
    stages = []
    real_refold = InitialRoundRunner.refold
    fail_once = [True]

    async def decide(context, **kwargs):
        stages.append(context["scientific_decision_stage"])
        return choose(context, "continue")

    async def refold(self, candidate):
        if fail_once:
            fail_once.pop()
            raise RuntimeError("temporary final-validation failure")
        return await real_refold(self, candidate)

    monkeypatch.setattr(planner, "planner_choice", decide)
    monkeypatch.setattr(InitialRoundRunner, "refold", refold)
    config = config_for_stages(round_config)
    with pytest.raises(RuntimeError, match="temporary final-validation"):
        await InitialRoundRunner(config, transport=transport).run()
    assert stages == [1, 1, 2]
    result = await InitialRoundRunner(config, transport=transport).run()
    assert stages == [1, 1, 2, 2]
    assert calls[("rosetta", "POST")] == 2
    assert result["accepted"] == 2


def test_two_stage_layout_requires_rosetta_and_preserves_legacy_config(round_config):
    from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig

    legacy = autonomous_config(round_config, decision_policy="planner")
    assert "planner_layout" not in legacy.autonomous.model_dump()
    value = config_for_stages(round_config).model_dump(mode="json")
    value["evaluation"]["rosetta"] = None
    with pytest.raises(ValueError, match="require.*Rosetta"):
        InitialRoundConfig.model_validate(value)


async def test_explicit_rules_mode_keeps_its_original_execution_and_counters(round_config, fake_services, monkeypatch):
    transport, calls = fake_services

    async def unexpected(*args, **kwargs):
        pytest.fail("explicit rules must not call Qwen")

    monkeypatch.setattr(planner, "planner_choice", unexpected)
    config = config_for_stages(round_config)
    config.autonomous.decision_policy = "rules"
    result = await InitialRoundRunner(config, transport=transport).run()
    assert result["accepted"] == 2
    assert "planner_layout" not in result
    assert result["stop_reason"] == "rules_stop"
    assert calls[("rosetta", "POST")] == 0
