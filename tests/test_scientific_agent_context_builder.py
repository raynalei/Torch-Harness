"""Scientific projections preserve unknown evidence, executable identities and size bounds."""

from __future__ import annotations

import copy
import json

import pytest

from opendde_harness.plugin.protein_design.scientific_agent.context_builder import (
    GUIDES,
    REFERENCES,
    ContextBudgetError,
    build_context,
    decision_guidance,
    extract,
    token_cost,
)
from tests._slurm_api import mock_slurm_apis  # noqa: F401
from tests.test_protein_design_initial_round import fake_services, round_config  # noqa: F401

pytestmark = [pytest.mark.in_memory_transport, pytest.mark.usefixtures("mock_slurm_apis")]


def candidate(identifier="parent", **fields):
    value = {
        "candidate_id": identifier,
        "source": "ppiflow",
        "status": "rejected",
        "generated_structure": "/shared/generated.pdb",
        "metrics": {"torchscore_iptm": 0.7, "torchscore_ptm": 0.9, "plddt": 85.0, "rosetta_interface_score": -7},
        "metric_units": {"plddt": "0-100"},
        "diagnostics": {
            "epitope_hit": False,
            "hotspot_structure": "generated",
            "hotspots": [{"hotspot": "A1", "distance": 9.0}],
        },
        "interface_fixed_positions": {"B": [1]},
        "interface_residue_energies": {"B": {"1": -7}},
        "metadata": {"developability_evidence": {"available": True, "pI": 8.1, "liabilities": ["unpaired_cys"]}},
    }
    value.update(fields)
    return value


def option(parent="parent", action="partial_flow", **fields):
    return {
        "action": action,
        "source": "ppiflow",
        "parent_id": parent,
        "request": {"token": "must-not-enter-prompt", "complex_pdb": "/shared/full.pdb"},
        **fields,
    }


def build(candidates=None, options=None, **kwargs):
    return build_context(
        [candidate()] if candidates is None else candidates,
        [option()] if options is None else options,
        round_index=1,
        **kwargs,
    )


def test_extract_preserves_structure_interface_hotspot_and_measured_developability():
    evidence = extract(candidate())
    assert evidence["structure"]["path"] == "/shared/generated.pdb"
    assert evidence["structure"]["metrics"]["plddt"] == 85.0
    assert evidence["structure"]["units"]["plddt"] == "0-100"
    assert evidence["interface"]["fixed_positions"]["B"]["positions"] == [1]
    assert evidence["hotspot"]["epitope_hit"] is False
    assert evidence["developability"]["pI"] == 8.1


def test_missing_or_nonfinite_values_remain_unknown_and_do_not_mix_scorers():
    evidence = extract(candidate(metrics={"af3score_iptm": 0.99, "torchscore_ptm": float("nan")}, metadata={}))
    assert evidence["metrics"] == {"torchscore_iptm": None, "torchscore_ptm": None}
    assert evidence["developability"] == {"available": False}
    json.dumps(evidence, allow_nan=False)


def test_serialized_core_candidate_metadata_is_projected_without_inventing_torchscore():
    evidence = extract(
        {
            "candidate_id": "legacy",
            "sequence": "SECRET-SEQUENCE",
            "structure_path": "/shared/core.pdb",
            "metrics": {"iptm": 0.75},
            "metadata": {"fold": {"plddt": 0.9}, "gate_evidence": {"gate_passed": False, "missed_hotspots": [1]}},
        }
    )
    assert evidence["structure"]["metrics"]["plddt"] == 0.9
    assert evidence["structure"]["metrics"]["iptm"] == 0.75
    assert evidence["metrics"]["torchscore_iptm"] is None
    assert evidence["hotspot"]["missed_hotspots"] == [1]
    assert "SECRET-SEQUENCE" not in json.dumps(evidence)


def test_rule_recommendation_is_distinct_from_real_executable_capabilities():
    c = candidate(generation_request={"mode": "partial_nanobody", "fixed_positions": "B1", "cdr_position": "B2-6"})
    c["diagnostics"]["allowed_actions"] = ["invented-action"]
    context = build([c], [option(action="torchopt")], max_tokens=8000)
    record = context["candidates"][0]
    assert record["stage"] == "partial_flow"
    assert record["rule_recommendation"] == record["available_actions"] == ["torchopt"]
    assert record["lineage_constraints"]["fixed_positions"] == "B1"
    assert "lowest_residue_energies" not in record["interface"]
    # No registered/advertised model is promoted into an action by the builder.
    assert "invented-action" not in json.dumps(context)


def test_context_contains_history_results_constraints_and_trusted_skill_guidance():
    child = candidate("child", parent_id="parent", round_index=1, generation_request={"mode": "partial_nanobody"})
    context = build(
        [candidate(), child],
        [option("child", "torchopt")],
        max_tokens=12000,
        history=[
            {
                "round": 1,
                "action": "partial_flow",
                "parent_id": "parent",
                "state": "completed",
                "request": {"token": "hidden"},
            }
        ],
        constraints={"ppiflow": {"binder_chains": ["B"], "target_chains": ["A"]}},
    )
    assert context["history"][0]["results"][0]["candidate_id"] == "child"
    assert context["history"][0]["results"][0]["metrics"]["torchscore_iptm"] == 0.7
    assert context["constraints"]["ppiflow"]["binder_chains"] == ["B"]
    assert set(context["guidance"]) == set(GUIDES)
    assert "TorchOptimization" in context["guidance"]["generation_strategy.md"]
    assert "must-not-enter-prompt" not in json.dumps(context)
    assert "hidden" not in json.dumps(context)


def test_stage_guidance_is_loaded_from_packaged_skill_references():
    for name in GUIDES:
        assert (REFERENCES / name).is_file()
    partial = decision_guidance({"partial_flow"})["generation_strategy.md"]
    opt = decision_guidance({"torchopt"})["generation_strategy.md"]
    assert "## Partial redesign" in partial and "## TorchOptimization" not in partial
    assert "## TorchOptimization" in opt and "## Partial redesign" not in opt


def test_default_payload_keeps_guidance_and_is_bounded():
    context = build()
    assert len(context["guidance"]) == 3
    assert token_cost(context) <= 4096
    assert context["candidates"][0]["rule_recommendation"] == ["partial_flow"]


def test_large_unicode_evidence_and_menus_never_exceed_budget_or_change_option_ids():
    candidates = [candidate(f"candidate-{i}") for i in range(90)]
    for c in candidates:
        c["diagnostics"]["notes"] = ["复杂结构结果" * 10000]
        c["interface_fixed_positions"] = {"B": list(range(5000))}
    options = [option(c["candidate_id"]) for c in candidates]
    before = copy.deepcopy((candidates, options))
    context = build(candidates, options, max_tokens=3500)
    assert token_cost(context) <= 3500
    assert context["compression"]["omitted_options"] > 0
    assert context["options"]
    for visible in context["options"]:
        assert visible["parent_id"] == options[visible["option_id"]]["parent_id"]
        assert any(c["candidate_id"] == visible["parent_id"] for c in context["candidates"])
    assert (candidates, options) == before


def test_option_pagination_keeps_original_indices_when_an_earlier_parent_cannot_fit():
    first, second = candidate("too-large"), candidate("small")
    first["diagnostics"]["notes"] = ["extra evidence" * 50] * 12
    context = build([first, second], [option("too-large"), option("small")], max_tokens=2900)
    assert [o["option_id"] for o in context["options"]] == [1]
    assert context["candidates"][0]["candidate_id"] == "small"


def test_compression_fallback_keeps_essential_rules_and_explicitly_marks_minimal_evidence():
    context = build(max_tokens=900)
    assert context["evidence_detail"] == "minimal"
    assert token_cost(context) <= 900
    record = context["candidates"][0]
    assert record["rule_recommendation"] == record["available_actions"] == ["partial_flow"]
    assert record["metrics"]["torchscore_iptm"] == 0.7


@pytest.mark.parametrize("budget", [1, 50, 150])
def test_impossible_budget_fails_instead_of_sending_an_invalid_context(budget):
    with pytest.raises(ContextBudgetError):
        build(max_tokens=budget)


def test_option_without_its_candidate_evidence_is_rejected():
    with pytest.raises(ContextBudgetError, match="no candidate evidence"):
        build([], [option("missing")])


@pytest.mark.asyncio
async def test_autonomous_agent_uses_the_scientific_builder(round_config, fake_services, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.autonomous import Choice
    from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner
    from tests.test_protein_design_initial_round import autonomous_config

    captured = []

    async def choose(context):
        captured.append(context)
        assert set(context["guidance"]) == set(GUIDES)
        assert context["thresholds"]["min_refold_score"] == 0.8
        assert context["constraints"]["ppiflow"]["binder_chains"] == ["B"]
        assert context["candidates"][0]["developability"] == {"available": False}
        assert token_cost(context) <= 4096
        return Choice(option_id=context["options"][0]["option_id"], reason="follow the scientific evidence")

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.autonomous.agent_choice", choose)
    result = await InitialRoundRunner(
        autonomous_config(round_config, decision_policy="agent"), transport=fake_services[0]
    ).run()
    assert result["accepted"] == 2 and len(captured) == 1


@pytest.mark.asyncio
async def test_context_budget_failure_uses_rule_fallback_without_calling_qwen(round_config, fake_services, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner
    from tests.test_protein_design_initial_round import autonomous_config

    async def forbidden(context):
        pytest.fail("an oversized context must never be sent to the provider")

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.autonomous.agent_choice", forbidden)
    config = autonomous_config(round_config, decision_policy="agent", decision_context_tokens=256)
    runner = InitialRoundRunner(config, transport=fake_services[0])
    assert (await runner.run())["accepted"] == 2
    decision = runner.checkpoint.data["autonomous"]["decisions"][0]
    assert "ContextBudgetError" in decision["fallback"]


@pytest.mark.asyncio
async def test_rules_mode_does_not_build_llm_context(round_config, fake_services, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner
    from tests.test_protein_design_initial_round import autonomous_config

    def forbidden(*args):
        pytest.fail("rules mode must not depend on LLM context building")

    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.scientific_agent.context_builder.context_for_runner", forbidden
    )
    assert (await InitialRoundRunner(autonomous_config(round_config), transport=fake_services[0]).run())[
        "accepted"
    ] == 2


def test_measured_developability_in_metrics_is_not_lost():
    evidence = extract(
        candidate(
            metrics={"torchscore_iptm": 0.7, "torchscore_ptm": 0.9, "pi": 6.5, "aggregation_score": 0.2}, metadata={}
        )
    )
    assert evidence["developability"]["pi"] == 6.5
    assert evidence["developability"]["aggregation_score"] == 0.2


def test_large_history_reports_omissions_and_preserves_failed_action_states():
    history = [{"round": i, "action": "partial_flow", "parent_id": "parent", "state": "failed"} for i in range(30)]
    context = build(history=history, max_tokens=8000)
    assert len(context["history"]) == 8
    assert context["compression"]["omitted_history"] == 22
    assert context["history_summary"] == {"partial_flow": 30}
    assert all(row["state"] == "failed" and row["results"] == [] for row in context["history"])
