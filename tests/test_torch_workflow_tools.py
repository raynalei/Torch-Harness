"""Public model workflows and native analysis do not load a compute client."""

import json
import subprocess
import sys

import pytest

from opendde_harness.plugin.protein_design.tools.workflow import AnalysisTool, WorkflowTool, prepare_workflow


def test_public_tool_import_does_not_load_retired_client():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import opendde_harness.plugin.protein_design.tools.workflow; assert 'opendde_harness.plugin.protein_design.servers.client' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_prepare_workflow_defaults_autonomous_without_mutating_input(monkeypatch):
    value = {"name": "task", "evaluation": {}}
    seen = {}

    def prepare(config, **kwargs):
        seen.update(config)
        return config

    monkeypatch.setattr("opendde_harness.plugin.protein_design.tools.workflow.prepare_run", prepare)
    prepare_workflow(value, registrations={})
    assert seen["autonomous"] == {"planner_layout": "two_stage"}
    assert "autonomous" not in value


async def test_context_reports_capabilities_without_leaking_credentials():
    tool = WorkflowTool(
        {"compute_url": "https://old.invalid", "model_services": {"score": {"kind": "torchscore", "token": "secret"}}},
        "context",
    )
    result = await tool.execute()
    value = json.loads(result.model_text)
    assert {item["alias"]: item["kind"] for item in value["models"]} == {
        "score": "torchscore",
        "rfd4": "rfd4",
        "proteinmpnn": "proteinmpnn",
    }
    assert value["compute_service_required"] is False
    assert "EpiMatrix" in value["unavailable"]
    assert "secret" not in result.model_text
    assert "old.invalid" not in result.model_text


async def test_retired_lookup_is_no_longer_in_analysis_tool():
    schema = AnalysisTool().parameters
    assert "lookup" not in schema["properties"]["action"]["enum"]


async def test_analysis_never_substitutes_unknown_immunogenicity():
    value = json.loads((await AnalysisTool().execute(action="developability", sequences={"B": "NGST"})).model_text)
    assert value["ptm_risks"]["available"] is True
    assert value["immunogenicity"]["available"] is False
    assert value["immunogenicity"]["scores"] is None


@pytest.mark.parametrize("action", ["start", "resume", "regions"])
def test_execution_tools_declare_external_effects(action):
    assert WorkflowTool({}, action).external_effects


def test_real_configuration_uses_torchscore_for_default_workflow(tmp_path):
    from tests.test_protein_design_initial_round import autonomous_config, round_config

    base = autonomous_config(round_config.__wrapped__(tmp_path)).model_dump(mode="json")
    base.pop("autonomous")
    config = prepare_workflow(base, registrations={})
    assert config.autonomous is not None
    assert config.evaluation.scoring_backend == "torchscore"
    assert "autonomous" not in base


def test_real_workflow_preparation_uses_torchscore_and_enables_router(tmp_path):
    from tests.test_protein_design_initial_round import autonomous_config, round_config

    config = autonomous_config(round_config.__wrapped__(tmp_path))
    value = config.model_dump(mode="json")
    value.pop("autonomous")
    value["evaluation"].pop("scoring_backend")
    prepared = prepare_workflow(value, registrations={})
    assert prepared.autonomous is not None
    assert prepared.evaluation.scoring_backend == "torchscore"
    assert "autonomous" not in value


def test_context_builder_retains_measured_developability_domains():
    from opendde_harness.plugin.protein_design.scientific_agent.backends.developability import assess_developability
    from opendde_harness.plugin.protein_design.scientific_agent.context_builder import extract

    measured = assess_developability({"B": "ANGSM"})
    candidate = {"candidate_id": "test", "metadata": {"developability_evidence": measured}}
    context = extract(candidate)
    assert context["developability"]["ptm_risks"]["count"] > 0
    assert context["developability"]["immunogenicity"]["available"] is False
    assert context["developability"]["surface_patches"]["scores"] is None
