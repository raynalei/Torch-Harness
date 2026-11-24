"""Protein-design tools integrate with the main assistant's built-in prompt."""

from pathlib import Path
from unittest.mock import Mock

from opendde_harness.agent.tools.shell import ExecTool
from opendde_harness.context_engine.segments import render
from opendde_harness.memory_engine.skill_local.registry import SkillRegistry
from opendde_harness.plugin import active_registry
from opendde_harness.plugin.protein_design.readiness import models_configured

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SKILLS = REPO_ROOT / "opendde_harness" / "plugin" / "protein_design" / "skills"


def test_plugin_is_active_with_its_data_contributions():
    registry = active_registry()
    assert "protein-design" in registry.activated_ids()
    assert registry.prompt_segments("identity") == []
    assert registry.prompt_segments("scope") == []
    assert registry.skills_dirs() == [PLUGIN_SKILLS]
    assert [plugin_id for plugin_id, _ in registry.readiness_checks()] == ["protein-design"]


def test_identity_prompt_keeps_the_antibody_block_in_place(tmp_path):
    text = render.identity_text(tmp_path, model="openai/gpt-x")
    head = f"# OpenDDE Harness ϒ\n\n{render.PROTEIN_DESIGN_IDENTITY}\n\n## Scope\n"
    assert text.startswith(head)
    assert f"{render.PROTEIN_DESIGN_SCOPE}\n\n## Runtime\n" in text
    assert text.count(render.PROTEIN_DESIGN_SCOPE) == 1
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings[:3] == ["## Scope", "## Running and reading a design", "## Runtime"]


def test_identity_without_design_plugin_preserves_other_contributions(monkeypatch, tmp_path):
    registry = Mock()
    registry.activated_ids.return_value = []
    registry.prompt_segments.side_effect = lambda slot: {
        "identity": ["You are Demo."],
        "scope": ["## Demo Scope\n- be brief"],
    }[slot]
    monkeypatch.setattr(render, "active_registry", lambda: registry)

    text = render.identity_text(tmp_path)

    assert "You are Demo." in text
    assert "## Demo Scope\n- be brief" in text
    assert render.PROTEIN_DESIGN_IDENTITY not in text
    assert render.PROTEIN_DESIGN_SCOPE not in text


def test_language_directive_sits_between_identity_and_scope(tmp_path):
    text = render.identity_text(tmp_path, model="openai/gpt-x", language="zh")
    assert f"{render.PROTEIN_DESIGN_IDENTITY}\n\nAlways respond in Simplified Chinese (简体中文)" in text
    assert "another language.\n\n## Scope" in text
    assert "简体中文" not in render.identity_text(tmp_path, model="openai/gpt-x")


def test_workspace_block_does_not_advertise_local_memory_for_external_backends(tmp_path):
    from opendde_harness.config.paths import get_workspace_storage

    skills_line = f"- Custom skills: {get_workspace_storage(tmp_path).skills}/{{skill-name}}/SKILL.md"

    off = render.identity_text(tmp_path, model="openai/gpt-x", long_term_memory=False)
    assert f"## Workspace\nYour workspace is at: {tmp_path}\n{skills_line}\n\n" in off
    assert "user_memory" not in off

    on = render.identity_text(tmp_path, model="openai/gpt-x", long_term_memory=True)
    assert "user.md" not in on
    assert "episodes.md" not in on


def test_the_model_line_is_omitted_when_the_turn_names_no_model(tmp_path):
    """The turn hands in the id the request reaches. Rendering used to read the
    configured id from config, which is not the routed one."""
    assert "running on model" not in render.identity_text(tmp_path)
    assert "You are running on model: openai/gpt-x." in render.identity_text(tmp_path, model="openai/gpt-x")


def test_bash_description_carries_the_plugin_note():
    """The note is keyed on the tool's advertised name, which is now ``bash``."""
    registry_note = "Use protein_model_list and protein_initial_round_start for model discovery and design workflows."
    assert ExecTool().description.endswith(registry_note)


def test_skills_are_discovered_from_the_plugin_directory(tmp_path):
    registry = SkillRegistry(tmp_path / "ws")
    builtin = {m.name: m.path for m in registry.list_all() if m.source == "builtin"}
    assert {"pdb-lookup", "uniprot-lookup", "pubmed-lookup", "protein-design"} <= set(builtin)
    assert "weather" in builtin
    assert all(
        builtin[name].is_relative_to(PLUGIN_SKILLS)
        for name in ("pdb-lookup", "uniprot-lookup", "pubmed-lookup", "protein-design")
    )


def test_readiness_uses_registered_services_and_workflow_profiles():
    assert not models_configured({})
    assert not models_configured({"compute_url": "http://127.0.0.1:8080"})
    assert models_configured({"model_services": "models.json"})
    assert models_configured({"initial_round_config": "workflow.yaml"})
    assert models_configured({"deployment_config": "deployment.yaml"})
