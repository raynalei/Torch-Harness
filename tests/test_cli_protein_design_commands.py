"""Protein-design YAML validation summaries expose reviewed cycle policies."""

import json

from typer.testing import CliRunner

from opendde_harness.cli import protein_design_commands as commands


def test_initial_round_cli_loads_separate_config_and_check_mode(tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig
    from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner

    source = tmp_path / "initial.yaml"
    source.write_text("name: example")
    sentinel = object()
    monkeypatch.setattr(InitialRoundConfig, "load", lambda path: sentinel)
    seen = {}

    def initialize(self, config, *, attach_jobs):
        seen.update(config=config, attach_jobs=attach_jobs)

    async def run(self, *, check_only):
        seen["check_only"] = check_only
        return {"status": "checked", "inference_verified": False}

    monkeypatch.setattr(InitialRoundRunner, "__init__", initialize)
    monkeypatch.setattr(InitialRoundRunner, "run", run)
    result = CliRunner().invoke(
        commands.protein_design_app,
        ["initial-round", "-c", str(source), "--check", "--attach-job", "generate/ppiflow=known"],
    )
    assert result.exit_code == 0, result.output
    assert seen == {"config": sentinel, "attach_jobs": {"generate/ppiflow": "known"}, "check_only": True}
    assert json.loads(result.output)["inference_verified"] is False


def test_initial_round_cli_invalid_config_is_nonzero(tmp_path):
    source = tmp_path / "initial.yaml"
    source.write_text("name: missing-fields")
    result = CliRunner().invoke(commands.protein_design_app, ["initial-round", "-c", str(source)])
    assert result.exit_code == 1
    assert "Initial round failed" in result.output


def test_initial_round_cli_detached_recovery_passes_attachments(tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig

    source = tmp_path / "initial.yaml"
    source.write_text("name: example")
    sentinel = object()
    monkeypatch.setattr(InitialRoundConfig, "load", lambda _: sentinel)
    seen = {}

    def launch(config, *, attach_jobs):
        seen.update(config=config, attach_jobs=attach_jobs)
        return {"state": "starting"}

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.automation.start_round", launch)
    result = CliRunner().invoke(
        commands.protein_design_app,
        ["initial-round", "-c", str(source), "--detach", "--attach-job", "generate/ppiflow=known"],
    )
    assert result.exit_code == 0, result.output
    assert seen == {"config": sentinel, "attach_jobs": {"generate/ppiflow": "known"}}


def test_initial_round_cli_partial_result_is_nonzero(tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig
    from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner

    source = tmp_path / "initial.yaml"
    source.write_text("name: example")
    monkeypatch.setattr(InitialRoundConfig, "load", lambda path: object())
    monkeypatch.setattr(InitialRoundRunner, "__init__", lambda *args, **kwargs: None)

    async def run(self, *, check_only):
        return {"status": "partial", "failed": ["candidate"]}

    monkeypatch.setattr(InitialRoundRunner, "run", run)
    result = CliRunner().invoke(commands.protein_design_app, ["initial-round", "-c", str(source)])
    assert result.exit_code == 2 and json.loads(result.output)["status"] == "partial"


def test_initial_round_resume_cli_uses_run_directory_without_yaml(tmp_path, monkeypatch):
    observed = []

    def resume(root):
        observed.append(root)
        return {"state": "starting", "run_dir": str(root)}

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.automation.resume_round", resume)
    result = CliRunner().invoke(commands.protein_design_app, ["initial-round-resume", "--run-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["state"] == "starting"
    assert observed == [tmp_path]


def test_cli_supplies_regions_to_the_pending_question_then_resumes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.initial_round.region_input.supply_regions",
        lambda *args: calls.append(args),
    )
    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.initial_round.automation.resume_round",
        lambda root: {"state": "starting", "run_dir": str(root)},
    )
    result = CliRunner().invoke(
        commands.protein_design_app,
        ["initial-round-regions", "--run-dir", str(tmp_path), "--question-id", "a" * 64, "--cdr-position", "B5-6"],
    )
    assert result.exit_code == 0, result.output
    assert calls == [(tmp_path, "a" * 64, "B5-6", None)]
    assert json.loads(result.output)["state"] == "starting"


def test_initial_round_list_cli_reads_saved_runs(tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round import automation

    run_dir = tmp_path / "run"
    expected = {"runs": [{"run_dir": str(run_dir), "state": "running"}]}

    def list_rounds(work_dir):
        assert work_dir == tmp_path
        return expected

    monkeypatch.setattr(automation, "list_rounds", list_rounds)
    result = CliRunner().invoke(commands.protein_design_app, ["initial-round-list", "--work-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == expected
