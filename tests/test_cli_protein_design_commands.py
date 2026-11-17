"""Public Torch-Harness CLI bypasses the retired compute-service workflow."""

import json

import pytest
from typer.testing import CliRunner

from opendde_harness.cli import protein_design_commands as commands


@pytest.fixture
def source(tmp_path, monkeypatch):
    path = tmp_path / "workflow.yaml"
    path.write_text("name: test")
    monkeypatch.setattr(commands, "prepare_workflow", lambda *args, **kwargs: object())
    return path


def test_start_detaches_with_recovery_ids(source, monkeypatch):
    seen = {}

    def launch(config, **kwargs):
        seen.update(kwargs)
        return {"state": "starting"}

    monkeypatch.setattr(commands, "start_round", launch)
    result = CliRunner().invoke(
        commands.protein_design_app, ["start", "-c", str(source), "--attach-job", "generate/ppiflow=42"]
    )
    assert result.exit_code == 0, result.output
    assert seen == {"retry_failed": False, "attach_jobs": {"generate/ppiflow": "42"}}


@pytest.mark.parametrize("attachment", ["bad", "=42", "key="])
def test_invalid_recovery_does_not_launch(source, monkeypatch, attachment):
    monkeypatch.setattr(commands, "start_round", lambda *a, **kw: pytest.fail("unexpected launch"))
    result = CliRunner().invoke(commands.protein_design_app, ["start", "-c", str(source), "--attach-job", attachment])
    assert result.exit_code == 1


def test_check_does_not_claim_inference(source, monkeypatch):
    class Runner:
        def __init__(self, *a, **kw):
            pass

        async def run(self, *, check_only):
            assert check_only
            return {"status": "checked", "inference_verified": False}

    monkeypatch.setattr(commands, "InitialRoundRunner", Runner)
    result = CliRunner().invoke(commands.protein_design_app, ["start", "-c", str(source), "--check"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["inference_verified"] is False


def test_foreground_partial_is_nonzero(source, monkeypatch):
    class Runner:
        def __init__(self, *a, **kw):
            pass

        async def run(self, **kw):
            return {"status": "partial"}

    monkeypatch.setattr(commands, "InitialRoundRunner", Runner)
    result = CliRunner().invoke(commands.protein_design_app, ["start", "-c", str(source), "--foreground"])
    assert result.exit_code == 2
    assert json.loads(result.output)["status"] == "partial"


def test_status_and_resume_use_existing_checkpoint(tmp_path, monkeypatch):
    for action, function in [("status", "round_status"), ("resume", "resume_round")]:
        monkeypatch.setattr(commands, function, lambda path: {"run_dir": str(path)})
        result = CliRunner().invoke(commands.protein_design_app, [action, "--run-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["run_dir"] == str(tmp_path)


def test_native_analysis_needs_no_model_or_compute_service(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"action": "developability", "sequences": {"B": "ANGSM"}}))
    result = CliRunner().invoke(commands.protein_design_app, ["analyze", "--request", str(path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["immunogenicity"]["scores"] is None


def test_initial_round_is_not_a_public_command():
    result = CliRunner().invoke(commands.protein_design_app, ["--help"])
    assert "initial-round" not in result.output
    assert "analyze" in result.output


def test_invalid_configuration_is_clean_error(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("name: missing-fields")
    result = CliRunner().invoke(commands.protein_design_app, ["start", "-c", str(path)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
