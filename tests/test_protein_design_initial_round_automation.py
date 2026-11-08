"""Detached initial-round control and Qwen launch integration."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from opendde_harness.cli.protein_design_commands import protein_design_app
from opendde_harness.plugin import active_registry
from opendde_harness.plugin.context import ServiceLocator
from opendde_harness.plugin.protein_design.initial_round.automation import prepare_run, round_status, start_round
from opendde_harness.plugin.protein_design.initial_round.storage import write_json


@pytest.fixture
def profile(tmp_path):
    import yaml

    value = yaml.safe_load((Path(__file__).parents[1] / "docs/examples/initial_round.yaml").read_text())
    value.update(
        name="automated", work_dir=str(tmp_path), path_mappings=[{"local": str(tmp_path), "remote": "/shared"}]
    )
    value["generators"].pop("torchcraft")
    value["generators"]["ppiflow"]["fold_template"] = None
    value["generators"]["ppiflow"]["request"]["antigen_pdb"] = "/shared/input.pdb"
    value["generators"]["ppiflow"]["request"]["framework_dir"] = "/shared/frameworks"
    (tmp_path / "input.pdb").write_text("input")
    (tmp_path / "frameworks").mkdir()
    value["evaluation"]["rosetta"] = None
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(value))
    return path


def test_input_pdb_and_name_override_without_changing_profile(profile, tmp_path):
    pdb = tmp_path / "input.pdb"
    pdb.write_text("test input")
    before = profile.read_bytes()
    config = prepare_run(profile, pdb_path=pdb, name="target_001")
    assert config.name == "target_001"
    assert config.generators["ppiflow"].request["antigen_pdb"] == "/shared/input.pdb"
    assert profile.read_bytes() == before


def test_detached_launch_reuses_completed_run_and_refuses_changed_config(profile, monkeypatch):
    config = prepare_run(profile)
    root = config.work_dir / config.name
    calls = []

    class Process:
        pid = 12345

        def wait(self):
            return 0

    def popen(*args, **kwargs):
        calls.append(args)
        return Process()

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.automation.subprocess.Popen", popen)
    result = start_round(config)
    assert result["state"] == "starting"
    assert "worker" in " ".join(calls[0][0])
    write_json(root / "worker-status.json", {"state": "completed", "summary": {"accepted": 1}})
    assert start_round(config)["state"] == "completed" and len(calls) == 1
    config.evaluation.min_af3score_iptm = 0.9
    with pytest.raises(ValueError, match="configuration changed"):
        start_round(config)


def test_exited_worker_is_reported_as_interrupted(profile, monkeypatch):
    config = prepare_run(profile)
    root = config.work_dir / config.name
    root.mkdir()
    write_json(root / "launch.json", {"pid": 123})
    write_json(root / "worker-status.json", {"state": "running"})
    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.automation._alive", lambda *_: False)
    assert round_status(root)["state"] == "interrupted"


async def test_qwen_pipeline_tool_launches_once_and_does_not_poll_apis(profile, tmp_path, monkeypatch):
    registry = active_registry()
    tool = registry.build_tool(
        "protein_initial_round_start",
        config={"initial_round_config": str(profile)},
        services=ServiceLocator(workspace=tmp_path, user_id="test", agent_id="test"),
    )
    pdb = tmp_path / "input.pdb"
    pdb.write_text("input")
    launched = []

    def launch(config):
        launched.append(config)
        return {"state": "starting", "run_dir": str(config.work_dir / config.name)}

    monkeypatch.setattr("opendde_harness.plugin.protein_design.tools.initial_round.start_round", launch)
    result = await tool.execute(pdb_path=str(pdb), name="run1", user_confirmed=True)
    assert json.loads(result.model_text)["state"] == "starting"
    assert len(launched) == 1 and launched[0].generators["ppiflow"].request["antigen_pdb"] == "/shared/input.pdb"
    assert "protein_initial_round_status" in registry.tool_names()


def test_cli_detach_returns_without_running_foreground(profile, monkeypatch):
    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.initial_round.automation.start_round",
        lambda _: {"state": "starting", "run_dir": "/shared/run"},
    )
    result = CliRunner().invoke(protein_design_app, ["initial-round", "-c", str(profile), "--detach"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["state"] == "starting"


def test_status_on_another_login_node_does_not_treat_remote_pid_as_dead(profile, monkeypatch):
    config = prepare_run(profile)
    root = config.work_dir / config.name
    root.mkdir()
    write_json(root / "launch.json", {"pid": 123, "hostname": "another-login-node"})
    write_json(root / "worker-status.json", {"state": "running"})
    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.automation._alive", lambda *_: False)
    result = round_status(root)
    assert result["state"] == "running" and result["worker_host"] == "another-login-node"
