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


@pytest.fixture(params=["nanobody", "antibody"])
def profile(tmp_path, request):
    import yaml

    value = yaml.safe_load((Path(__file__).parents[1] / "docs/examples/initial_round_sbatch.yaml").read_text())
    value.update(
        name="automated", work_dir=str(tmp_path), path_mappings=[{"local": str(tmp_path), "remote": "/shared"}]
    )
    value["generators"].pop("torchcraft", None)
    generator = value["generators"]["ppiflow"]
    generator["request"] = {
        "mode": "nanobody",
        "name": "test",
        "antigen_pdb": "/shared/input.pdb",
        "antigen_chain": "A",
        "framework_dir": "/shared/frameworks",
        "heavy_chain": "B",
    }
    generator["chains"] = {"target": ["A"], "binder": ["B"], "refold_map": {"A": "X", "B": "Y"}}
    # These control tests mock the detached worker rather than submit Slurm jobs.
    for service in [
        generator["service"],
        *(value["evaluation"][name] for name in ("af3score", "torchfold", "rosetta")),
    ]:
        service.pop("sbatch_script", None)
        service.pop("sbatch_state_file", None)
    value["generators"]["ppiflow"]["fold_template"] = None
    value["generators"]["ppiflow"]["request"]["antigen_pdb"] = "/shared/input.pdb"
    value["generators"]["ppiflow"]["request"]["framework_dir"] = "/shared/frameworks"
    if request.param == "antibody":
        generator = value["generators"]["ppiflow"]
        generator["request"].update(mode="antibody", light_chain="L", cdr_length="CDRH3,10-20,CDRL3,9-11")
        generator["chains"]["binder"].append("L")
        generator["chains"]["refold_map"]["L"] = "Z"
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


def test_detached_recovery_passes_verified_ids_to_worker(profile, monkeypatch):
    import runpy
    import sys

    from opendde_harness.plugin.protein_design.initial_round import worker

    config = prepare_run(profile)
    root = config.work_dir / config.name
    root.mkdir()
    write_json(root / "checkpoint.json", {"jobs": {"generate/ppiflow": {"state": "submission_uncertain"}}})
    attachments = {"generate/ppiflow": "recovered"}
    commands = []

    class Process:
        pid = 12345

        def wait(self):
            return 0

    def popen(command, **kwargs):
        commands.append(command)
        return Process()

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.automation.subprocess.Popen", popen)
    assert start_round(config, attach_jobs=attachments)["state"] == "starting"
    command = commands[0]
    assert json.loads(command[command.index("--attach-jobs") + 1]) == attachments
    seen = {}

    class Runner:
        def __init__(self, value, *, attach_jobs):
            seen.update(config=value, attach_jobs=attach_jobs)

        async def run(self):
            return {"status": "completed"}

    monkeypatch.setattr(worker, "InitialRoundRunner", Runner)
    worker.run(root, attach_jobs=attachments)
    assert seen["attach_jobs"] == attachments
    assert seen["config"] == config
    assert json.loads((root / "worker-status.json").read_text())["state"] == "completed"

    # Check the actual module entry point used by the detached subprocess, too.
    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.runner.InitialRoundRunner", Runner)
    monkeypatch.setattr(sys, "argv", ["worker", *command[3:]])
    runpy.run_path(worker.__file__, run_name="__main__")
    assert seen["attach_jobs"] == attachments


@pytest.mark.parametrize("kind", ["unknown", "invalid", "conflicting", "active"])
def test_detached_recovery_rejects_invalid_or_active_attachment(profile, monkeypatch, kind):
    config = prepare_run(profile)
    root = config.work_dir / config.name
    root.mkdir()
    job = {"state": "submission_uncertain"}
    attachments = {"generate/ppiflow": "known"}
    if kind == "unknown":
        attachments = {"missing": "known"}
    elif kind == "invalid":
        attachments = {"generate/ppiflow": "../invalid"}
    elif kind == "conflicting":
        job["id"] = "other"
    elif kind == "active":
        write_json(root / "launch.json", {"pid": 123})
        write_json(root / "worker-status.json", {"state": "running"})
        monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.automation._alive", lambda *_: True)
    write_json(root / "checkpoint.json", {"jobs": {"generate/ppiflow": job}})

    def unexpected_launch(*args, **kwargs):
        pytest.fail("invalid recovery must not launch a worker")

    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.initial_round.automation.subprocess.Popen", unexpected_launch
    )
    with pytest.raises(ValueError):
        start_round(config, attach_jobs=attachments)


def test_worker_reports_uncertain_submission_as_recoverable(profile, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round import worker
    from opendde_harness.plugin.protein_design.initial_round.services import SubmissionUncertainError

    config = prepare_run(profile)
    root = config.work_dir / config.name
    root.mkdir()
    write_json(root / "worker-config.json", config.model_dump(mode="json"))

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self):
            raise SubmissionUncertainError("generate/ppiflow: resume with --attach-job generate/ppiflow=ID")

    monkeypatch.setattr(worker, "InitialRoundRunner", Runner)
    with pytest.raises(SubmissionUncertainError):
        worker.run(root)
    state = json.loads((root / "worker-status.json").read_text())
    assert state["state"] == "submission_uncertain"
    assert "--attach-job generate/ppiflow=ID" in state["error"]


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


async def test_natural_language_tool_accepts_inline_config_without_yaml(profile, tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.tools.initial_round import InitialRoundTool

    value = prepare_run(profile).model_dump(mode="json")
    profile.unlink()  # The tool must not read a YAML file on this path.
    tool = InitialRoundTool({}, "start")
    launched = []

    def launch(config):
        launched.append(config)
        return {"state": "starting", "run_dir": str(config.work_dir / config.name)}

    monkeypatch.setattr("opendde_harness.plugin.protein_design.tools.initial_round.start_round", launch)
    result = await tool.execute(run_config=value, name="spoken_task", user_confirmed=True)
    assert json.loads(result.model_text)["state"] == "starting"
    assert launched[0].name == "spoken_task"
    assert launched[0].generators["ppiflow"].request == value["generators"]["ppiflow"]["request"]
    assert not profile.exists()
    schema = tool.parameters
    assert "run_config" in schema["properties"]
    assert "pdb_path" not in schema["required"]
    import jsonschema

    jsonschema.validate({"run_config": value, "user_confirmed": True}, schema)


async def test_inline_launch_still_validates_inputs_and_authorization(profile, monkeypatch):
    from opendde_harness.plugin.protein_design.tools.initial_round import InitialRoundTool

    tool = InitialRoundTool({}, "start")
    launched = []
    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.tools.initial_round.start_round", lambda config: launched.append(config)
    )
    value = prepare_run(profile).model_dump(mode="json")
    result = await tool.execute(run_config=value, user_confirmed=False)
    assert "authorization" in json.loads(result.model_text)["error"]
    value["generators"]["ppiflow"]["chains"]["binder"] = []
    result = await tool.execute(run_config=value, user_confirmed=True)
    assert "error" in json.loads(result.model_text) and not launched
    result = await tool.execute(run_config=value, config_path=str(profile), user_confirmed=True)
    assert "not both" in json.loads(result.model_text)["error"]


def test_registered_autonomous_example_resolves_for_yaml_and_run_config(tmp_path, monkeypatch):
    import copy
    from types import SimpleNamespace

    import yaml

    from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig

    examples = Path(__file__).parents[1] / "docs/examples"
    path = examples / "autonomous_round_registered.yaml"
    value = yaml.safe_load(path.read_text())
    original = copy.deepcopy(value)
    registrations = json.loads((examples / "model_services.json").read_text())["plugins"]["config"]["protein-design"][
        "model_services"
    ]
    monkeypatch.setattr(
        "opendde_harness.config.loader.load_config",
        lambda: SimpleNamespace(plugins=SimpleNamespace(config={"protein-design": {"model_services": registrations}})),
    )
    config = prepare_run(value, registrations=registrations)
    assert value == original
    assert InitialRoundConfig.load(path) == config
    assert prepare_run(value) == config
    assert config.evaluation.scoring_backend == "torchscore"
    assert config.evaluation.af3score is None
    assert config.evaluation.torchopt.url.endswith(":8060")
    assert config.evaluation.rosetta.sbatch_script == Path("/lustre/project/apis/rosetta/rosetta-api.sbatch")
    assert config.generators["torchcraft"].service == config.generators["torchcraft"].service.model_validate(
        registrations["torchcraft"]["service"]
    )
    for kind in ("ppiflow", "torchcraft"):
        assert config.generators[kind].request == original["generators"][kind]["request"]
    # A saved, resolved configuration needs no global service metadata to validate.
    snapshot = config.model_dump(mode="json")
    registrations.clear()
    assert InitialRoundConfig.model_validate(snapshot) == config


def test_explicit_service_wins_and_ambiguous_registration_is_rejected(profile):
    import yaml

    value = yaml.safe_load(profile.read_text())
    explicit = value["generators"]["ppiflow"]["service"]
    registrations = {
        "one": {"kind": "ppiflow", "service": {"url": "http://slurm-api.invalid:9999"}},
        "two": {"kind": "ppiflow", "service": {"url": "http://slurm-api.invalid:9998"}},
    }
    assert prepare_run(value, registrations=registrations).generators["ppiflow"].service.url == explicit["url"]
    del value["generators"]["ppiflow"]["service"]
    with pytest.raises(ValueError, match="multiple registered ppiflow"):
        prepare_run(value, registrations=registrations)
    with pytest.raises(ValueError, match="generator services"):
        prepare_run(value, registrations={})
