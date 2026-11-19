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


@pytest.fixture(autouse=True)
def isolated_run_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.initial_round.automation.run_index_path",
        lambda: tmp_path / "instance-runs.json",
    )


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
    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.tools.workflow.prepare_workflow",
        lambda value, **kwargs: prepare_run(Path(value) if isinstance(value, str) else value, **kwargs),
    )
    registry = active_registry()
    tool = registry.build_tool(
        "protein_design_start",
        config={"initial_round_config": str(profile)},
        services=ServiceLocator(workspace=tmp_path, user_id="test", agent_id="test"),
    )
    pdb = tmp_path / "input.pdb"
    pdb.write_text("input")
    launched = []

    def launch(config):
        launched.append(config)
        return {"state": "starting", "run_dir": str(config.work_dir / config.name)}

    monkeypatch.setattr("opendde_harness.plugin.protein_design.tools.workflow.start_round", launch)
    result = await tool.execute(config_path=str(profile), pdb_path=str(pdb), name="run1", user_confirmed=True)
    assert json.loads(result.model_text)["state"] == "starting"
    assert len(launched) == 1 and launched[0].generators["ppiflow"].request["antigen_pdb"] == "/shared/input.pdb"
    assert "protein_design_status" in registry.tool_names()


def test_cli_detach_returns_without_running_foreground(profile, monkeypatch):
    monkeypatch.setattr(
        "opendde_harness.cli.protein_design_commands.start_round",
        lambda *args, **kwargs: {"state": "starting", "run_dir": "/shared/run"},
    )
    monkeypatch.setattr(
        "opendde_harness.cli.protein_design_commands.prepare_workflow",
        lambda value, **kwargs: prepare_run(Path(value) if isinstance(value, str) else value, **kwargs),
    )
    result = CliRunner().invoke(protein_design_app, ["start", "-c", str(profile), "--detach"])
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
    from opendde_harness.plugin.protein_design.tools.workflow import WorkflowTool as InitialRoundTool

    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.tools.workflow.prepare_workflow",
        lambda value, **kwargs: prepare_run(value, **kwargs),
    )
    value = prepare_run(profile).model_dump(mode="json")
    profile.unlink()  # The tool must not read a YAML file on this path.
    tool = InitialRoundTool({}, "start")
    launched = []

    def launch(config):
        launched.append(config)
        return {"state": "starting", "run_dir": str(config.work_dir / config.name)}

    monkeypatch.setattr("opendde_harness.plugin.protein_design.tools.workflow.start_round", launch)
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
    from opendde_harness.plugin.protein_design.tools.workflow import WorkflowTool as InitialRoundTool

    tool = InitialRoundTool({}, "start")
    launched = []
    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.tools.workflow.start_round", lambda config: launched.append(config)
    )
    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.tools.workflow.prepare_workflow",
        lambda value, **kwargs: prepare_run(value, **kwargs),
    )
    value = prepare_run(profile).model_dump(mode="json")
    result = await tool.execute(run_config=value, user_confirmed=False)
    assert "authorize" in json.loads(result.model_text)["error"]
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
    assert config.evaluation.rosetta.backend == "slurm"
    assert config.evaluation.rosetta.sbatch_script is None
    assert config.evaluation.rosetta.url is None
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


def test_resume_uses_saved_snapshot_and_enables_failed_step_retry(profile, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round import automation
    from opendde_harness.plugin.protein_design.tools.workflow import WorkflowTool as InitialRoundTool

    config = prepare_run(profile)
    root = config.work_dir / config.name
    root.mkdir()
    write_json(root / "worker-config.json", config.model_dump(mode="json"))
    seen = []

    def launch(saved, *, retry_failed):
        seen.append((saved, retry_failed))
        return {"state": "starting", "run_dir": str(root)}

    monkeypatch.setattr(automation, "start_round", launch)
    # Global registrations and the original profile are no longer needed for recovery.
    profile.unlink()
    assert automation.resume_round(root)["state"] == "starting"
    assert seen == [(config, True)]
    assert InitialRoundTool({}, "resume").parameters["required"] == ["run_dir"]
    assert InitialRoundTool({}, "resume").external_effects


def test_detached_resume_passes_retry_flag_without_changing_saved_inputs(profile, monkeypatch):
    config = prepare_run(profile)
    root = config.work_dir / config.name
    commands = []

    class Process:
        pid = 12345

        def wait(self):
            return 0

    def launch(command, **kwargs):
        commands.append(command)
        return Process()

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.automation.subprocess.Popen", launch)
    result = start_round(config, retry_failed=True)
    assert result["state"] == "starting"
    assert "--retry-failed" in commands[0]
    assert json.loads((root / "worker-config.json").read_text()) == config.model_dump(mode="json")


def test_run_discovery_survives_new_session_and_finds_older_runs(tmp_path, monkeypatch):
    import socket

    from opendde_harness.plugin.protein_design.initial_round import automation

    index = tmp_path / "instance" / "runs.json"
    monkeypatch.setattr(automation, "run_index_path", lambda: index)
    monkeypatch.setattr(automation, "_alive", lambda *_: True)
    parent = tmp_path / "shared-runs"
    first, older = parent / "current", parent / "older"
    for root in (first, older):
        root.mkdir(parents=True)
        write_json(root / "worker-config.json", {})
        write_json(root / "launch.json", {"pid": 123, "hostname": socket.gethostname()})
        write_json(root / "worker-status.json", {"state": "running"})
    automation.remember_run(first)
    result = automation.list_rounds()
    assert [item["run_dir"] for item in result["runs"]] == [str(first)]
    assert result["runs"][0]["state"] == "running"
    assert {item["run_dir"] for item in automation.list_rounds(parent)["runs"]} == {str(first), str(older)}
    # An inaccessible old run does not hide other tasks.
    (first / "launch.json").unlink()
    result = automation.list_rounds(parent)
    assert {item["state"] for item in result["runs"]} == {"unavailable", "running"}


def test_run_discovery_does_not_claim_remote_worker_is_alive(tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round import automation

    root = tmp_path / "remote"
    root.mkdir()
    write_json(root / "launch.json", {"pid": 123, "hostname": "different-login-node"})
    write_json(root / "worker-status.json", {"state": "running"})
    monkeypatch.setattr(automation, "run_index_path", lambda: tmp_path / "index.json")
    automation.remember_run(root)
    assert automation.list_rounds()["runs"][0]["state"] == "remote_worker_unverified"


def test_workflow_is_indexed_before_worker_launch_and_fails_closed(profile, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round import automation

    config = prepare_run(profile)
    calls = []

    def launch(saved, **kwargs):
        assert automation.run_index_path().is_file()
        calls.append(saved)
        return {"state": "starting"}

    monkeypatch.setattr(automation, "_start_round", launch)
    assert automation.start_round(config)["state"] == "starting"
    assert calls == [config]

    def no_index(root):
        raise OSError("index unavailable")

    monkeypatch.setattr(automation, "remember_run", no_index)
    with pytest.raises(OSError, match="index unavailable"):
        automation.start_round(config)
    assert calls == [config]


def test_worker_persists_cdr_question_without_reporting_inference_failure(profile, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round import worker
    from opendde_harness.plugin.protein_design.initial_round.region_input import AwaitingRegionsError

    config = prepare_run(profile)
    root = config.work_dir / config.name
    root.mkdir()
    write_json(root / "worker-config.json", config.model_dump(mode="json"))
    pending = {
        "question_id": "a" * 64,
        "question": "Provide CDR ranges for this candidate",
        "candidate_id": "candidate",
    }

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self):
            raise AwaitingRegionsError(pending)

    monkeypatch.setattr(worker, "InitialRoundRunner", Runner)
    worker.run(root)
    assert json.loads((root / "worker-status.json").read_text()) == {
        "state": "awaiting_regions",
        "pending_regions": pending,
    }


async def test_region_resolution_tool_does_not_invent_user_answer(tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.tools.workflow import WorkflowTool as InitialRoundTool

    calls = []
    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.tools.workflow.resume_round", lambda root: calls.append(root)
    )
    tool = InitialRoundTool({}, "regions")
    result = await tool.execute(run_dir=str(tmp_path), question_id="a" * 64, cdr_position="B5-6")
    assert "user must supply/confirm" in json.loads(result.model_text)["error"]
    assert not calls


def test_detached_resume_accepts_status_selector_fix_but_preserves_scientific_guard(profile, monkeypatch):
    config = prepare_run(profile)
    calls = []

    class Process:
        pid = 12345

        def wait(self):
            return 0

    def launch(command, **kwargs):
        calls.append(command)
        return Process()

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.automation.subprocess.Popen", launch)
    start_round(config)
    root = config.work_dir / config.name
    write_json(root / "worker-status.json", {"state": "interrupted"})
    config.evaluation.torchfold.status_field = "state"
    assert start_round(config)["state"] == "starting"
    assert len(calls) == 2
    assert json.loads((root / "worker-config.json").read_text())["evaluation"]["torchfold"]["status_field"] == "state"
    config.evaluation.min_refold_iptm = 0.6
    with pytest.raises(ValueError, match="configuration changed"):
        start_round(config)
    assert len(calls) == 2


def test_resume_normalizes_legacy_snapshot_defaults_before_launch_comparison(profile, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round import automation

    config = prepare_run(profile)
    root = config.work_dir / config.name
    root.mkdir()
    legacy = config.model_dump(mode="json")
    legacy["generators"]["ppiflow"]["service"].pop("startup_poll_interval")
    legacy["evaluation"]["torchfold"].pop("slurm_command_timeout")
    write_json(root / "worker-config.json", legacy)
    calls = []

    class Process:
        pid = 12345

        def wait(self):
            return 0

    def popen(command, **kwargs):
        calls.append(command)
        return Process()

    monkeypatch.setattr(automation.subprocess, "Popen", popen)
    assert automation.resume_round(root)["state"] == "starting"
    assert len(calls) == 1
    assert "--retry-failed" in calls[0]
    saved = json.loads((root / "worker-config.json").read_text())
    assert saved == config.model_dump(mode="json")
    changed = config.model_copy(deep=True)
    changed.evaluation.min_refold_iptm = 0.1
    with pytest.raises(ValueError, match=r"evaluation\.min_refold_iptm") as error:
        automation.start_round(changed)
    assert "0.1" not in str(error.value)  # Diagnostics contain paths, not configuration values.
    assert len(calls) == 1
