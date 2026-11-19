"""Persistent defaults are resolved before new-target task validation and launch."""

import copy
import csv
import json
from pathlib import Path

import pytest

from opendde_harness.plugin.protein_design.core.model_registry import ProteinModelRegistry
from opendde_harness.plugin.protein_design.initial_round.storage import SharedFiles
from opendde_harness.plugin.protein_design.initial_round.task_inputs import prepare_design_task
from tests._slurm_api import mock_slurm_apis  # noqa: F401
from tests.test_protein_api_startup import api  # noqa: F401
from tests.test_protein_design_initial_round import autonomous_config, fake_services, pdb, round_config  # noqa: F401

pytestmark = [pytest.mark.in_memory_transport, pytest.mark.usefixtures("mock_slurm_apis")]


@pytest.fixture
def settings(round_config):
    from opendde_harness.plugin.protein_design.core.model_requests import TARGET_REQUEST_FIELDS

    value = autonomous_config(round_config).model_dump(mode="json")
    generators = value.pop("generators")
    registrations = {}
    for kind, generator in generators.items():
        registrations[kind] = {"kind": kind, "service": generator.pop("service")}
        generator["request"] = {k: v for k, v in generator["request"].items() if k not in TARGET_REQUEST_FIELDS}
        generator["fold_template"] = None
    for kind in ("torchscore", "torchfold", "rosetta"):
        registrations[kind] = {"kind": kind, "service": value["evaluation"].pop(kind)}
    value["evaluation"]["rosetta"] = {"backend": "api", **registrations["rosetta"]["service"]}
    value.pop("name")
    return {
        "model_services": registrations,
        "workflow_defaults": value,
        "design_profiles": {"vhh": {"generators": generators}},
        "target_adapters": {
            "torchcraft": {
                "hotspot_numbering": "sequence_0_based",
                "csv_id_column": "id",
                "csv_sequence_column": "sequence",
            }
        },
    }


@pytest.fixture
def task(tmp_path):
    return {
        "pdb_path": str(pdb(tmp_path / "new.pdb")),
        "target_chain": "A",
        "design_type": "vhh",
        "hotspots": [1, 2],
        "initial_samples": 10,
        "framework_dir": "/shared/frameworks",
        "framework_cif_path": "/shared/framework.cif.pdb",
    }


def test_new_target_uses_static_settings_and_creates_fresh_csv(task, settings):
    before = copy.deepcopy(settings)
    settings["model_services"]["ppiflow"]["request_defaults"] = {
        "num_timesteps": 77,
        "model_weights": "/shared/static/weights.ckpt",
    }
    config = prepare_design_task(task, settings)
    ppiflow = config.generators["ppiflow"].request
    craft = config.generators["torchcraft"].request
    assert ppiflow["samples_per_target"] == craft["num_designs"] == 5
    assert "model_weights" not in ppiflow
    assert "num_timesteps" not in ppiflow
    assert ppiflow["specified_hotspots"] == "A1,A2"
    assert craft["hotspot_indices"] == "0,1"
    files = SharedFiles(config.path_mappings)
    rows = list(csv.DictReader(files.local(craft["target_full_seq"]).open()))
    assert rows == [{"id": "target", "sequence": "GGGGGG"}]
    assert craft["target_full_seq"] != "/shared/target.csv"
    assert config.evaluation.dockq_python == Path(before["workflow_defaults"]["evaluation"]["dockq_python"])
    assert settings["design_profiles"] == before["design_profiles"]
    again = prepare_design_task(task, settings)
    assert again.model_dump(mode="json") == config.model_dump(mode="json")


def test_task_overrides_defaults_and_never_changes_persistent_values(task, settings):
    settings["model_services"]["ppiflow"]["request_defaults"] = {"num_timesteps": 77}
    task.update(
        generators=["ppiflow"],
        generator_overrides={"ppiflow": {"request": {"num_timesteps": 42}}},
        framework_dir="/shared/new-framework",
    )
    config = prepare_design_task(task, settings)
    assert config.generators["ppiflow"].request["num_timesteps"] == 42
    assert config.generators["ppiflow"].request["samples_per_target"] == 10
    assert config.generators["ppiflow"].request["framework_dir"] == "/shared/new-framework"
    assert settings["model_services"]["ppiflow"]["request_defaults"] == {"num_timesteps": 77}


def test_new_target_chain_and_hotspots_replace_all_old_target_information(task, settings):
    from biotite.structure.io.pdb import PDBFile

    atoms = PDBFile.read(task["pdb_path"]).get_structure(model=1)
    atoms.chain_id[atoms.chain_id == "A"] = "D"
    atoms.res_id[atoms.chain_id == "D"] += 100
    writer = PDBFile()
    writer.set_structure(atoms)
    writer.write(task["pdb_path"])
    task.update(target_chain="D", hotspots=["D101", "D103"])
    config = prepare_design_task(task, settings)
    assert config.generators["ppiflow"].request["antigen_chain"] == "D"
    assert config.generators["ppiflow"].request["specified_hotspots"] == "D101,D103"
    assert config.generators["torchcraft"].request["hotspot_indices"] == "0,2"
    assert config.generators["ppiflow"].chains.target == ["D"]
    assert "D" in config.generators["ppiflow"].chains.refold_map


@pytest.mark.parametrize(
    "change",
    [
        {"hotspots": [999]},
        {"hotspots": ["Z1"]},
        {"target_chain": "Z"},
        {"initial_samples": 1},
        {"generator_samples": {"ppiflow": 10, "torchcraft": 10}},
    ],
)
def test_bad_new_task_inputs_are_rejected_before_submission(task, settings, change):
    task.update(change)
    with pytest.raises(ValueError):
        prepare_design_task(task, settings)


def test_torchcraft_indexing_uses_confirmed_deployment_when_not_overridden(task, settings):
    settings.pop("target_adapters")
    config = prepare_design_task(task, settings)
    assert config.generators["torchcraft"].request["hotspot_indices"] == "1,2"


def test_old_target_inputs_cannot_be_persisted_as_defaults(settings):
    settings["design_profiles"]["vhh"]["generators"]["torchcraft"]["request"]["target_full_seq"] = "/shared/old.csv"
    from opendde_harness.plugin.protein_design.initial_round.defaults import validate_settings

    with pytest.raises(ValueError, match="target-specific"):
        validate_settings(settings)


def test_unmatched_sequence_csv_uses_native_pdb_fallback(task, settings, tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("id,sequence\nold,AAAAAA\n")
    task["target_sequence_csv"] = str(path)
    config = prepare_design_task(task, settings)
    assert "target_full_seq" not in config.generators["torchcraft"].request


async def test_compact_tool_launches_without_yaml_or_repeated_static_inputs(task, settings, monkeypatch):
    from opendde_harness.plugin.protein_design.tools.initial_round import InitialRoundTool

    launched = []
    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.tools.initial_round.start_round",
        lambda config: launched.append(config) or {"run_dir": str(config.work_dir / config.name), "state": "starting"},
    )
    result = await InitialRoundTool(settings, "start").execute(task=task, user_confirmed=True)
    assert "run_dir" in json.loads(result.model_text), result.model_text
    assert len(launched) == 1
    assert launched[0].generators["ppiflow"].request["framework_dir"]
    result = await InitialRoundTool(settings, "start").execute(task=task, user_confirmed=False)
    assert "error" in json.loads(result.model_text)
    assert len(launched) == 1


async def test_complete_compact_workflow_reuses_registered_services(task, settings, fake_services):
    from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner

    task["initial_samples"] = 2
    config = prepare_design_task(task, settings)
    result = await InitialRoundRunner(config, transport=fake_services[0]).run()
    assert result["accepted"] == 2


async def test_standalone_request_defaults_are_applied_and_task_values_win(tmp_path):
    import httpx

    submitted = []

    def handler(request):
        if request.method == "POST":
            submitted.append(json.loads(request.content))
            return httpx.Response(202, json={"design_id": "native"})
        return httpx.Response(200, json={"ok": True})

    registry = ProteinModelRegistry(
        {
            "craft": {
                "kind": "torchcraft",
                "service": {"url": "https://craft.test"},
                "request_defaults": {"mode": "vhh", "seed": 42},
            }
        },
        tmp_path,
        transport=httpx.MockTransport(handler),
    )
    await registry.submit(
        "craft",
        "task",
        {
            "name": "new",
            "pdb_path": "/shared/new.pdb",
            "hotspot_indices": "0",
            "target_full_seq": "/shared/new.csv",
            "framework_cif_path": "/shared/framework.cif.pdb",
            "seed": 7,
        },
    )
    assert submitted[0]["seed"] == 7


def test_service_environment_is_persistent_but_not_exposed_in_catalogue(tmp_path):
    registry = ProteinModelRegistry(
        {
            "score": {
                "kind": "torchscore",
                "service": {
                    "url": "https://score.test",
                    "environment": {"TORCHSCORE_ROOT": "/private/source", "EXAMPLE_SECRET": "do-not-show"},
                },
            }
        },
        tmp_path,
    )
    text = json.dumps(registry.catalogue())
    assert "do-not-show" not in text and "/private/source" not in text
    assert "TORCHSCORE_ROOT" in text


async def test_service_environment_is_exported_only_to_sbatch(monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.api_startup import slurm_command
    from opendde_harness.plugin.protein_design.initial_round.config import Service

    seen = []

    class Process:
        returncode = 0

        async def communicate(self):
            return b"123\n", b""

    async def execute(*args, **kwargs):
        seen.append((args, kwargs))
        return Process()

    monkeypatch.setattr("asyncio.create_subprocess_exec", execute)
    service = Service(url="https://flow.test", environment={"PPIFLOW_ROOT": "/shared/source"})
    await slurm_command(service, "sbatch", "script.sbatch")
    await slurm_command(service, "squeue", "--jobs", "123")
    assert seen[0][1]["env"]["PPIFLOW_ROOT"] == "/shared/source"
    assert "env" not in seen[1][1]
    assert "environment" not in Service(url="https://flow.test").model_dump()


@pytest.mark.parametrize("defaults", [{"missing_field": 1}, {"antigen_pdb": "/shared/old.pdb"}, {"num_timesteps": -1}])
def test_invalid_request_defaults_are_rejected_at_registration(tmp_path, defaults):
    with pytest.raises(ValueError):
        ProteinModelRegistry(
            {"flow": {"kind": "ppiflow", "service": {"url": "https://flow.test"}, "request_defaults": defaults}},
            tmp_path,
        )


def test_target_chain_collision_is_normalized_and_recorded(task, settings):
    task.update(target_chain="B", hotspots=[1])
    config = prepare_design_task(task, settings)
    assert config.generators["ppiflow"].request["antigen_chain"] != "B"
    provenance = json.loads((config.work_dir / config.name / "inputs/target-provenance.json").read_text())
    assert provenance["source_chain"] == "B"
    assert provenance["normalized_chain"] == config.generators["ppiflow"].chains.target[0]


def test_persistent_defaults_do_not_introduce_global_budgets(task, settings):
    config = prepare_design_task(task, settings)
    assert (
        not {"max_rounds", "max_samples", "batch_size", "max_batch_size", "patience"}
        & config.autonomous.model_dump().keys()
    )


async def test_service_environment_is_part_of_live_allocation_binding(api):
    from opendde_harness.plugin.protein_design.initial_round.api_startup import ApiStartupError, validate_binding
    from opendde_harness.plugin.protein_design.initial_round.services import ServiceConnection

    config, commands, _, transport = api
    config.environment = {"PPIFLOW_ROOT": "/shared/source"}
    client = ServiceConnection(config, transport=transport)
    try:
        await client.get("/health")
        assert "--export=ALL" in commands[0]
        record = client.allocations[-1]
        validate_binding(client.config, client.client.base_url, record)
        changed = client.config.model_copy(update={"environment": {"PPIFLOW_ROOT": "/shared/other"}})
        with pytest.raises(ApiStartupError, match="bound"):
            validate_binding(changed, client.client.base_url, record)
    finally:
        await client.close()


@pytest.mark.parametrize("design", ["antibody", "minibinder"])
def test_other_design_types_use_their_own_static_profile(task, settings, design):
    profile = copy.deepcopy(settings["design_profiles"]["vhh"])
    flow = profile["generators"]["ppiflow"]
    if design == "antibody":
        profile["generators"].pop("torchcraft")
        flow["request"].update(mode="antibody", light_chain="L", cdr_length="CDRH3,10-20,CDRL3,9-11")
        flow["chains"]["binder"].append("L")
        flow["chains"]["refold_map"]["L"] = "Z"
    else:
        flow["request"] = {
            "mode": "binder",
            "target_chain": "A",
            "binder_chain": "B",
            "samples_min_length": 50,
            "samples_max_length": 70,
        }
        profile["generators"]["torchcraft"]["request"] = {"mode": "mini_binder", "design_length": 60}
    settings["design_profiles"][design] = profile
    task["design_type"] = design
    config = prepare_design_task(task, settings)
    assert config.generators["ppiflow"].request["mode"] == ("antibody" if design == "antibody" else "binder")
    assert (
        sum(g.request.get("samples_per_target", g.request.get("num_designs", 1)) for g in config.generators.values())
        == task["initial_samples"]
    )


def test_partial_flow_keeps_task_fixed_points_in_addition_to_energy_fixed_points(task, settings):
    from opendde_harness.plugin.protein_design.initial_round.autonomous import _partial_request
    from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner

    task.update(generators=["ppiflow"], fixed_positions="B3")
    config = prepare_design_task(task, settings)
    runner = InitialRoundRunner(config)
    candidate = {
        "source": "ppiflow",
        "chains": {"A": "GGGGGG", "B": "AAAAAA"},
        "generated_structure": runner.files.remote(Path(task["pdb_path"])),
        "interface_fixed_positions": {"B": [1]},
        "generation_request": config.generators["ppiflow"].request,
    }
    request, regions = _partial_request(runner, candidate, "partial")
    assert request["fixed_positions"] == "B1,B3"
    assert regions["mutable_positions"] == {"B": [2, 4, 5, 6]}


def test_named_model_schema_does_not_require_persisted_defaults(tmp_path):
    from opendde_harness.plugin.protein_design.tools.models import ProteinModelSubmitTool

    registry = ProteinModelRegistry(
        {
            "craft": {
                "kind": "torchcraft",
                "service": {"url": "https://craft.test"},
                "request_defaults": {"mode": "vhh"},
            }
        },
        tmp_path,
    )
    schema = ProteinModelSubmitTool(registry, "torchcraft").parameters
    assert "mode" not in schema["properties"]["request"]["required"]
    assert schema["properties"]["request"]["properties"]["mode"]["default"] == "vhh"


def test_explicit_service_replaces_persistent_connection_settings():
    from opendde_harness.plugin.protein_design.initial_round.defaults import workflow_values

    defaults = {
        "workflow_defaults": {
            "generators": {"ppiflow": {"service": {"url": "https://old.test", "environment": {"OLD": "value"}}}},
            "evaluation": {"torchscore": {"url": "https://old.test", "sbatch_script": "/old/script"}},
        }
    }
    value = workflow_values(
        {
            "generators": {"ppiflow": {"service": {"url": "https://new.test"}}},
            "evaluation": {"torchscore": {"url": "https://new.test"}},
        },
        defaults,
    )
    assert value["generators"]["ppiflow"]["service"] == {"url": "https://new.test"}
    assert value["evaluation"]["torchscore"] == {"url": "https://new.test"}


def test_old_framework_and_fixed_points_are_not_reused(task, settings):
    request = settings["design_profiles"]["vhh"]["generators"]["ppiflow"]["request"]
    request.update(framework_dir="/shared/old-framework", fixed_positions="B2", cdr_position="B1-2")
    settings["model_services"]["ppiflow"]["request_defaults"] = {"fixed_positions": "B4"}
    config = prepare_design_task(task, settings)
    resolved = config.generators["ppiflow"].request
    assert resolved["framework_dir"] == task["framework_dir"]
    assert not resolved.get("fixed_positions")
    assert not resolved.get("cdr_position")


@pytest.mark.parametrize("field", ["framework_dir", "fixed_positions", "cdr_position"])
def test_persistent_experimental_choices_are_rejected(field):
    from opendde_harness.plugin.protein_design.core.model_requests import validate_request_defaults

    with pytest.raises(ValueError, match="each task"):
        validate_request_defaults("ppiflow", {field: "B2"})


@pytest.mark.parametrize("source", ["ppiflow", "torchcraft"])
async def test_prompt_can_select_either_generator_without_submitting_other(task, settings, fake_services, source):
    from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner

    task.update(generators=[source], initial_samples=1)
    config = prepare_design_task(task, settings)
    assert set(config.generators) == {source}
    result = await InitialRoundRunner(config, transport=fake_services[0]).run()
    assert result["accepted"] == 1
    candidates = json.loads((config.work_dir / config.name / "candidates.json").read_text())
    assert {candidate["source"] for candidate in candidates} == {source}


def test_torchcraft_only_profile_does_not_need_ppiflow(task, settings):
    settings["design_profiles"]["vhh"]["generators"].pop("ppiflow")
    settings["model_services"].pop("ppiflow")
    task["generators"] = ["torchcraft"]
    config = prepare_design_task(task, settings)
    assert set(config.generators) == {"torchcraft"}
    from opendde_harness.plugin.protein_design.initial_round.autonomous import _partial_request
    from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner

    with pytest.raises(ValueError, match="Partial Flow unavailable"):
        _partial_request(InitialRoundRunner(config), {}, "refine")


@pytest.mark.parametrize("source", ["ppiflow", "torchcraft"])
def test_minibinder_prompt_selects_generator_and_native_mode(task, settings, source):
    profile = copy.deepcopy(settings["design_profiles"]["vhh"])
    profile["generators"]["ppiflow"]["request"] = {
        "mode": "binder",
        "target_chain": "A",
        "binder_chain": "B",
        "samples_min_length": 50,
        "samples_max_length": 70,
    }
    profile["generators"]["torchcraft"]["request"] = {"mode": "mini_binder", "design_length": 60}
    settings["design_profiles"]["minibinder"] = profile
    task.update(design_type="minibinder", generators=[source])
    config = prepare_design_task(task, settings)
    assert set(config.generators) == {source}
    assert config.generators[source].request["mode"] == ("binder" if source == "ppiflow" else "mini_binder")


@pytest.mark.parametrize("value", ["", "   ", None])
def test_empty_dockq_uses_deployment_default(value):
    from opendde_harness.plugin.protein_design.initial_round.config import Evaluation
    from opendde_harness.plugin.protein_design.initial_round.defaults import workflow_values

    config = Evaluation(dockq_python=value, af3score={"url": "https://score.test"})
    assert str(config.dockq_python) == "/lustre/grp/cmclab/leiy/conda_envs/ppi_env/bin/python"
    merged = workflow_values(
        {"evaluation": {"dockq_python": value}},
        {"workflow_defaults": {"evaluation": {"dockq_python": "/shared/custom/python"}}},
    )
    assert merged["evaluation"]["dockq_python"] == "/shared/custom/python"


async def test_historical_yaml_is_not_implicitly_loaded(tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.tools.initial_round import InitialRoundTool

    path = tmp_path / "old.yaml"
    path.write_text("this is not a workflow")
    result = json.loads(
        (await InitialRoundTool({"initial_round_config": str(path)}, "start").execute(user_confirmed=True)).model_text
    )
    assert "never loaded automatically" in result["error"]


def test_unspecified_generator_parameters_are_omitted_from_payload():
    from opendde_harness.plugin.protein_design.core.model_requests import validate_request

    value = validate_request(
        "torchcraft",
        {
            "mode": "mini_binder",
            "name": "new",
            "pdb_path": "/shared/x.pdb",
            "hotspot_indices": "0",
            "target_full_seq": "/shared/x.csv",
            "design_length": 50,
            "num_designs": 100,
        },
        omit_defaults=True,
    )
    assert "seed" not in value
    assert value["num_designs"] == 100


def test_bundled_deployment_is_the_only_automatic_fallback(task, tmp_path):
    import yaml

    from opendde_harness.plugin.protein_design.initial_round.defaults import (
        DEFAULT_DEPLOYMENT_CONFIG,
        load_deployment_defaults,
    )

    task["generators"] = ["torchcraft"]
    task["generator_overrides"] = {
        "torchcraft": {"chains": {"target": ["A"], "binder": ["B"], "refold_map": {"A": "A", "B": "B"}}}
    }
    config = prepare_design_task(
        task,
        {
            "workflow_defaults": {
                "work_dir": str(tmp_path / "runs"),
                "path_mappings": [{"local": str(tmp_path), "remote": "/shared"}],
            }
        },
    )
    request = config.generators["torchcraft"].request
    assert request["hotspot_indices"] == "1,2"
    assert "target_full_seq" not in request
    assert str(config.evaluation.dockq_python) == "/lustre/grp/cmclab/leiy/conda_envs/ppi_env/bin/python"
    assert (
        str(config.generators["torchcraft"].service.sbatch_script)
        == "/lustre/grp/cmclab/leiy/torchcraft_api/torchcraft-api.sbatch"
    )
    assert len(load_deployment_defaults({})["model_services"]) == 9
    assert yaml.safe_load(DEFAULT_DEPLOYMENT_CONFIG.read_text()) == yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docs/examples/cluster_deployment.yaml").read_text()
    )


def test_native_csv_preserves_full_sequence_and_matches_prepared_pdb(task, settings, tmp_path):
    settings["target_adapters"]["torchcraft"] = {
        "hotspot_numbering": "pdb_residue_id",
        "csv_id_column": "target",
        "csv_sequence_column": "real_sequence",
        "csv_optional": True,
    }
    path = tmp_path / "full.csv"
    path.write_text("target,real_sequence\nnew,MGGGGGGK\nother,AAAAAA\n")
    task.update(generators=["torchcraft"], target_sequence_csv=str(path))
    config = prepare_design_task(task, settings)
    request = config.generators["torchcraft"].request
    rows = list(csv.DictReader(SharedFiles(config.path_mappings).local(request["target_full_seq"]).open()))
    assert rows == [{"target": Path(request["pdb_path"]).stem, "real_sequence": "MGGGGGGK"}]
    assert request["hotspot_indices"] == "1,2"


def test_deployment_override_never_accepts_a_historical_run_config(tmp_path):
    from opendde_harness.plugin.protein_design.initial_round.defaults import load_deployment_defaults

    path = tmp_path / "old-task.yaml"
    path.write_text("name: old-task\ngenerators: {}\n")
    with pytest.raises(ValueError, match="not a historical task"):
        load_deployment_defaults({"deployment_config": str(path)})
    path.write_text("workflow_defaults:\n  evaluation:\n    dockq_python: /shared/custom/python\n")
    assert (
        load_deployment_defaults({"deployment_config": str(path)})["workflow_defaults"]["evaluation"]["dockq_python"]
        == "/shared/custom/python"
    )


async def test_launch_tool_uses_deployment_yaml_for_incomplete_run_config(task, tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.tools.initial_round import InitialRoundTool

    launched = []
    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.tools.initial_round.start_round",
        lambda config: launched.append(config) or {"state": "starting", "run_dir": str(config.work_dir / config.name)},
    )
    tool = InitialRoundTool(
        {
            "workflow_defaults": {
                "work_dir": str(tmp_path / "runs"),
                "path_mappings": [{"local": str(tmp_path), "remote": "/shared"}],
            }
        },
        "start",
    )
    result = json.loads(
        (
            await tool.execute(
                user_confirmed=True,
                run_config={
                    "name": "fallback-test",
                    "design_type": "vhh",
                    "evaluation": {"dockq_python": ""},
                    "generators": {
                        "ppiflow": {
                            "chains": {"target": ["A"], "binder": ["B"], "refold_map": {"A": "A", "B": "B"}},
                            "request": {
                                "mode": "nanobody",
                                "antigen_pdb": "/shared/new.pdb",
                                "antigen_chain": "A",
                                "heavy_chain": "B",
                                "framework_dir": "/shared/frameworks",
                                "samples_per_target": 100,
                            },
                        }
                    },
                },
            )
        ).model_text
    )
    assert result["state"] == "starting", result
    assert str(launched[0].evaluation.dockq_python) == "/lustre/grp/cmclab/leiy/conda_envs/ppi_env/bin/python"
    assert (
        str(launched[0].generators["ppiflow"].service.sbatch_script)
        == "/lustre/grp/cmclab/leiy/ppiflow_api/ppiflow-api.sbatch"
    )


async def test_public_workflow_task_uses_two_stage_planner_and_persistent_defaults(task, settings, monkeypatch):
    from opendde_harness.plugin.protein_design.tools.workflow import WorkflowTool

    settings["workflow_defaults"].pop("autonomous")
    launched = []
    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.tools.workflow.start_round",
        lambda config: launched.append(config) or {"run_dir": str(config.work_dir / config.name)},
    )
    tool = WorkflowTool(settings, "start")
    assert "task" in tool.parameters["properties"]
    result = json.loads((await tool.execute(task=task, user_confirmed=True)).model_text)
    assert "error" not in result, result
    assert launched[0].autonomous.decision_policy == "planner"
    assert launched[0].autonomous.planner_layout == "two_stage"
    assert launched[0].evaluation.scoring_backend == "torchscore"
    assert launched[0].generators["ppiflow"].request["framework_dir"] == task["framework_dir"]
    result = json.loads((await tool.execute(task=task, user_confirmed=False)).model_text)
    assert "error" in result
    assert len(launched) == 1
    result = json.loads((await tool.execute(task=task, run_config={}, user_confirmed=True)).model_text)
    assert "mutually exclusive" in result["error"]
    assert len(launched) == 1
