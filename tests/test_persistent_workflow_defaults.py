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

    value = autonomous_config(round_config, max_samples=100).model_dump(mode="json")
    generators = value.pop("generators")
    registrations = {}
    for kind, generator in generators.items():
        registrations[kind] = {"kind": kind, "service": generator.pop("service")}
        generator["request"] = {k: v for k, v in generator["request"].items() if k not in TARGET_REQUEST_FIELDS}
        generator["fold_template"] = None
    for kind in ("torchscore", "torchfold", "rosetta"):
        registrations[kind] = {"kind": kind, "service": value["evaluation"].pop(kind)}
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
    assert ppiflow["model_weights"] == "/shared/static/weights.ckpt"
    assert ppiflow["num_timesteps"] == 77
    assert ppiflow["specified_hotspots"] == "A1,A2"
    assert craft["hotspot_indices"] == "0,1"
    files = SharedFiles(config.path_mappings)
    rows = list(csv.DictReader(files.local(craft["target_full_seq"]).open()))
    assert rows == [{"id": config.name, "sequence": "GGGGGG"}]
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


def test_torchcraft_indexing_is_required_once_not_guessed(task, settings):
    settings.pop("target_adapters")
    with pytest.raises(ValueError):
        prepare_design_task(task, settings)


def test_old_target_inputs_cannot_be_persisted_as_defaults(settings):
    settings["design_profiles"]["vhh"]["generators"]["torchcraft"]["request"]["target_full_seq"] = "/shared/old.csv"
    from opendde_harness.plugin.protein_design.initial_round.defaults import validate_settings

    with pytest.raises(ValueError, match="target-specific"):
        validate_settings(settings)


def test_explicit_sequence_csv_cannot_silently_reuse_old_target(task, settings, tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("id,sequence\nold,AAAAAA\n")
    task["target_sequence_csv"] = str(path)
    with pytest.raises(ValueError, match="does not match"):
        prepare_design_task(task, settings)


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
                "request_defaults": {"mode": "vhh", "framework_cif_path": "/shared/framework.pdb", "seed": 42},
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
            "seed": 7,
        },
    )
    assert submitted[0]["framework_cif_path"] == "/shared/framework.pdb"
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


def test_explicit_budget_conflicts_are_not_silently_changed(task, settings):
    settings["workflow_defaults"]["autonomous"]["max_samples"] = 2
    with pytest.raises(ValueError, match="exceed"):
        prepare_design_task(task, settings)
    assert settings["workflow_defaults"]["autonomous"]["max_samples"] == 2


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
                "request_defaults": {"mode": "vhh", "framework_cif_path": "/shared/framework.pdb"},
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
