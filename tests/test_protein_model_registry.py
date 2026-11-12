"""Model tools expose real plugin functions without blocking a chat on Slurm."""

from __future__ import annotations

import json

import httpx
import pytest

from opendde_harness.agent.tools.registry import ToolRegistry
from opendde_harness.plugin import active_registry
from opendde_harness.plugin.context import ServiceLocator
from opendde_harness.plugin.protein_design.core.model_registry import ProteinModelRegistry
from opendde_harness.plugin.protein_design.core.model_requests import AF3ScoreRequest
from opendde_harness.plugin.protein_design.initial_round.config import Evaluation
from opendde_harness.plugin.protein_design.tools.models import ProteinModelTool

pytestmark = pytest.mark.in_memory_transport


@pytest.fixture
def registration():
    return {"af3": {"kind": "af3score", "service": {"url": "https://af3.test"}, "allowed_input_roots": ["/shared"]}}


@pytest.fixture
def api():
    requests = []
    remote = {"status": "PENDING", "exit_code": None}

    def handler(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        if request.method == "POST":
            return httpx.Response(
                202, json={"task_id": "job123", "design_id": "design123", "fold_id": "fold123", "slurm_job_id": "42"}
            )
        if request.url.path.endswith("/results"):
            return httpx.Response(200, json={"files": ["/shared/outputs/input.json"]})
        if request.url.path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": [{"iptm": 0.85}]})
        return httpx.Response(200, json=remote)

    return httpx.MockTransport(handler), requests, remote


@pytest.mark.parametrize(
    "mode,extra,expected",
    [
        ("pdb", {}, True),
        ("pdb", {"run_data_pipeline": True, "run_inference": True}, True),
        ("prepared", {"json_dir": "/shared/json", "h5_dir": "/shared/h5", "run_data_pipeline": True, "run_inference": True}, False),
        ("metrics", {"af3score_output_dir": "/shared/out", "run_data_pipeline": False, "run_inference": False}, True),
    ],
)
def test_af3score_modes_distinguish_artifacts_and_metrics(mode, extra, expected):
    request = {"mode": mode, **extra}
    if mode != "prepared":
        request["input_pdb_dir"] = "/shared/pdbs"
    value = AF3ScoreRequest.model_validate(request)
    assert value.produces_metrics is expected
    # Every supported mode writes the TorchFold fold input JSON.
    assert value.write_fold_input_json_file is True


@pytest.mark.parametrize(
    "changes",
    [
        {"num_samples": 21},
        {"num_samples": 0},
        {"run_inference": False, "run_data_pipeline": False},
        {"run_inference": "false"},
        {"unexpected": True},
        {"mode": "prepared"},
    ],
)
def test_invalid_af3score_controls_fail_before_submission(changes):
    with pytest.raises(ValueError):
        AF3ScoreRequest.model_validate({"input_pdb_dir": "/shared/pdbs", **changes})


@pytest.mark.parametrize("run_inference", [True])
def test_af3score_pipeline_always_writes_fold_json(run_inference):
    """run_data_pipeline=true 必须强制写 fold JSON，与 run_inference 独立。

    这样 pdb 推理模式(pipeline=true, inference=true)能同时拿到分数 + fold 输入，
    供后续 TorchFold refold 使用；用户显式传 write_fold_input_json_file=False 也被覆盖。
    """
    value = AF3ScoreRequest.model_validate(
        {
            "input_pdb_dir": "/shared/pdbs",
            "run_data_pipeline": run_inference,
            "run_inference": run_inference,
            "write_fold_input_json_file": False,
        }
    )
    assert value.write_fold_input_json_file is True


def test_af3score_pipeline_only_is_rejected_outside_metrics():
    """pdb/prepared 模式下 run_data_pipeline=true + run_inference=false 不再允许(仅 metrics 可)。"""
    with pytest.raises(ValueError, match="run_inference=true is required"):
        AF3ScoreRequest.model_validate(
            {"input_pdb_dir": "/shared/pdbs", "run_data_pipeline": True, "run_inference": False}
        )


def test_af3score_no_pipeline_fast_scoring_always_writes_fold_json():
    """false/true 快速评分模式也强制写 fold JSON(每种模式都输出 TorchFold fold JSON)。"""
    value = AF3ScoreRequest.model_validate(
        {"input_pdb_dir": "/shared/pdbs", "run_data_pipeline": False, "run_inference": True, "write_fold_input_json_file": False}
    )
    assert value.write_fold_input_json_file is True


async def test_fast_scoring_submits_controls_and_returns_without_polling(registration, api, tmp_path):
    transport, requests, _ = api
    registry = ProteinModelRegistry(registration, tmp_path, transport=transport)
    result = await registry.submit("af3", "fast", {"input_pdb_dir": "/shared/pdbs"})
    assert result["state"] == "submitted" and result["remote_id"] == "job123"
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["run_inference"] and not body["run_data_pipeline"] and body["init_guess"]
    assert body["seed"] == 10 and body["num_workers"] == 4
    assert await registry.submit("af3", "fast", {"input_pdb_dir": "/shared/pdbs"}) == result
    assert len(requests) == 1


async def test_pipeline_only_results_never_request_nonexistent_metrics(registration, api, tmp_path):
    transport, requests, remote = api
    registry = ProteinModelRegistry(registration, tmp_path, transport=transport)
    await registry.submit(
        "af3", "pipeline", {"input_pdb_dir": "/shared/pdbs", "run_data_pipeline": True, "run_inference": False}
    )
    body = json.loads(requests[0].content)
    assert body["write_fold_input_json_file"] is True
    remote.update(status="COMPLETED", exit_code="0:0")
    result = await registry.results("pipeline")
    assert result["results_available"] and result["metrics"] is None
    assert not any(r.url.path.endswith("/metrics") for r in requests)
    assert (tmp_path / "pipeline.results.json").exists()


async def test_prepared_inference_does_not_claim_automatic_metrics(registration, api, tmp_path):
    transport, requests, remote = api
    registry = ProteinModelRegistry(registration, tmp_path, transport=transport)
    await registry.submit(
        "af3",
        "prepared",
        {"mode": "prepared", "json_dir": "/shared/json", "h5_dir": "/shared/h5", "run_data_pipeline": True},
    )
    remote.update(status="COMPLETED", exit_code=0)
    assert (await registry.results("prepared"))["metrics"] is None
    assert not any(r.url.path.endswith("/metrics") for r in requests)


async def test_metrics_mode_extracts_existing_outputs_without_inference(registration, api, tmp_path):
    transport, requests, remote = api
    registry = ProteinModelRegistry(registration, tmp_path, transport=transport)
    await registry.submit(
        "af3", "metrics", {"mode": "metrics", "input_pdb_dir": "/shared/pdbs", "af3score_output_dir": "/shared/out"}
    )
    body = json.loads(requests[0].content)
    assert body["run_inference"] is False and body["run_data_pipeline"] is False
    remote.update(status="COMPLETED", exit_code="0:0")
    assert (await registry.results("metrics"))["metrics"]["metrics"][0]["iptm"] == 0.85


@pytest.mark.parametrize(
    "status,exit_code",
    [("UNKNOWN", None), ("RUNNING", None), ("COMPLETED", None), ("COMPLETED", "1:0"), ("FAILED", "0:0")],
)
async def test_results_require_verified_success(registration, api, tmp_path, status, exit_code):
    transport, requests, remote = api
    registry = ProteinModelRegistry(registration, tmp_path, transport=transport)
    await registry.submit("af3", "task", {"input_pdb_dir": "/shared/pdbs"})
    remote.update(status=status, exit_code=exit_code)
    assert not (await registry.results("task"))["results_available"]
    assert not any(r.url.path.endswith("/results") or r.url.path.endswith("/metrics") for r in requests)


async def test_uncertain_submission_is_durable_across_registry_restarts(registration, tmp_path):
    posted = []

    def timeout(request):
        posted.append(request)
        raise httpx.ReadTimeout("lost response", request=request)

    transport = httpx.MockTransport(timeout)
    first = ProteinModelRegistry(registration, tmp_path, transport=transport)
    value = await first.submit("af3", "stable", {"input_pdb_dir": "/shared/pdbs"})
    assert value["state"] == "submission_uncertain"
    second = ProteinModelRegistry(registration, tmp_path, transport=transport)
    assert (await second.submit("af3", "stable", {"input_pdb_dir": "/shared/pdbs"}))["state"] == "submission_uncertain"
    assert len(posted) == 1
    assert "No remote ID" in (await second.status("stable"))["error"]


async def test_request_id_cannot_be_reused_for_different_parameters(registration, api, tmp_path):
    transport, _, _ = api
    registry = ProteinModelRegistry(registration, tmp_path, transport=transport)
    await registry.submit("af3", "task", {"input_pdb_dir": "/shared/pdbs"})
    with pytest.raises(ValueError, match="different"):
        await registry.submit("af3", "task", {"input_pdb_dir": "/shared/pdbs", "seed": 99})


async def test_af3score_cancel_is_requested_not_falsely_reported_as_cancelled(registration, api, tmp_path):
    transport, requests, _ = api
    registry = ProteinModelRegistry(registration, tmp_path, transport=transport)
    await registry.submit("af3", "task", {"input_pdb_dir": "/shared/pdbs"})
    result = await registry.cancel("task")
    assert result["cancel_state"] == "requested" and result["state"] != "cancelled"
    await registry.cancel("task")
    assert sum(r.url.path.endswith("/cancel") for r in requests) == 1


@pytest.mark.parametrize(
    "name,kind,payload,operation,route",
    [
        (
            "ppi",
            "ppiflow",
            {
                "mode": "nanobody",
                "name": "n",
                "antigen_pdb": "/shared/t.pdb",
                "antigen_chain": "A",
                "framework_dir": "/shared/fw",
                "heavy_chain": "B",
            },
            None,
            "/v1/designs",
        ),
        (
            "craft",
            "torchcraft",
            {
                "mode": "mini_binder",
                "name": "m",
                "pdb_path": "/shared/t.pdb",
                "hotspot_indices": "1",
                "target_full_seq": "/shared/t.csv",
                "design_length": 80,
            },
            None,
            "/v1/designs",
        ),
        ("fold", "torchfold", {"json_path": "/shared/input.json"}, None, "/v1/folds"),
        (
            "energy",
            "rosetta",
            {"input_pdb": "/shared/model.pdb", "chain_group_a": "B", "chain_group_b": "A"},
            "interface_analyze",
            "/v1/tools/interface_analyze",
        ),
    ],
)
async def test_each_model_routes_through_its_registered_protocol(api, tmp_path, name, kind, payload, operation, route):
    transport, requests, _ = api
    registry = ProteinModelRegistry(
        {name: {"kind": kind, "service": {"url": "https://model.test"}}}, tmp_path, transport=transport
    )
    await registry.submit(name, "task", payload, operation)
    assert requests[0].url.path == route
    await registry.status("task")
    if kind == "rosetta":
        assert requests[-1].url.path == "/v1/tasks/job123"
    with pytest.raises(ValueError, match="cancellation"):
        await registry.cancel("task")


async def test_model_tools_are_registered_in_the_actual_assistant_dispatcher(registration, api, tmp_path):
    plugins = active_registry()
    names = {f"protein_model_{action}" for action in ("list", "submit", "status", "results", "cancel")}
    assert names <= set(plugins.tool_names())
    tools = ToolRegistry()
    services = ServiceLocator(workspace=tmp_path, user_id="test", agent_id="test")
    for name in names:
        tool = plugins.build_tool(
            name, config={"model_services": registration, "model_jobs_dir": str(tmp_path / "jobs")}, services=services
        )
        tool.registry.transport = api[0]
        tools.register(tool)
    schemas = {s["function"]["name"] for s in tools.get_definitions()}
    assert schemas == names
    listed = json.loads(await tools.execute("protein_model_list", {}))
    assert listed["models"][0]["kind"] == "af3score"
    assert "run_data_pipeline" in listed["models"][0]["request_schema"]["properties"]
    response = json.loads(
        await tools.execute(
            "protein_model_submit",
            {
                "model": "af3",
                "request_id": "chat",
                "request": {"input_pdb_dir": "/shared/pdbs"},
                "user_confirmed": True,
            },
        )
    )
    assert response["state"] == "submitted"
    assert tools.get("protein_model_submit").external_effects
    assert not tools.get("protein_model_status").external_effects


async def test_tool_rejects_unapproved_submission_before_network(registration, api, tmp_path):
    registry = ProteinModelRegistry(registration, tmp_path, transport=api[0])
    tool = ProteinModelTool(registry, "submit")
    result = await tool.execute(model="af3", request_id="task", request={"input_pdb_dir": "/shared/pdbs"})
    assert "authorization" in json.loads(result.model_text)["error"]
    assert not api[1]


async def test_remote_inputs_and_sample_limits_are_validated(registration, api, tmp_path):
    registration["af3"]["max_samples"] = 2
    registry = ProteinModelRegistry(registration, tmp_path, transport=api[0])
    for request in (
        {"input_pdb_dir": "/private/data"},
        {"input_pdb_dir": "relative"},
        {"input_pdb_dir": "/shared/../private"},
        {"input_pdb_dir": "/shared/pdbs", "num_samples": 3},
    ):
        with pytest.raises(ValueError):
            await registry.submit("af3", "bad", request)
    assert not api[1]


async def test_missing_credential_does_not_poison_request_id(registration, api, tmp_path, monkeypatch):
    registration["af3"]["service"]["token_env"] = "TEST_AF3_TOKEN"
    monkeypatch.delenv("TEST_AF3_TOKEN", raising=False)
    registry = ProteinModelRegistry(registration, tmp_path, transport=api[0])
    with pytest.raises(ValueError, match="absent"):
        await registry.submit("af3", "task", {"input_pdb_dir": "/shared/pdbs"})
    assert not (tmp_path / "task.json").exists()
    monkeypatch.setenv("TEST_AF3_TOKEN", "synthetic-secret")
    assert (await registry.submit("af3", "task", {"input_pdb_dir": "/shared/pdbs"}))["state"] == "submitted"
    assert "synthetic-secret" not in (tmp_path / "task.json").read_text()


def test_initial_round_rejects_pipeline_only_without_selection_metrics():
    config = {
        "af3score": {"url": "https://af3.test"},
        "torchfold": {"url": "https://fold.test"},
        "rosetta": {"url": "https://rosetta.test"},
        "dockq_python": "/tools/python",
        "af3score_options": {"run_data_pipeline": True, "run_inference": False},
    }
    with pytest.raises(ValueError, match="requires AF3Score run_inference"):
        Evaluation.model_validate(config)


async def test_four_named_models_have_independent_schemas_and_chat_dispatch(api, tmp_path):
    definitions = {
        "af3score": {"input_pdb_dir": "/shared/pdbs", "run_data_pipeline": True, "run_inference": False},
        "ppiflow": {
            "mode": "nanobody",
            "name": "n",
            "antigen_pdb": "/shared/t.pdb",
            "antigen_chain": "A",
            "framework_dir": "/shared/fw",
            "heavy_chain": "B",
        },
        "torchcraft": {
            "mode": "vhh",
            "name": "v",
            "pdb_path": "/shared/t.pdb",
            "hotspot_indices": "1",
            "target_full_seq": "/shared/t.csv",
            "framework_cif_path": "/shared/fw.cif",
        },
        "torchfold": {"json_path": "/shared/input.json"},
    }
    registrations = {kind: {"kind": kind, "service": {"url": f"https://{kind}.test"}} for kind in definitions}
    plugins = active_registry()
    tools = ToolRegistry()
    for kind, payload in definitions.items():
        name = f"protein_model_{kind}"
        assert name in plugins.tool_names()
        tool = plugins.build_tool(
            name,
            config={"model_services": registrations, "model_jobs_dir": str(tmp_path)},
            services=ServiceLocator(workspace=tmp_path, user_id="test", agent_id="test"),
        )
        tool.registry.transport = api[0]
        tools.register(tool)
        schema = tool.parameters["properties"]["request"]["properties"]
        expected = {
            "af3score": "run_inference",
            "ppiflow": "framework_dir",
            "torchcraft": "framework_cif_path",
            "torchfold": "json_path",
        }[kind]
        assert expected in schema
        result = json.loads(
            await tools.execute(name, {"request_id": "chat_" + kind, "request": payload, "user_confirmed": True})
        )
        assert result["state"] == "submitted" and result["kind"] == kind
    assert len([r for r in api[1] if r.method == "POST"]) == 4


async def test_named_model_tool_cannot_dispatch_to_another_model_kind(registration, api, tmp_path):
    from opendde_harness.plugin.protein_design.tools.models import ProteinModelSubmitTool

    tool = ProteinModelSubmitTool(ProteinModelRegistry(registration, tmp_path, transport=api[0]), "torchfold")
    result = await tool.execute(
        model="af3", request_id="wrong", request={"input_pdb_dir": "/shared/pdbs"}, user_confirmed=True
    )
    assert "registered torchfold alias" in json.loads(result.model_text)["error"]
    assert not api[1]


async def test_response_profile_can_be_corrected_without_changing_remote_server(registration, api, tmp_path):
    transport, _, remote = api
    registry = ProteinModelRegistry(registration, tmp_path, transport=transport)
    await registry.submit("af3", "task", {"input_pdb_dir": "/shared/pdbs"})
    registration["af3"]["service"]["status_field"] = "slurm.state"
    registration["af3"]["service"]["exit_code_field"] = "slurm.exit"
    corrected = ProteinModelRegistry(registration, tmp_path, transport=transport)
    remote.update(slurm={"state": "COMPLETED", "exit": "0:0"})
    assert (await corrected.status("task"))["state"] == "succeeded"
    registration["af3"]["service"]["url"] = "https://another.test"
    changed = ProteinModelRegistry(registration, tmp_path, transport=transport)
    with pytest.raises(ValueError, match="registration changed"):
        await changed.status("task")


async def test_health_does_not_hide_other_models_when_one_credential_is_missing(
    registration, api, tmp_path, monkeypatch
):
    registration["af3"]["service"]["token_env"] = "TEST_ABSENT_TOKEN"
    monkeypatch.delenv("TEST_ABSENT_TOKEN", raising=False)
    registration["fold"] = {"kind": "torchfold", "service": {"url": "https://fold.test"}}
    result = await ProteinModelRegistry(registration, tmp_path, transport=api[0]).catalogue_with_health()
    models = {row["model"]: row for row in result["models"]}
    assert not models["af3"]["reachable"] and models["fold"]["reachable"]
    assert not models["fold"]["inference_verified"]


def test_documented_registration_fragment_has_all_five_models():
    from pathlib import Path

    value = json.loads((Path(__file__).parents[1] / "docs/examples/model_services.json").read_text())
    models = value["plugins"]["config"]["protein-design"]["model_services"]
    registry = ProteinModelRegistry(models, Path("/unused"))
    assert {row["kind"] for row in registry.catalogue()["models"]} == {
        "af3score",
        "ppiflow",
        "torchcraft",
        "torchfold",
        "rosetta",
    }


def test_cli_bootstrap_exposes_named_models_to_existing_qwen_provider(registration, tmp_path):
    from opendde_harness.cli._plugin_stack import build_plugin_tools
    from opendde_harness.config.opendde_harness import OpenDDEHarnessConfig

    config = OpenDDEHarnessConfig.model_validate(
        {
            "plugins": {
                "config": {"protein-design": {"model_services": registration, "model_jobs_dir": str(tmp_path / "jobs")}}
            }
        }
    )
    tools = build_plugin_tools(tmp_path, config, model="qwen-turbo")
    names = {tool.name for tool in tools}
    assert {
        "protein_model_ppiflow",
        "protein_model_torchcraft",
        "protein_model_torchfold",
        "protein_model_af3score",
        "protein_model_status",
        "protein_model_results",
    } <= names
    assert (
        next(tool for tool in tools if tool.name == "protein_model_af3score").registry.models["af3"].kind == "af3score"
    )


def test_readiness_accepts_external_models_without_legacy_compute_url(registration):
    from opendde_harness.plugin.protein_design.readiness import compute_configured

    assert compute_configured({"model_services": registration})
    assert not compute_configured({"model_services": {}})


async def test_torchcraft_unconditional_mode_accepts_the_users_curl_shape(api, tmp_path):
    registration = {"torchcraft": {"kind": "torchcraft", "service": {"url": "https://torchcraft.test"}}}
    registry = ProteinModelRegistry(registration, tmp_path, transport=api[0])
    payload = {
        "mode": "monomer_unconditional",
        "name": "monomer_demo",
        "json_path": "/shared/TorchX/test-demo.json",
        "design_length": 100,
        "num_designs": 3,
        "seed": 1,
    }
    from opendde_harness.plugin.protein_design.tools.models import ProteinModelSubmitTool

    tools = ToolRegistry()
    tools.register(ProteinModelSubmitTool(registry, "torchcraft"))
    response = json.loads(
        await tools.execute(
            "protein_model_torchcraft", {"request_id": "unconditional", "request": payload, "user_confirmed": True}
        )
    )
    assert response["state"] == "submitted"
    assert json.loads(api[1][0].content) == payload
    assert (
        "monomer_unconditional"
        in tools.get_definitions()[0]["function"]["parameters"]["properties"]["request"]["properties"]["mode"]["enum"]
    )


def test_torchcraft_mode_specific_inputs_are_required():
    from opendde_harness.plugin.protein_design.core.model_requests import TorchCraftRequest

    for payload in (
        {"mode": "monomer_unconditional", "name": "x", "design_length": 100},
        {"mode": "mini_binder", "name": "x", "design_length": 80},
        {"mode": "vhh", "name": "x", "json_path": "/shared/input.json"},
    ):
        with pytest.raises(ValueError):
            TorchCraftRequest.model_validate(payload)


async def test_rosetta_has_its_own_named_tool_and_six_operations(api, tmp_path):
    plugins = active_registry()
    tool = plugins.build_tool(
        "protein_model_rosetta",
        config={
            "model_services": {"energy": {"kind": "rosetta", "service": {"url": "https://rosetta.test"}}},
            "model_jobs_dir": str(tmp_path),
        },
        services=ServiceLocator(workspace=tmp_path, user_id="test", agent_id="test"),
    )
    tool.registry.transport = api[0]
    assert len(tool.parameters["properties"]["operation"]["enum"]) == 6
    assert "operation" in tool.parameters["required"]
    tools = ToolRegistry()
    tools.register(tool)
    result = json.loads(
        await tools.execute(
            "protein_model_rosetta",
            {
                "request_id": "score",
                "operation": "structure_score",
                "request": {"input_pdb": "/shared/model.pdb"},
                "user_confirmed": True,
            },
        )
    )
    assert result["state"] == "submitted"
    assert api[1][0].url.path == "/v1/tools/structure_score"


async def test_ppiflow_antibody_accepts_cli_paths_and_rejects_paths_outside_roots(tmp_path, api):
    transport, requests, _ = api
    registry = ProteinModelRegistry(
        {"ppi": {"kind": "ppiflow", "service": {"url": "https://ppi.test"}, "allowed_input_roots": ["/shared"]}},
        tmp_path,
        transport=transport,
    )
    payload = {
        "mode": "antibody",
        "name": "7n0a",
        "antigen_pdb": "/shared/target.pdb",
        "antigen_chain": "C",
        "framework_dir": "/shared/framework_scfv",
        "heavy_chain": "A",
        "light_chain": "B",
        "cdr_length": "CDRH3,10-20,CDRL3,9-11",
        "config": "/shared/config.yaml",
        "model_weights": "/shared/model.ckpt.enc",
        "output_dir": "/shared/outputs/",
    }
    assert (await registry.submit("ppi", "paired", payload))["state"] == "submitted"
    body = json.loads(next(r for r in requests if r.method == "POST").content)
    assert all(body[key] == value for key, value in payload.items())
    for field in ("config", "model_weights", "output_dir"):
        with pytest.raises(ValueError, match="input roots"):
            await registry.submit("ppi", "bad_" + field, {**payload, field: "/outside/file"})
