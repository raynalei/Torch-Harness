"""Uploaded native APIs exercised through the standalone Harness registry."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from opendde_harness.plugin.protein_design.core.model_registry import ProteinModelRegistry
from opendde_harness.plugin.protein_design.core.model_requests import validate_request
from tests._slurm_api import mock_slurm_apis  # noqa: F401
from tests.test_protein_design_initial_round import pdb

pytestmark = [pytest.mark.in_memory_transport, pytest.mark.usefixtures("mock_slurm_apis")]
ROOT = Path(__file__).resolve().parents[1]


def load_server(kind):
    spec = importlib.util.spec_from_file_location(f"test_{kind}_server", ROOT / "deploy" / kind / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "payload",
    [
        {"inputs": "../secret.cif"},
        {"inputs": "/inputs/x.pdb"},
        {"inputs": ["/inputs/x.json"]},
        {"inputs": ["x.pdb"]},
        {"inputs": []},
        {"inputs": "x.cif", "num_replicates": True},
        {"inputs": "x.cif", "batch_size": 0},
        {"inputs": "x.cif", "seed": -1},
        {"inputs": "x.cif", "generation": "inverse_folding", "partial_trajectory_fraction_steps": 0.5},
        {"inputs": "x.cif", "partial_trajectory_fraction_steps": True},
        {"inputs": "x.cif", "fixed_positions": "B1"},
    ],
)
def test_rfd4_rejects_unsupported_native_inputs(payload):
    with pytest.raises(ValueError):
        validate_request("rfd4", payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"pdb_path": "relative.pdb", "design_chains": ["B"]},
        {"pdb_path": "/shared/a.cif", "design_chains": ["B"]},
        {"pdb_path": "/shared/a.pdb", "design_chains": ["B", "B"]},
        {"pdb_path": "/shared/a.pdb", "design_chains": ["AB"]},
        {"pdb_path": "/shared/a.pdb", "design_chains": ["B"], "num_seq_per_target": 513},
        {"pdb_path": "/shared/a.pdb", "design_chains": ["B"], "seed": 0},
        {"pdb_path": "/shared/a.pdb", "design_chains": ["B"], "mask_positions": {"B": [2]}},
    ],
)
def test_proteinmpnn_rejects_unsupported_constraints(payload):
    with pytest.raises(ValueError):
        validate_request("proteinmpnn", payload)


@pytest.mark.parametrize("suffix", [".cif", ".json"])
def test_rfd4_client_accepts_absolute_input_without_a_bundled_root(suffix):
    payload = {"inputs": f"/other/shared/folder (1)/design{suffix}"}
    assert validate_request("rfd4", payload) == payload


def test_native_optional_controls_stay_on_the_server():
    assert validate_request("rfd4", {"inputs": "specs/target.yaml"}) == {"inputs": "specs/target.yaml"}
    assert validate_request("proteinmpnn", {"pdb_path": "/shared/a.pdb", "design_chains": ["B"]}) == {
        "pdb_path": "/shared/a.pdb",
        "design_chains": ["B"],
    }


@pytest.mark.parametrize(
    "generation,fraction", [("inference_base", None), ("inverse_folding", None), ("inference_base", 0.35)]
)
async def test_rfd4_native_asgi_roundtrip_and_relative_artifacts(tmp_path, monkeypatch, generation, fraction):
    module = load_server("rfd4")
    for name in ("inputs", "jobs", "weights"):
        (tmp_path / name).mkdir()
    (tmp_path / "model.sif").write_text("container-placeholder")
    (tmp_path / "inputs" / "spec.yaml").write_text("native input specification")
    for key, value in {
        "RFD4_INPUTS": "inputs",
        "RFD4_JOBS": "jobs",
        "RFD4_WEIGHTS": "weights",
        "RFD4_SIF": "model.sif",
    }.items():
        monkeypatch.setenv(key, str(tmp_path / value))
    executed = []

    def native_run(args, **kwargs):
        executed.append(args)
        mount = next(arg for arg in args if arg.endswith(":/job:rw"))
        output = Path(mount[: -len(":/job:rw")]) / "output"
        output.mkdir()
        (output / "model.cif").write_text("native model artifact")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", native_run)
    module.app.service = module.RFD4Service()
    registry = ProteinModelRegistry(
        {"rfd4": {"kind": "rfd4", "max_samples": 8, "service": {"url": "https://rfd4.test"}}},
        tmp_path / "client",
        transport=httpx.ASGITransport(app=module.app),
    )
    payload = {"inputs": "spec.yaml", "generation": generation, "num_replicates": 2}
    if fraction is not None:
        payload["partial_trajectory_fraction_steps"] = fraction
    submitted = await registry.submit("rfd4", "rfd4-test", payload)
    module.app.service.worker.shutdown(wait=True)
    assert submitted["state"] == "submitted"
    assert await registry.submit("rfd4", "rfd4-test", payload) == submitted
    result = await registry.results("rfd4-test")
    assert result["state"] == "succeeded" and result["metrics"] is None
    assert result["results"]["relative_files"] == ["model.cif"]
    assert Path(result["results"]["files"][0]).read_text() == "native model artifact"
    assert len(executed) == 1
    assert "inputs=/inputs/spec.yaml" in executed[0]
    if fraction is not None:
        assert "generation.args.partial_trajectory_fraction_steps=0.35" in executed[0]
    assert result["api_cleanup"][0]["state"] in {"cancel_requested", "not_in_queue"}
    assert (await registry.results("rfd4-test"))["results"] == result["results"]
    module.app.service.lock.close()


async def test_proteinmpnn_native_fastapi_roundtrip_caches_fasta_before_cleanup(tmp_path, monkeypatch):
    module = load_server("proteinmpnn")
    module.WORK = tmp_path / "jobs"
    module.WORK.mkdir()
    module.INPUT_ROOTS = (tmp_path,)
    module.WEIGHTS = tmp_path / "v_48_020.pt"
    module.WEIGHTS.write_text("weights-placeholder")
    monkeypatch.setattr(module, "dependencies", lambda: {"fake": True})
    executed = []

    def native_run(args, **kwargs):
        executed.append(args)
        directory = Path(args[args.index("--chdir") + 1])
        output = directory / "output" / "seqs"
        output.mkdir(parents=True)
        (output / "input.fa").write_text(">input, native\nGGGGGG/AAAAAA\n>sample=1, score=1.2\nGGGGGG/ACAAAA\n")
        (directory / "completed").touch()
        return subprocess.CompletedProcess(args, 0, stdout="1999;cluster\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", native_run)
    model = pdb(tmp_path / "complex.pdb")
    registry = ProteinModelRegistry(
        {
            "mpnn": {
                "kind": "proteinmpnn",
                "max_samples": 512,
                "allowed_input_roots": [str(tmp_path)],
                "service": {"url": "https://proteinmpnn.test"},
            }
        },
        tmp_path / "client",
        transport=httpx.ASGITransport(app=module.app),
    )
    payload = {"pdb_path": str(model), "design_chains": ["B"], "num_seq_per_target": 1}
    submitted = await registry.submit("mpnn", "mpnn-test", payload)
    assert submitted["state"] == "submitted" and submitted["slurm_job_id"] == "1999"
    result = await registry.results("mpnn-test")
    assert result["state"] == "succeeded" and result["metrics"] is None
    assert result["results"]["fasta"]["records"][1]["sequence"] == "GGGGGG/ACAAAA"
    assert result["results"]["files"][0].endswith("output/seqs/input.fa")
    assert len(executed) == 1
    assert "--use_soluble_model" in executed[0][-1]
    assert result["api_cleanup"][0]["state"] in {"cancel_requested", "not_in_queue"}
    assert (await registry.results("mpnn-test"))["results"]["fasta"] == result["results"]["fasta"]
    artifact = json.loads(Path(result["results_path"]).read_text())
    assert artifact["results"]["fasta"]["text"].startswith(">input")


@pytest.mark.parametrize(
    "kind,payload",
    [("rfd4", {"inputs": "spec.yaml"}), ("proteinmpnn", {"pdb_path": "/shared/a.pdb", "design_chains": ["B"]})],
)
async def test_uncertain_native_submissions_are_never_reposted(tmp_path, kind, payload):
    posts = []

    def handler(request):
        if request.method == "POST":
            posts.append(request)
            raise httpx.ReadTimeout("lost response", request=request)
        return httpx.Response(200, json={"status": "ok"})

    registry = ProteinModelRegistry(
        {kind: {"kind": kind, "service": {"url": f"https://{kind}.test"}}},
        tmp_path / "client",
        transport=httpx.MockTransport(handler),
    )
    first = await registry.submit(kind, "uncertain-native", payload)
    assert first["state"] == "submission_uncertain"
    second = await registry.submit(kind, "uncertain-native", payload)
    assert second["state"] == "submission_uncertain" and len(posts) == 1


async def test_independent_skills_tools_registered_without_changing_planner_menu(tmp_path):
    from opendde_harness.plugin import active_registry
    from opendde_harness.plugin.context import ServiceLocator

    registry = active_registry()
    assert {"protein_model_rfd4", "protein_model_proteinmpnn"} <= set(registry.tool_names())
    tools = [
        registry.build_tool(
            name,
            config={"model_services": {}},
            services=ServiceLocator(workspace=tmp_path, user_id="test", agent_id="test"),
        )
        for name in ("protein_model_rfd4", "protein_model_proteinmpnn")
    ]
    for tool in tools:
        assert "request" in tool.parameters["properties"]
        result = await tool.execute(request_id="test", request={}, user_confirmed=False)
        assert "error" in json.loads(result.model_text)


async def test_proteinmpnn_server_default_respects_registered_sample_cap(tmp_path):
    registry = ProteinModelRegistry(
        {"mpnn": {"kind": "proteinmpnn", "max_samples": 1, "service": {"url": "https://proteinmpnn.test"}}}, tmp_path
    )
    with pytest.raises(ValueError, match="num_seq_per_target exceeds"):
        await registry.submit("mpnn", "cap-test", {"pdb_path": "/shared/complex.pdb", "design_chains": ["B"]})
