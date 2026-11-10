"""API allocation startup without a real scheduler or external network."""

import asyncio
import json

import httpx
import pytest

from opendde_harness.plugin.protein_design.initial_round.config import Service
from opendde_harness.plugin.protein_design.initial_round.services import ServiceConnection

pytestmark = pytest.mark.in_memory_transport


@pytest.fixture
def api(tmp_path, monkeypatch):
    script = tmp_path / "api.sbatch"
    script.write_text("#!/bin/bash\nsrun uvicorn main:app --host 0.0.0.0 --port 8010\n")
    config = Service(
        url="http://127.0.0.1:8010/base",
        sbatch_script=script,
        sbatch_state_file=tmp_path / "allocation.json",
        startup_poll_interval=0.001,
    )
    commands = []
    requests = []
    phases = ["PENDING|", "RUNNING|gpu[01]"]

    async def command(config, *args):
        commands.append(args)
        if args[0] == "sbatch":
            return "123"
        if args[0] == "squeue":
            return phases.pop(0) if phases else "RUNNING|gpu[01]"
        return "gpu01"

    def handler(request):
        requests.append(request)
        if request.url.host == "127.0.0.1":
            return httpx.Response(503)
        if request.method == "POST":
            return httpx.Response(202, json={"design_id": "design1"})
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", command)
    return config, commands, requests, httpx.MockTransport(handler)


async def test_sbatch_precedes_http_post_and_resolves_node_and_port(api):
    config, commands, requests, transport = api
    client = ServiceConnection(config, transport=transport)
    try:
        response = await client.post("v1/designs", json={"name": "test"})
        assert response.status_code == 202
        assert commands[0] == ("sbatch", "--parsable", str(config.sbatch_script))
        assert str(requests[-1].url) == "http://gpu01:8010/base/v1/designs"
        state = json.loads(client.config.sbatch_state_file.read_text())
        assert state["job_id"] == "123" and state["state"] == "ready"
        await client.post("v1/designs", json={})
        assert sum(command[0] == "sbatch" for command in commands) == 1
    finally:
        await client.close()
    # A new task/client always submits a fresh API allocation.
    other = ServiceConnection(config, transport=transport)
    try:
        await other.get("health")
        assert other.client.base_url.host == "gpu01"
        assert sum(command[0] == "sbatch" for command in commands) == 2
        assert other.config.sbatch_state_file != client.config.sbatch_state_file
    finally:
        await other.close()


async def test_health_check_never_submits_allocation(api):
    config, commands, _, transport = api
    client = ServiceConnection(config, transport=transport)
    try:
        with pytest.raises(RuntimeError, match="does not submit"):
            await client.get("health", start_service=False)
        assert commands == []
        assert not config.sbatch_state_file.exists()
    finally:
        await client.close()


async def test_uncertain_sbatch_is_not_repeated(api, monkeypatch):
    config, commands, requests, transport = api

    async def command(config, *args):
        commands.append(args)
        raise TimeoutError("scheduler response lost")

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", command)
    client = ServiceConnection(config, transport=transport)
    try:
        with pytest.raises(TimeoutError):
            await client.post("v1/designs", json={})
        with pytest.raises(RuntimeError, match="uncertain"):
            await client.post("v1/designs", json={})
        assert len(commands) == 1
        assert not any(request.method == "POST" for request in requests)
    finally:
        await client.close()


async def test_terminal_allocation_fails_without_inference_post(api, monkeypatch):
    config, commands, requests, transport = api

    async def command(config, *args):
        commands.append(args)
        return "123" if args[0] == "sbatch" else "FAILED|gpu01"

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", command)
    client = ServiceConnection(config, transport=transport)
    try:
        with pytest.raises(RuntimeError, match="FAILED"):
            await client.post("v1/designs", json={})
        assert not any(request.method == "POST" for request in requests)
    finally:
        await client.close()


async def test_startup_timeout_reuses_pending_job_on_retry(api, monkeypatch):
    config, commands, _, transport = api
    config.startup_timeout = 0.001

    async def command(config, *args):
        commands.append(args)
        return "123" if args[0] == "sbatch" else "PENDING|"

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", command)
    client = ServiceConnection(config, transport=transport)
    try:
        for _ in range(2):
            with pytest.raises(TimeoutError, match="reuse"):
                await client.post("v1/designs", json={})
        assert sum(command[0] == "sbatch" for command in commands) == 1
    finally:
        await client.close()


async def test_concurrent_tasks_submit_separate_allocations(api):
    config, commands, _, transport = api
    clients = [ServiceConnection(config, transport=transport) for _ in range(2)]
    try:
        await asyncio.gather(*(client.post("v1/designs", json={}) for client in clients))
        assert sum(command[0] == "sbatch" for command in commands) == 2
    finally:
        for client in clients:
            await client.close()


async def test_untracked_healthy_api_is_not_reused(api):
    config, commands, _, _ = api
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        return httpx.Response(200, json={"ok": True})

    client = ServiceConnection(config, transport=httpx.MockTransport(handler))
    try:
        await client.post("v1/designs", json={})
        assert commands[0][0] == "sbatch"
        assert set(hosts) == {"gpu01"}
    finally:
        await client.close()


async def test_registry_starts_api_before_recording_or_posting_inference(api, tmp_path):
    from opendde_harness.plugin.protein_design.core.model_registry import ProteinModelRegistry

    config, commands, requests, transport = api
    registry = ProteinModelRegistry(
        {"ppi": {"kind": "ppiflow", "service": config.model_dump(mode="json")}},
        tmp_path / "requests",
        transport=transport,
    )
    payload = {
        "mode": "nanobody",
        "name": "test",
        "antigen_pdb": "/shared/target.pdb",
        "antigen_chain": "C",
        "framework_dir": "/shared/frameworks",
        "heavy_chain": "A",
    }
    result = await registry.submit("ppi", "test1", payload)
    assert result["state"] == "submitted"
    assert [request.url.host for request in requests if request.method == "POST"] == ["gpu01"]
    assert await registry.submit("ppi", "test1", payload) == result
    assert sum(command[0] == "sbatch" for command in commands) == 1


async def test_registry_api_failure_does_not_poison_inference_request(api, tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.core.model_registry import ProteinModelRegistry

    config, _, requests, transport = api
    registry = ProteinModelRegistry(
        {"fold": {"kind": "torchfold", "service": config.model_dump(mode="json")}},
        tmp_path / "requests",
        transport=transport,
    )

    async def fail(config, *args):
        raise TimeoutError("sbatch stdout lost")

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", fail)
    with pytest.raises(TimeoutError):
        await registry.submit("fold", "fold1", {"json_path": "/shared/input.json"})
    assert not (tmp_path / "requests/fold1.json").exists()
    assert not any(request.method == "POST" for request in requests)


@pytest.mark.parametrize("field", ["sbatch_script", "sbatch_state_file"])
def test_sbatch_paths_must_be_absolute(field):
    with pytest.raises(ValueError, match="absolute"):
        Service(url="http://localhost:8010", **{field: "relative/path"})


async def test_stale_shared_state_never_blocks_new_task(api):
    config, commands, _, transport = api
    config.sbatch_state_file.write_text(json.dumps({"state": "submission_uncertain", "job_id": "999"}))
    client = ServiceConnection(config, transport=transport)
    try:
        await client.post("v1/designs", json={})
        assert commands[0][0] == "sbatch"
        assert json.loads(config.sbatch_state_file.read_text())["job_id"] == "999"
        assert client.config.sbatch_state_file != config.sbatch_state_file
    finally:
        await client.close()


async def test_registry_polling_reconnects_to_own_task_not_new_allocation(api, tmp_path):
    from opendde_harness.plugin.protein_design.core.model_registry import ProteinModelRegistry

    config, commands, requests, transport = api
    registry = ProteinModelRegistry(
        {"ppi": {"kind": "ppiflow", "service": config.model_dump(mode="json")}},
        tmp_path / "jobs",
        transport=transport,
    )
    payload = {
        "mode": "nanobody",
        "name": "test",
        "antigen_pdb": "/shared/target.pdb",
        "antigen_chain": "C",
        "framework_dir": "/shared/frameworks",
        "heavy_chain": "A",
    }
    await registry.submit("ppi", "first", payload)
    await registry.status("first")
    await registry.results("first")
    assert sum(command[0] == "sbatch" for command in commands) == 1
    assert all(r.url.host == "gpu01" for r in requests)
    await registry.submit("ppi", "second", payload)
    assert sum(command[0] == "sbatch" for command in commands) == 2
