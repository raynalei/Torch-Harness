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
        assert commands[0][:2] == ("sbatch", "--parsable")
        assert commands[0][2].startswith("--job-name=harness-api-")
        assert commands[0][-1] == str(config.sbatch_script)
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


async def test_empty_squeue_and_accounting_lag_wait_without_resubmission(api, monkeypatch):
    config, commands, requests, transport = api
    queue = iter(["", "RUNNING|gpu01"])

    async def command(config, *args):
        commands.append(args)
        if args[0] == "sbatch":
            return "123"
        if args[0] == "squeue":
            return next(queue)
        return "" if args[0] == "sacct" else "gpu01"

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", command)
    client = ServiceConnection(config, transport=transport)
    try:
        assert (await client.post("v1/designs", json={})).status_code == 202
        assert sum(c[0] == "sbatch" for c in commands) == 1
        assert sum(c[0] == "sacct" for c in commands) == 1
        assert all(r.url.host == "gpu01" for r in requests)
    finally:
        await client.close()


async def test_accounting_confirms_api_exited_instead_of_assuming_queue_delay(api, monkeypatch):
    config, commands, requests, transport = api

    async def command(config, *args):
        commands.append(args)
        return {"sbatch": "123", "squeue": "", "sacct": "123.batch|RUNNING|0:0|gpu01\n123|FAILED|1:0|gpu01"}[args[0]]

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", command)
    client = ServiceConnection(config, transport=transport)
    try:
        with pytest.raises(RuntimeError, match="FAILED.*1:0"):
            await client.post("v1/designs", json={})
        assert not requests
        state = json.loads(client.config.sbatch_state_file.read_text())
        assert state["state"] == "terminated"
    finally:
        await client.close()


async def test_slurm_query_errors_have_bounded_retries(api, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.api_startup import ApiStartupError

    config, commands, requests, transport = api

    async def command(config, *args):
        commands.append(args)
        if args[0] == "sbatch":
            return "123"
        raise ApiStartupError("controller unavailable; inspect command log")

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", command)
    client = ServiceConnection(config, transport=transport)
    try:
        with pytest.raises(RuntimeError, match="failed 3 times"):
            await client.post("v1/designs", json={})
        assert sum(c[0] == "squeue" for c in commands) == 3
        assert sum(c[0] == "sbatch" for c in commands) == 1
        assert not requests
    finally:
        await client.close()


async def test_empty_slurm_visibility_remains_bounded_by_startup_timeout(api, monkeypatch):
    config, commands, requests, transport = api
    config.startup_timeout = 0.002

    async def command(config, *args):
        commands.append(args)
        return "123" if args[0] == "sbatch" else ""

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", command)
    client = ServiceConnection(config, transport=transport)
    try:
        with pytest.raises(TimeoutError, match="UNKNOWN"):
            await client.post("v1/designs", json={})
        assert sum(c[0] == "sbatch" for c in commands) == 1
        assert not requests
    finally:
        await client.close()


async def test_failed_slurm_command_preserves_stderr_in_local_log(tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.api_startup import ApiStartupError, slurm_command

    config = Service(
        url="http://localhost:8010", sbatch_script=tmp_path / "api.sbatch", sbatch_state_file=tmp_path / "api.json"
    )

    async def process(*args, **kwargs):
        class Failed:
            returncode = 1

            async def communicate(self):
                return b"", b"slurm_load_jobs error: Unable to contact slurm controller"

        return Failed()

    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.initial_round.api_startup.asyncio.create_subprocess_exec", process
    )
    with pytest.raises(ApiStartupError, match="slurm.log"):
        await slurm_command(config, "squeue", "--jobs", "123")
    assert "Unable to contact slurm controller" in config.sbatch_state_file.with_suffix(".slurm.log").read_text()


async def test_cleanup_only_cancels_owned_scheduler_name_and_is_idempotent(api, monkeypatch):
    config, commands, _, transport = api
    client = ServiceConnection(config, transport=transport)
    await client.ensure_ready()
    record = client.allocations[0]
    seen = []

    async def command(config, *args):
        seen.append(args)
        if args[0] == "squeue":
            return "RUNNING|" + record["allocation_name"]
        return ""

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", command)
    first = await client.release_allocations()
    assert first == [{"state": "cancel_requested", "job_id": "123"}]
    assert await client.release_allocations() == first
    assert seen.count(("scancel", "123")) == 1
    await client.close()


async def test_cleanup_refuses_reused_job_id_with_different_name(api, monkeypatch):
    config, _, _, transport = api
    client = ServiceConnection(config, transport=transport)
    await client.ensure_ready()
    seen = []

    async def command(config, *args):
        seen.append(args)
        return "RUNNING|someone-elses-job"

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", command)
    result = await client.release_allocations()
    assert result[0]["state"] == "cleanup_failed"
    assert not any(args[0] == "scancel" for args in seen)
    await client.close()


async def test_cleanup_does_not_treat_accounting_lag_as_termination(api, monkeypatch):
    config, _, _, transport = api
    client = ServiceConnection(config, transport=transport)
    await client.ensure_ready()
    seen = []

    async def command(config, *args):
        seen.append(args)
        return ""

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", command)
    assert (await client.release_allocations())[0]["state"] == "cleanup_deferred"
    assert not any(args[0] == "scancel" for args in seen)
    await client.close()


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8070", "http://localhost:8060", "http://[::1]:8040", "https://api.test"]
)
async def test_missing_sbatch_never_sends_http_even_for_remote_urls(url):
    requests = []
    client = ServiceConnection(
        Service(url=url), transport=httpx.MockTransport(lambda request: requests.append(request))
    )
    try:
        with pytest.raises(RuntimeError, match="sbatch_script"):
            await client.post("/v1/tasks", json={})
        with pytest.raises(RuntimeError, match="sbatch_script"):
            await client.get("/health")
        assert not requests
    finally:
        await client.close()


@pytest.mark.parametrize(
    "hostname", ["localhost", "localhost.", "127.0.0.2", "0.0.0.0", "127.1", "0x7f000001", "localhost6"]
)
async def test_scheduler_loopback_or_ambiguous_host_never_gets_http(api, monkeypatch, hostname):
    config, _, requests, transport = api

    async def command(config, *args):
        if args[0] == "sbatch":
            return "123"
        if args[0] == "squeue":
            return "RUNNING|node"
        return hostname

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", command)
    client = ServiceConnection(config, transport=transport)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            await client.post("/v1/tasks", json={})
        assert not requests
    finally:
        await client.close()


@pytest.mark.parametrize(
    "path",
    ["http://127.0.0.1:8070/v1/tasks", "//localhost:8060/v1/tasks", "https://other.test/tasks", "\\localhost\\tasks"],
)
async def test_request_path_cannot_override_allocated_endpoint(api, path):
    config, commands, requests, transport = api
    client = ServiceConnection(config, transport=transport)
    try:
        with pytest.raises(RuntimeError, match="relative paths"):
            await client.post(path, json={})
        assert not commands and not requests
    finally:
        await client.close()


async def test_recovery_rejects_local_or_unbound_endpoint_before_http(api):
    config, _, requests, transport = api
    first = ServiceConnection(config, transport=transport)
    await first.ensure_ready()
    record = first.allocations[0]
    requests.clear()
    restored = ServiceConnection(config, transport=transport)
    try:
        with pytest.raises(RuntimeError):
            restored.attach_api("http://127.0.0.1:8010/base/", record)
        with pytest.raises(RuntimeError, match="not bound"):
            restored.attach_api("http://other-node:8010/base/", record)
        with pytest.raises(RuntimeError, match="saved sbatch"):
            restored.attach_api("http://gpu01:8010/base/")
        restored.attach_api("http://gpu01:8010/base/", record)
        await restored.get("/health")
        assert requests and all(request.url.host == "gpu01" for request in requests)
    finally:
        await first.close()
        await restored.close()


async def test_redirect_and_runtime_endpoint_tampering_never_reach_localhost(api):
    config, _, requests, _ = api

    def handler(request):
        requests.append(request)
        assert request.url.host == "gpu01"
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(302, headers={"Location": "http://127.0.0.1:8070/v1/tasks"})

    client = ServiceConnection(config, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get("/v1/tasks")
        requests.clear()
        client.client.base_url = "http://127.0.0.1:8010/base/"
        with pytest.raises(ValueError, match="verified sbatch"):
            await client.post("/v1/tasks", json={})
        assert not requests
    finally:
        await client.close()


def test_disabling_host_resolution_is_not_a_configuration_mode(api):
    config, _, _, _ = api
    with pytest.raises(ValueError):
        Service.model_validate({**config.model_dump(mode="json"), "sbatch_resolve_host": False})


@pytest.mark.parametrize(
    "kind,payload,operation",
    [
        (
            "ppiflow",
            {
                "mode": "nanobody",
                "name": "test",
                "antigen_pdb": "/shared/input.pdb",
                "antigen_chain": "C",
                "framework_dir": "/shared/framework",
                "heavy_chain": "A",
            },
            None,
        ),
        (
            "torchcraft",
            {
                "mode": "vhh",
                "name": "test",
                "pdb_path": "/shared/input.pdb",
                "hotspot_indices": "1",
                "target_full_seq": "/shared/target.csv",
                "framework_cif_path": "/shared/framework.cif",
            },
            None,
        ),
        ("af3score", {"input_pdb_dir": "/shared/pdbs"}, None),
        ("torchscore", {"input_pdb_dir": "/shared/pdbs"}, None),
        ("torchfold", {"json_path": "/shared/input.json"}, None),
        ("torchopt", {"pdb_paths": ["/shared/input.pdb"]}, None),
        (
            "rosetta",
            {"input_pdb": "/shared/input.pdb", "chain_group_a": "A", "chain_group_b": "C"},
            "interface_analyze",
        ),
    ],
)
async def test_all_registered_scientific_tools_require_sbatch_before_http(tmp_path, kind, payload, operation):
    from opendde_harness.plugin.protein_design.core.model_registry import ProteinModelRegistry

    requests = []
    registry = ProteinModelRegistry(
        {kind: {"kind": kind, "service": {"url": "http://127.0.0.1:8070"}}},
        tmp_path,
        transport=httpx.MockTransport(lambda request: requests.append(request)),
    )
    with pytest.raises(RuntimeError, match="sbatch_script"):
        await registry.submit(kind, "test", payload, operation)
    assert requests == []
