"""Direct Slurm execution, exact accounting evidence and immutable native artifacts."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

from opendde_harness.plugin.protein_design.initial_round.config import PathMapping, Service
from opendde_harness.plugin.protein_design.initial_round.rosetta_slurm import RosettaSlurmService
from opendde_harness.plugin.protein_design.initial_round.services import RemoteJobFailedError, SubmissionUncertainError
from opendde_harness.plugin.protein_design.initial_round.storage import Checkpoint, SharedFiles, retry_failed_jobs
from tests.test_rosetta_api_deployment import FAKE_PYROSETTA, complex_pdb


@pytest.fixture
def direct(tmp_path, monkeypatch):
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "pyrosetta.py").write_text(FAKE_PYROSETTA)
    monkeypatch.setenv("PYTHONPATH", str(fake))
    monkeypatch.setattr("shutil.which", lambda command: "/mock/" + command)
    calls = []
    state = {"accounted": True, "fail": False, "uncertain": False}

    async def command(config, *args):
        calls.append(args)
        if args[0] == "sbatch":
            process = await asyncio.create_subprocess_exec(
                "bash", args[-1], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=dict(os.environ)
            )
            stdout, stderr = await process.communicate()
            assert process.returncode == 0, (stdout + stderr).decode()
            if state["uncertain"]:
                raise TimeoutError("lost sbatch response")
            return "12345;cluster"
        if args[0] == "squeue":
            return ""
        if args[0] == "sacct":
            if not state["accounted"]:
                return ""
            return "12345.batch|FAILED|1:0\n12345|" + (
                "FAILED|1:0" if state["fail"] else "COMPLETED|" + state.get("exit_code", "0:0")
            )
        raise AssertionError(args)

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.rosetta_slurm.command", command)
    checkpoint = Checkpoint(tmp_path / "run" / "checkpoint.json", {})
    config = Service(
        backend="slurm", python=Path(sys.executable), partition="cpu", poll_interval=0.001, job_timeout=0.02
    )
    files = SharedFiles([PathMapping(local=tmp_path, remote=str(tmp_path))])
    service = RosettaSlurmService(config, checkpoint, files)
    payload = {
        "input_pdb": str(complex_pdb(tmp_path / "complex.pdb")),
        "chain_group_a": "B",
        "chain_group_b": "A",
        "pack_input": False,
        "pack_separated": True,
    }
    return service, checkpoint, payload, calls, state


async def run(service, payload, **kwargs):
    return await service.run("interface/parent", "interface_analyze", "slurm_job_id", payload, **kwargs)


async def test_direct_worker_returns_dg_and_pair_logs_without_http(direct, monkeypatch):
    service, checkpoint, payload, calls, state = direct
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda *args, **kwargs: pytest.fail("Rosetta must not construct an HTTP client")
    )
    result = await run(service, payload)
    assert result["metrics"] == [{"dG_separated": -12.0, "total_score": -30.0}]
    assert any("-6.25" in Path(path).read_text() for path in result["files"] if path.endswith(".out"))
    assert checkpoint.data["jobs"]["interface/parent"]["id"] == "12345"
    assert checkpoint.data["jobs"]["interface/parent"]["state"] == "completed"
    assert service.allocations == []
    assert await service.release_allocations() == []
    before = list(calls)
    assert await run(service, payload) == result
    assert calls == before
    assert sum(call[0] == "sbatch" for call in calls) == 1
    assert not any(call[0] in {"scancel", "uvicorn"} for call in calls)


async def test_uncertain_sbatch_requires_reconciliation_and_never_submits_twice(direct):
    service, checkpoint, payload, calls, state = direct
    state["uncertain"] = True
    with pytest.raises(SubmissionUncertainError):
        await run(service, payload)
    with pytest.raises(SubmissionUncertainError):
        await run(service, payload)
    state["uncertain"] = False
    result = await run(service, payload, attach_id="12345")
    assert result["metrics"][0]["dG_separated"] == -12.0
    assert sum(call[0] == "sbatch" for call in calls) == 1


async def test_accounting_lag_preserves_job_id_and_completed_result_alone_is_not_success(direct):
    service, checkpoint, payload, calls, state = direct
    state["accounted"] = False
    with pytest.raises(TimeoutError, match="remains submitted"):
        await run(service, payload)
    job = checkpoint.data["jobs"]["interface/parent"]
    assert job["id"] == "12345" and job["state"] == "submitted"
    assert Path(job["result_path"]).is_file()
    state["accounted"] = True
    await run(service, payload)
    assert sum(call[0] == "sbatch" for call in calls) == 1


async def test_failed_slurm_job_retries_with_new_outputs_only_when_requested(direct):
    service, checkpoint, payload, calls, state = direct
    state["fail"] = True
    with pytest.raises(RemoteJobFailedError):
        await run(service, payload)
    old = checkpoint.data["jobs"]["interface/parent"]["result_path"]
    with pytest.raises(RemoteJobFailedError):
        await run(service, payload)
    assert sum(call[0] == "sbatch" for call in calls) == 1
    assert retry_failed_jobs(checkpoint) == ["interface/parent"]
    state["fail"] = False
    await run(service, payload)
    assert checkpoint.data["jobs"]["interface/parent"]["result_path"] != old
    assert len(checkpoint.data["job_attempts"]["interface/parent"]) == 1
    assert sum(call[0] == "sbatch" for call in calls) == 2


async def test_cached_result_tampering_cannot_trigger_a_new_computation(direct):
    service, checkpoint, payload, calls, state = direct
    result = await run(service, payload)
    Path(result["files"][0]).write_text("corrupted")
    with pytest.raises(ValueError, match="output changed"):
        await run(service, payload)
    assert sum(call[0] == "sbatch" for call in calls) == 1


def test_default_workflow_uses_direct_rosetta_even_with_legacy_api_registration():
    from opendde_harness.plugin.protein_design.initial_round.defaults import load_deployment_defaults

    value = load_deployment_defaults(
        {"workflow_defaults": {"evaluation": {"rosetta": {"url": "http://old-api.invalid:8050"}}}}
    )
    config = Service.model_validate(value["workflow_defaults"]["evaluation"]["rosetta"])
    assert config.backend == "slurm" and config.url is None and config.sbatch_script is None
    assert str(config.python) == "/lustre/grp/cmclab/leiy/conda_envs/ppi_env/bin/python"


async def test_autonomous_interface_consumes_direct_outputs_and_computes_ranking(direct, monkeypatch):
    from types import SimpleNamespace

    from opendde_harness.plugin.protein_design.initial_round.config import Chains
    from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner

    service, checkpoint, payload, calls, state = direct
    monkeypatch.setattr("httpx.AsyncClient", lambda *args, **kwargs: pytest.fail("no Rosetta HTTP"))
    (checkpoint.path.parent / "candidate").mkdir()

    async def job(kind, key, route, id_field, request):
        assert kind == "rosetta"
        return await service.run(key, route, id_field, request)

    runner = SimpleNamespace(
        root=checkpoint.path.parent,
        files=service.files,
        save_candidates=checkpoint.save,
        job=job,
        config=SimpleNamespace(
            autonomous=SimpleNamespace(interface_distance=12.0, interface_energy_threshold=-5.0),
            generators={
                "ppiflow": SimpleNamespace(chains=Chains(target=["A"], binder=["B"], refold_map={"A": "A", "B": "B"}))
            },
            evaluation=SimpleNamespace(
                rosetta=service.config,
                rosetta_options={},
                score_metric="torchscore_iptm",
                rosetta_residue_glob="*.out",
                rosetta_scores_glob="*.sc",
            ),
        ),
    )
    candidate = {
        "candidate_id": "candidate",
        "source": "ppiflow",
        "generated_structure": payload["input_pdb"],
        "metrics": {"torchscore_iptm": 0.7},
    }
    await InitialRoundRunner.score_interface(runner, candidate)
    assert candidate["interface_fixed_positions"] == {"B": [1]}
    assert candidate["metrics"]["rosetta_interface_score"] == -6.25
    assert candidate["metrics"]["rosetta_interface_dg"] == -12.0
    assert candidate["metrics"]["ranking_score"] == 82.0
    before = list(calls)
    await InitialRoundRunner.score_interface(runner, candidate)
    assert calls == before


async def test_scheduler_queue_error_falls_back_to_exact_accounting(direct, monkeypatch):
    service, checkpoint, payload, calls, state = direct
    from opendde_harness.plugin.protein_design.initial_round import rosetta_slurm

    original = rosetta_slurm.command

    async def query(config, *args):
        if args[0] == "squeue":
            raise RuntimeError("Invalid job id specified: no longer in queue")
        return await original(config, *args)

    monkeypatch.setattr(rosetta_slurm, "command", query)
    assert (await run(service, payload))["metrics"][0]["dG_separated"] == -12.0
    assert checkpoint.data["jobs"]["interface/parent"]["state"] == "completed"


async def test_native_worker_supports_docking_partners_binding_variant(direct):
    service, checkpoint, payload, calls, state = direct
    fake = Path(os.environ["PYTHONPATH"]) / "pyrosetta.py"
    fake.write_text(
        fake.read_text()
        + """
from types import SimpleNamespace
class DockingPartners(str): pass
def typed_interface(self, partners):
    if not isinstance(partners, DockingPartners):
        raise TypeError("DockingPartners required")
InterfaceAnalyzerMover.set_interface = typed_interface
rosetta.protocols.docking = SimpleNamespace(DockingPartners=DockingPartners)
"""
    )
    assert (await run(service, payload))["metrics"][0]["dG_separated"] == -12.0


async def test_completed_without_exit_code_keeps_polling_same_job(direct):
    service, checkpoint, payload, calls, state = direct
    state["exit_code"] = ""
    with pytest.raises(TimeoutError):
        await run(service, payload)
    assert checkpoint.data["jobs"]["interface/parent"]["state"] == "submitted"
    state["exit_code"] = "0:0"
    await run(service, payload)
    assert sum(call[0] == "sbatch" for call in calls) == 1
