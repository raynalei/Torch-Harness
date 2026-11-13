"""Exercise external protocols, resumability, chain identity and scientific outputs."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import httpx
import pytest

from opendde_harness.plugin.protein_design.initial_round.artifacts import measured, prepare_fold, refold_output
from opendde_harness.plugin.protein_design.initial_round.config import Chains, InitialRoundConfig, PathMapping, Service
from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner
from opendde_harness.plugin.protein_design.initial_round.services import (
    RemoteJobFailedError,
    SlurmService,
    SubmissionUncertainError,
)
from opendde_harness.plugin.protein_design.initial_round.storage import Checkpoint, SharedFiles
from tests._slurm_api import mock_slurm_apis  # noqa: F401

pytestmark = [pytest.mark.in_memory_transport, pytest.mark.usefixtures("mock_slurm_apis")]


def pdb(path, *, binder="ALA", target="A", chain="B", light=None):
    lines = []
    serial = 1
    rows = [(target, "GLY", 0), (chain, binder, 4)]
    if light is not None:
        rows.append((light, "VAL", 8))
    for chain_id, residue, y in rows:
        for index in range(1, 7):
            for atom, dx, dz, element in [
                ("N", 0, 0, "N"),
                ("CA", 1.2, 0.5, "C"),
                ("C", 2.4, 0, "C"),
                ("O", 2.8, -0.8, "O"),
            ]:
                x = index * 3.4 + dx
                z = index % 2 + dz
                lines.append(
                    f"ATOM  {serial:5d} {atom:^4s} {residue} {chain_id}{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00          {element:>2s}"
                )
                serial += 1
        lines.append("TER")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


@pytest.fixture
def round_config(tmp_path):
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "dialect": "alphafold3",
                "version": 1,
                "name": "template",
                "modelSeeds": [1],
                "sequences": [
                    {"protein": {"id": "X", "sequence": "GGGGGG", "unpairedMsa": "", "pairedMsa": "", "templates": []}},
                    {"protein": {"id": "Y", "sequence": "AAAAAA", "unpairedMsa": "OLD", "templates": []}},
                ],
            }
        )
    )

    def service(name):
        return {"url": f"https://{name}.test", "poll_interval": 0.001}

    chains = {"target": ["A"], "binder": ["B"], "refold_map": {"A": "X", "B": "Y"}}
    pdb(tmp_path / "input.pdb")
    (tmp_path / "frameworks").mkdir()
    (tmp_path / "target.csv").write_text("id,sequence\ntarget,GGGGGG\n")
    pdb(tmp_path / "framework.cif.pdb")
    return InitialRoundConfig.model_validate(
        {
            "name": "test",
            "work_dir": str(tmp_path),
            "path_mappings": [{"local": str(tmp_path), "remote": "/shared"}],
            "generators": {
                "ppiflow": {
                    "service": service("ppiflow"),
                    "request": {
                        "mode": "nanobody",
                        "antigen_pdb": "/shared/input.pdb",
                        "antigen_chain": "A",
                        "framework_dir": "/shared/frameworks",
                        "heavy_chain": "B",
                    },
                    "chains": chains,
                    "fold_template": str(template),
                },
                "torchcraft": {
                    "service": service("torchcraft"),
                    "request": {
                        "mode": "vhh",
                        "pdb_path": "/shared/input.pdb",
                        "hotspot_indices": "1,2",
                        "target_full_seq": "/shared/target.csv",
                        "framework_cif_path": "/shared/framework.cif.pdb",
                    },
                    "chains": chains,
                    "fold_template": str(template),
                },
            },
            "evaluation": {
                "af3score": service("af3score"),
                "torchfold": service("torchfold"),
                "rosetta": service("rosetta"),
                "dockq_python": sys.executable,
            },
        }
    )


def test_shared_paths_refuse_traversal_and_symlink_escape(tmp_path):
    files = SharedFiles([PathMapping(local=tmp_path / "mount", remote="/shared")])
    with pytest.raises(ValueError, match="escapes"):
        files.local("/shared/../outside")
    (tmp_path / "mount").mkdir()
    (tmp_path / "mount/link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        files.local("/shared/link/outside")
    with pytest.raises(ValueError, match="no shared"):
        files.local("/other/file")


@pytest.mark.parametrize("raw", [True, "NaN", "inf", 1.1, -0.1])
def test_confidence_values_are_real_bounded_measurements(raw):
    with pytest.raises(ValueError):
        measured([{"iptm": raw}], "iptm", confidence=True)
    assert measured([{"iptm": 0}], "iptm", confidence=True) == 0


def test_multiple_samples_are_not_silently_collapsed():
    with pytest.raises(ValueError, match="exactly one"):
        measured([{"iptm": 0.9}, {"iptm": 0.1}], "iptm", confidence=True)


async def test_submission_timeout_is_not_retried_and_can_be_reconciled(tmp_path):
    checkpoint = Checkpoint(tmp_path, {})
    posts = []

    def handler(request):
        if request.method == "POST":
            posts.append(request)
            raise httpx.ReadTimeout("lost response", request=request)
        if request.url.path.endswith("results"):
            return httpx.Response(200, json={"files": []})
        return httpx.Response(200, json={"status": "COMPLETED", "exit_code": "0:0"})

    service = SlurmService(Service(url="https://service.test"), checkpoint, transport=httpx.MockTransport(handler))
    try:
        for _ in range(2):
            with pytest.raises(SubmissionUncertainError):
                await service.run("generate/ppiflow", "/v1/designs", "design_id", {})
        result = await service.run("generate/ppiflow", "/v1/designs", "design_id", {}, attach_id="recovered")
        assert result == {"files": []}
        assert len(posts) == 1
        saved = json.loads(checkpoint.path.read_text())["jobs"]["generate/ppiflow"]
        assert "ReadTimeout: lost response" in saved["submission_error"]
        assert saved["id"] == "recovered"
    finally:
        await service.close()


async def test_attach_without_checkpoint_never_submits(tmp_path):
    requests = []
    service = SlurmService(
        Service(url="https://service.test"),
        Checkpoint(tmp_path, {}),
        transport=httpx.MockTransport(lambda request: requests.append(request)),
    )
    try:
        with pytest.raises(ValueError, match="must exist in checkpoint"):
            await service.run("generate/ppiflow", "/v1/designs", "design_id", {}, attach_id="known")
        assert requests == []
    finally:
        await service.close()


@pytest.mark.parametrize("recover_on_resume", [False, True])
async def test_lost_response_reconciles_exact_submission_without_reposting(tmp_path, recover_on_resume):
    from opendde_harness.plugin.protein_design.initial_round.storage import fingerprint

    checkpoint = Checkpoint(tmp_path, {})
    posts = []
    available = not recover_on_resume

    def handler(request):
        if request.method == "POST":
            posts.append(request)
            raise httpx.ReadTimeout("lost after persistence", request=request)
        if request.url.path.endswith("/results"):
            return httpx.Response(200, json={"files": []})
        if not available:
            return httpx.Response(503)
        key = posts[0].headers["Idempotency-Key"]
        return httpx.Response(
            200,
            json={
                "design_id": key,
                "submission_key": key,
                "request_hash": fingerprint({}),
                "status": "COMPLETED",
                "exit_code": "0:0",
            },
        )

    service = SlurmService(Service(url="https://service.test"), checkpoint, transport=httpx.MockTransport(handler))
    try:
        if recover_on_resume:
            with pytest.raises(SubmissionUncertainError):
                await service.run("generate/ppiflow", "/v1/designs", "design_id", {})
            await service.close()
            checkpoint = Checkpoint(tmp_path, {})
            service = SlurmService(
                Service(url="https://service.test"), checkpoint, transport=httpx.MockTransport(handler)
            )
            available = True
        assert await service.run("generate/ppiflow", "/v1/designs", "design_id", {}) == {"files": []}
        assert len(posts) == 1
        assert checkpoint.data["jobs"]["generate/ppiflow"]["id"] == posts[0].headers["Idempotency-Key"]
    finally:
        await service.close()


async def test_reconciliation_rejects_mismatched_request_hash(tmp_path):
    def handler(request):
        if request.method == "POST":
            raise httpx.ReadTimeout("lost", request=request)
        key = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json={"design_id": key, "submission_key": key, "request_hash": "wrong"})

    checkpoint = Checkpoint(tmp_path, {})
    service = SlurmService(Service(url="https://service.test"), checkpoint, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(SubmissionUncertainError):
            await service.run("generate/ppiflow", "/v1/designs", "design_id", {})
        assert "id" not in checkpoint.data["jobs"]["generate/ppiflow"]
    finally:
        await service.close()


async def test_server_submission_uncertainty_stops_polling_without_reposting(tmp_path):
    posts = []

    def handler(request):
        if request.method == "POST":
            posts.append(request)
            return httpx.Response(202, json={"design_id": "known"})
        return httpx.Response(200, json={"status": "SUBMISSION_UNCERTAIN", "submission_error": "sbatch timed out"})

    checkpoint = Checkpoint(tmp_path, {})
    service = SlurmService(Service(url="https://service.test"), checkpoint, transport=httpx.MockTransport(handler))
    try:
        for _ in range(2):
            with pytest.raises(SubmissionUncertainError, match="Slurm submission is uncertain"):
                await service.run("generate/ppiflow", "/v1/designs", "design_id", {})
        assert len(posts) == 1
        assert checkpoint.data["jobs"]["generate/ppiflow"]["id"] == "known"
    finally:
        await service.close()


@pytest.mark.parametrize("state", ["completed", "failed"])
async def test_attach_does_not_reset_terminal_job(tmp_path, state):
    from opendde_harness.plugin.protein_design.initial_round.storage import fingerprint

    checkpoint = Checkpoint(tmp_path, {})
    checkpoint.data["jobs"]["generate/ppiflow"] = {
        "request_hash": fingerprint({"url": "https://service.test", "route": "/v1/designs", "payload": {}}),
        "id": "known",
        "state": state,
        "results": {"files": []},
    }
    requests = []
    service = SlurmService(
        Service(url="https://service.test"),
        checkpoint,
        transport=httpx.MockTransport(lambda request: requests.append(request)),
    )
    try:
        if state == "failed":
            with pytest.raises(RemoteJobFailedError, match="persisted failed"):
                await service.run("generate/ppiflow", "/v1/designs", "design_id", {}, attach_id="known")
        else:
            assert await service.run("generate/ppiflow", "/v1/designs", "design_id", {}, attach_id="known") == {
                "files": []
            }
        assert requests == []
    finally:
        await service.close()


@pytest.mark.parametrize("status,exit_code", [("FAILED", "0:0"), ("COMPLETED", "1:0"), ("COMPLETED", None)])
async def test_success_requires_terminal_state_and_exit_code(tmp_path, status, exit_code):
    reads = []

    def handler(request):
        reads.append(request.url.path)
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "job"})
        return httpx.Response(200, json={"status": status, "exit_code": exit_code})

    service = SlurmService(
        Service(url="https://service.test"), Checkpoint(tmp_path, {}), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(RemoteJobFailedError):
            await service.run("score", "/v1/tasks", "task_id", {})
        assert not any(p.endswith("/results") for p in reads)
    finally:
        await service.close()


async def test_unknown_and_queue_states_are_polled_before_results(tmp_path):
    states = iter(["UNKNOWN", "PENDING", "RUNNING", "COMPLETED"])
    calls = Counter()

    def handler(request):
        calls[request.method] += 1
        if request.method == "POST":
            return httpx.Response(202, json={"fold_id": "job"})
        if request.url.path.endswith("results"):
            return httpx.Response(200, json={"ready": True})
        return httpx.Response(200, json={"status": next(states), "exit_code": "0:0"})

    service = SlurmService(
        Service(url="https://service.test", poll_interval=0.001),
        Checkpoint(tmp_path, {}),
        transport=httpx.MockTransport(handler),
    )
    try:
        assert await service.run("fold", "/v1/folds", "fold_id", {}) == {"ready": True}
        assert await service.run("fold", "/v1/folds", "fold_id", {}) == {"ready": True}
        assert calls == {"POST": 1, "GET": 5}
    finally:
        await service.close()


def test_fold_input_updates_binder_without_reusing_stale_msa(round_config, tmp_path):
    generator = round_config.generators["ppiflow"]
    output = tmp_path / "fold.json"
    prepare_fold(generator.fold_template, {"A": "GGGGGG", "B": "VVVVVV"}, generator.chains, "new", output)
    value = json.loads(output.read_text())
    assert value["sequences"][1]["protein"] == {
        "id": "Y",
        "sequence": "VVVVVV",
        "unpairedMsa": "",
        "pairedMsa": "",
        "templates": [],
    }
    assert json.loads(generator.fold_template.read_text())["sequences"][1]["protein"]["unpairedMsa"] == "OLD"
    with pytest.raises(ValueError, match="target chain"):
        prepare_fold(generator.fold_template, {"A": "WRONG", "B": "AAAAAA"}, generator.chains, "bad", output)


def test_refold_pairs_model_with_its_own_confidence(tmp_path):
    for name, score in [("a_", 0.2), ("b_", 0.9)]:
        (tmp_path / f"{name}summary_confidences.json").write_text(json.dumps({"iptm": score}))
        (tmp_path / f"{name}model.cif").write_text("placeholder")
    files = SharedFiles([PathMapping(local=tmp_path, remote="/shared")])
    result = {"files": [files.remote(p) for p in tmp_path.iterdir()]}
    path, score = refold_output(result, Service(url="https://fold.test"), files, "*summary_confidences*.json")
    assert path.name == "b_model.cif" and score == 0.9


@pytest.fixture
def fake_services(round_config, tmp_path, monkeypatch):
    calls = Counter()
    outputs = {}
    files = SharedFiles(round_config.path_mappings)
    for source in round_config.generators:
        pdb(tmp_path / source / "generated.pdb")

    # Keep protocol tests independent of the optional isolated DockQ installation.
    async def fake_process(*args, **kwargs):
        class Process:
            returncode = 0

            async def communicate(self):
                payload = {"dockq_available": True} if args[-1] == "--check" else {"dockq": 1.0, "interfaces": []}
                return json.dumps(payload).encode(), b""

        return Process()

    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.initial_round.runner.asyncio.create_subprocess_exec", fake_process
    )

    def handler(request):
        source = request.url.host.split(".")[0]
        calls[(source, request.method)] += 1
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"ok": True})
        if request.method == "POST":
            body = json.loads(request.content)
            job = source + str(calls[(source, "POST")])
            if source in round_config.generators:
                outputs[job] = {"files": [f"/shared/{source}/generated.pdb"]}
                return httpx.Response(202, json={"design_id": job})
            if source == "torchfold":
                prepared = json.loads(files.local(body["json_path"]).read_text())
                assert prepared["sequences"][1]["protein"]["sequence"] == "AAAAAA"
                out = tmp_path / job
                pdb(out / "model.pdb", target="X", chain="Y", light="Z" if len(prepared["sequences"]) == 3 else None)
                (out / "summary_confidences.json").write_text(json.dumps({"iptm": 0.95}))
                outputs[job] = {
                    "files": [files.remote(out / "model.pdb"), files.remote(out / "summary_confidences.json")]
                }
                return httpx.Response(202, json={"fold_id": job})
            if source == "rosetta":
                assert body["chain_group_a"] == (
                    "YZ" if round_config.generators["ppiflow"].request["mode"] == "antibody" else "Y"
                )
                assert body["chain_group_b"] == "X"
                outputs[job] = {"metrics": {"dG_separated": -12.0}}
            else:
                assert body["mode"] == "pdb" and body["num_samples"] == 1
                outputs[job] = {"files": []}
            return httpx.Response(202, json={"task_id": job})
        job = path.split("/")[3]
        if path.endswith("/results"):
            return httpx.Response(200, json=outputs[job])
        if path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": [{"iptm": 0.9}]})
        return httpx.Response(200, json={"status": "COMPLETED", "exit_code": "0:0"})

    return httpx.MockTransport(handler), calls


async def test_both_generators_complete_round_and_resume_without_resubmitting(round_config, fake_services):
    transport, calls = fake_services
    result = await InitialRoundRunner(round_config, transport=transport).run()
    assert result["status"] == "completed" and result["accepted"] == 2
    root = round_config.work_dir / round_config.name
    candidates = json.loads((root / "initial_candidates.json").read_text())
    assert {c["metadata"]["generator"] for c in candidates} == {"ppiflow", "torchcraft"}
    assert all(c["metrics"]["design_refold_dockq"] == 1.0 for c in candidates)
    before = {k: v for k, v in calls.items() if k[1] == "POST"}
    assert before == {
        ("ppiflow", "POST"): 1,
        ("torchcraft", "POST"): 1,
        ("af3score", "POST"): 2,
        ("torchfold", "POST"): 2,
        ("rosetta", "POST"): 2,
    }
    assert await InitialRoundRunner(round_config, transport=transport).run() == result
    assert {k: v for k, v in calls.items() if k[1] == "POST"} == before


async def test_check_submits_nothing(round_config, fake_services):
    transport, calls = fake_services
    result = await InitialRoundRunner(round_config, transport=transport).run(check_only=True)
    assert result["status"] == "checked" and not result["inference_verified"]
    assert not any(method == "POST" for _, method in calls)


async def test_thresholds_produce_no_hits_without_fabricating_results(round_config, fake_services):
    transport, calls = fake_services
    round_config.evaluation.min_af3score_iptm = 0.99
    result = await InitialRoundRunner(round_config, transport=transport).run()
    assert result["accepted"] == 0 and not result["ranking"]
    assert calls[("torchfold", "POST")] == 0


async def test_resume_refuses_modified_generated_artifact(round_config, fake_services):
    transport, _ = fake_services
    await InitialRoundRunner(round_config, transport=transport).run()
    (round_config.work_dir / "ppiflow/generated.pdb").write_text("changed")
    result = await InitialRoundRunner(round_config, transport=transport).run()
    assert result["status"] == "partial" and len(result["failed"]) == 1


def test_checkpoint_refuses_changed_configuration(tmp_path):
    checkpoint = Checkpoint(tmp_path, {"model": "first"})
    checkpoint.data["jobs"]["submitted"] = {"id": "existing"}
    checkpoint.save()
    with pytest.raises(ValueError, match="configuration changed"):
        Checkpoint(tmp_path, {"model": "second"})


def test_chain_mapping_must_be_bijective():
    with pytest.raises(ValueError, match="map every"):
        Chains(target=["A"], binder=["B"], refold_map={"A": "X", "B": "X"})


async def test_bad_result_field_is_a_candidate_error_not_a_zero_score(round_config, fake_services):
    transport, _ = fake_services
    round_config.evaluation.af3score.metric_fields = {"iptm": "missing_measurement"}
    result = await InitialRoundRunner(round_config, transport=transport).run()
    assert result["status"] == "partial" and result["accepted"] == 0
    candidates = json.loads((round_config.work_dir / round_config.name / "candidates.json").read_text())
    assert all("af3score_iptm" not in c["metrics"] for c in candidates)


async def test_changed_template_cannot_resume_submitted_jobs(round_config, fake_services):
    transport, _ = fake_services
    await InitialRoundRunner(round_config, transport=transport).run()
    template = round_config.generators["ppiflow"].fold_template
    value = json.loads(template.read_text())
    value["modelSeeds"] = [99]
    template.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="configuration changed"):
        await InitialRoundRunner(round_config, transport=transport).run()


async def test_one_failed_generator_still_checks_the_other_source(round_config, fake_services):
    transport, calls = fake_services
    original = transport.handler

    def handler(request):
        if request.url.host == "ppiflow.test" and request.url.path.endswith("/results"):
            return httpx.Response(200, json={"files": []})
        return original(request)

    transport.handler = handler
    with pytest.raises(RuntimeError, match="all configured generators are required"):
        await InitialRoundRunner(round_config, transport=transport).run()
    assert calls[("torchcraft", "POST")] == 1
    assert calls[("af3score", "POST")] == 0


async def test_nonstandard_response_fields_are_explicitly_configurable(tmp_path):
    def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"fold_id": "job"})
        if request.url.path.endswith("/results"):
            return httpx.Response(200, json={"files": []})
        return httpx.Response(200, json={"slurm": {"state": "COMPLETED", "exit": 0}})

    service = SlurmService(
        Service(url="https://service.test", status_field="slurm.state", exit_code_field="slurm.exit"),
        Checkpoint(tmp_path, {}),
        transport=httpx.MockTransport(handler),
    )
    try:
        assert await service.run("fold", "/v1/folds", "fold_id", {}) == {"files": []}
    finally:
        await service.close()


def test_check_only_checkpoint_can_be_reconfigured_before_any_submission(tmp_path):
    Checkpoint(tmp_path, {"threshold": 0.8})
    Checkpoint(tmp_path, {"threshold": 0.9})


@pytest.mark.parametrize(
    "change", [{"job_timeout": float("inf")}, {"poll_interval": 0}, {"url": "https://user:secret@host.test"}]
)
def test_service_configuration_rejects_invalid_limits_and_embedded_credentials(change):
    with pytest.raises(ValueError):
        Service.model_validate({"url": "https://service.test", **change})


async def test_generator_can_probe_openapi_when_health_route_is_not_documented(round_config, fake_services):
    transport, calls = fake_services
    original = transport.handler
    for generator in round_config.generators.values():
        generator.service.health_path = "/openapi.json"

    def handler(request):
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json={"openapi": "3.1.0", "paths": {"/v1/designs": {}}})
        return original(request)

    transport.handler = handler
    result = await InitialRoundRunner(round_config, transport=transport).run(check_only=True)
    assert result["status"] == "checked"
    assert not any(method == "POST" for _, method in calls)


def test_example_configuration_parses_without_optional_model_imports():
    config = InitialRoundConfig.load(Path(__file__).parents[1] / "docs/examples/initial_round_sbatch.yaml")
    assert set(config.generators) == {"ppiflow"}
    assert config.generators["ppiflow"].service.health_path == "/openapi.json"


async def test_updated_af3score_controls_are_forwarded_in_initial_round(round_config, fake_services):
    transport, _ = fake_services
    original = transport.handler
    options = {
        "run_data_pipeline": True,
        "run_inference": True,
        "init_guess": False,
        "num_workers": 2,
        "seed": 37,
        "write_fold_input_json_file": True,
    }
    round_config.evaluation.af3score_options = options
    bodies = []

    def handler(request):
        if request.method == "POST" and request.url.host == "af3score.test":
            bodies.append(json.loads(request.content))
        return original(request)

    transport.handler = handler
    assert (await InitialRoundRunner(round_config, transport=transport).run())["accepted"] == 2
    assert len(bodies) == 2
    assert all(all(body[key] == value for key, value in options.items()) for body in bodies)


async def test_ppiflow_only_runs_all_stages_without_torchcraft_rosetta_or_template(round_config, fake_services):
    transport, calls = fake_services
    round_config.generators.pop("torchcraft")
    round_config.generators["ppiflow"].fold_template = None
    round_config.evaluation.rosetta = None
    result = await InitialRoundRunner(round_config, transport=transport).run()
    assert result["status"] == "completed" and result["accepted"] == 1
    assert calls[("ppiflow", "POST")] == calls[("af3score", "POST")] == calls[("torchfold", "POST")] == 1
    assert calls[("torchcraft", "GET")] == calls[("rosetta", "GET")] == 0
    candidate = json.loads((round_config.work_dir / round_config.name / "candidates.json").read_text())[0]
    assert candidate["metrics"]["design_refold_dockq"] == 1.0
    assert "rosetta_interface_dg" not in candidate["metrics"]


async def test_af3score_must_be_strictly_above_the_gate_before_refold(round_config, fake_services):
    transport, calls = fake_services
    round_config.generators.pop("torchcraft")
    round_config.evaluation.rosetta = None
    round_config.evaluation.min_af3score_iptm = 0.9
    result = await InitialRoundRunner(round_config, transport=transport).run()
    assert result["outcome"] == "no_hits"
    assert calls[("torchfold", "POST")] == 0


async def test_every_passing_structure_can_refold_without_a_top_k_cap(round_config, fake_services):
    transport, calls = fake_services
    round_config.generators.pop("torchcraft")
    round_config.generators["ppiflow"].fold_template = None
    round_config.generators["ppiflow"].per_source_refold_limit = None
    round_config.generators["ppiflow"].deduplicate_sequences = False
    round_config.evaluation.rosetta = None
    for i in range(7):
        pdb(round_config.work_dir / "ppiflow" / f"extra-{i}.pdb")
    original = transport.handler

    def handler(request):
        if request.url.host == "ppiflow.test" and request.url.path.endswith("/results"):
            return httpx.Response(
                200,
                json={
                    "files": [
                        f"/shared/ppiflow/{path.name}" for path in (round_config.work_dir / "ppiflow").glob("*.pdb")
                    ]
                },
            )
        return original(request)

    transport.handler = handler
    result = await InitialRoundRunner(round_config, transport=transport).run()
    assert result["accepted"] == 8
    assert calls[("torchfold", "POST")] == 8


def test_generated_fold_input_supports_query_only_or_pipeline_search(round_config, tmp_path):
    chains = round_config.generators["ppiflow"].chains
    for pipeline in (False, True):
        output = tmp_path / f"auto-{pipeline}.json"
        prepare_fold(
            None, {"A": "GGGGGG", "B": "AAAAAA"}, chains, "auto", output, seeds=[7], run_data_pipeline=pipeline
        )
        content = json.loads(output.read_text())
        assert content["modelSeeds"] == [7]
        assert all(("unpairedMsa" not in entity["protein"]) == pipeline for entity in content["sequences"])


@pytest.fixture
def antibody_config(round_config):
    value = round_config.model_dump(mode="json")
    value["generators"].pop("torchcraft")
    generator = value["generators"]["ppiflow"]
    generator["request"].update(mode="antibody", name="paired", light_chain="L", cdr_length="CDRH3,10-20,CDRL3,9-11")
    generator["chains"] = {"target": ["A"], "binder": ["B", "L"], "refold_map": {"A": "X", "B": "Y", "L": "Z"}}
    generator["fold_template"] = None
    return InitialRoundConfig.model_validate(value)


def test_antibody_contract_matches_model_api_and_chain_roles(antibody_config):
    from opendde_harness.plugin.protein_design.core.model_requests import validate_request

    generator = antibody_config.generators["ppiflow"]
    assert validate_request("ppiflow", generator.request)["light_chain"] == "L"
    for field, bad, message in [
        ("light_chain", "B", "distinct"),
        ("antigen_chain", "Q", "target chains"),
        ("cdr_length", "", "requires"),
    ]:
        value = antibody_config.model_dump(mode="json")
        value["generators"]["ppiflow"]["request"][field] = bad
        with pytest.raises(ValueError, match=message):
            InitialRoundConfig.model_validate(value)


@pytest.mark.parametrize("template", [False, True])
def test_antibody_fold_keeps_all_chains_and_clears_both_binder_features(antibody_config, tmp_path, template):
    generator = antibody_config.generators["ppiflow"]
    output = tmp_path / "paired_fold.json"
    sequences = {"A": "GGGGGG", "B": "AAAAAA", "L": "VVVVVV"}
    prepare_fold(None, sequences, generator.chains, "paired", output)
    if template:
        value = json.loads(output.read_text())
        for entity in value["sequences"][1:]:
            entity["protein"].update(sequence="OLD", unpairedMsa="OLD", pairedMsa="OLD", templates=[{}])
        source = tmp_path / "paired_template.json"
        source.write_text(json.dumps(value))
        prepare_fold(source, sequences, generator.chains, "paired", output)
    proteins = {entity["protein"]["id"]: entity["protein"] for entity in json.loads(output.read_text())["sequences"]}
    assert {key: protein["sequence"] for key, protein in proteins.items()} == {
        "X": "GGGGGG",
        "Y": "AAAAAA",
        "Z": "VVVVVV",
    }
    for chain in ("Y", "Z"):
        assert proteins[chain]["unpairedMsa"] == proteins[chain]["pairedMsa"] == ""
        assert proteins[chain]["templates"] == []


async def test_antibody_round_exports_both_sequences_and_resumes(
    antibody_config, round_config, fake_services, tmp_path
):
    # The protocol fixture observes this config for Rosetta grouping assertions.
    round_config.generators = antibody_config.generators
    pdb(tmp_path / "ppiflow/generated.pdb", light="L")
    transport, calls = fake_services
    result = await InitialRoundRunner(antibody_config, transport=transport).run()
    assert result["accepted"] == 1 and result["status"] == "completed"
    root = antibody_config.work_dir / antibody_config.name
    candidate = json.loads((root / "initial_candidates.json").read_text())[0]
    assert candidate["sequence"] == "AAAAAA|VVVVVV"
    assert set(candidate["metadata"]["chains"]) == {"A", "B", "L"}
    before = calls.copy()
    assert await InitialRoundRunner(antibody_config, transport=transport).run() == result
    assert {k: v for k, v in calls.items() if k[1] == "POST"} == {k: v for k, v in before.items() if k[1] == "POST"}


def test_antibody_dockq_scores_target_interfaces_not_heavy_light(monkeypatch):
    from types import ModuleType

    from opendde_harness.plugin.protein_design.initial_round.dockq import evaluate

    package = ModuleType("DockQ")
    module = ModuleType("DockQ.DockQ")
    module.load_PDB = lambda path: path

    def score(model, native, *, chain_map):
        assert (model, native) == ("refold.pdb", "design.pdb")
        assert chain_map == {"C": "X", "A": "Y", "B": "Z"}
        return {("C", "A"): {"DockQ": 0.8}, ("B", "C"): {"DockQ": 0.7}, ("A", "B"): {"DockQ": 0.1}}, None

    module.run_on_all_native_interfaces = score
    monkeypatch.setitem(sys.modules, "DockQ", package)
    monkeypatch.setitem(sys.modules, "DockQ.DockQ", module)
    result = evaluate("refold.pdb", "design.pdb", {"C": "X", "A": "Y", "B": "Z"}, ["C"], ["A", "B"])
    assert result["dockq"] == 0.7
    assert len(result["interfaces"]) == 2


async def test_sbatch_api_startup_is_lazy_and_check_remains_read_only(
    round_config, fake_services, tmp_path, monkeypatch
):
    transport, calls = fake_services
    script = tmp_path / "ppiflow.sbatch"
    script.write_text("#!/bin/bash\n")
    round_config.generators["ppiflow"].service.sbatch_script = script
    starts = []

    async def ensure(config, client, *, start=True):
        if config.sbatch_script == script:
            starts.append(start)
            assert calls[("ppiflow", "POST")] == 0
        client.base_url = httpx.URL(config.url.rstrip("/") + "/")

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.services.ensure_api", ensure)
    checked = await InitialRoundRunner(round_config, transport=transport).run(check_only=True)
    assert checked["status"] == "checked" and starts == []
    starts.clear()
    result = await InitialRoundRunner(round_config, transport=transport).run()
    assert result["accepted"] == 2 and starts == [True]


async def test_ppiflow_cli_options_are_forwarded_unchanged(round_config, fake_services):
    transport, _ = fake_services
    extras = {
        "config": "/shared/test_antibody_v3.yaml",
        "model_weights": "/shared/model_214.ckpt.enc",
        "output_dir": "/shared/sample_antibody/3/",
    }
    round_config.generators["ppiflow"].request.update(extras)
    original = transport.handler
    bodies = []

    def handler(request):
        if request.method == "POST" and request.url.host == "ppiflow.test":
            bodies.append(json.loads(request.content))
        return original(request)

    transport.handler = handler
    assert (await InitialRoundRunner(round_config, transport=transport).run())["accepted"] == 2
    assert len(bodies) == 1
    assert all(bodies[0][key] == value for key, value in extras.items())


async def test_cold_sbatch_preflight_never_probes_http_or_slurm(round_config, fake_services, tmp_path, monkeypatch):
    _, calls = fake_services
    for name, service in [
        *((name, g.service) for name, g in round_config.generators.items()),
        *((name, getattr(round_config.evaluation, name)) for name in ("af3score", "torchfold", "rosetta")),
    ]:
        service.sbatch_script = tmp_path / f"{name}.sbatch"
        service.sbatch_script.write_text("#!/bin/bash\n")

    def forbid_http(request):
        pytest.fail("sbatch-managed preflight must not probe HTTP")

    async def forbid_start(*args, **kwargs):
        pytest.fail("sbatch-managed preflight must not resolve or submit Slurm allocations")

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.services.ensure_api", forbid_start)
    result = await InitialRoundRunner(round_config, transport=httpx.MockTransport(forbid_http)).run(check_only=True)
    assert result["status"] == "checked" and not result["inference_verified"]
    assert all(v == {"reachable": None, "startup": "deferred_until_first_request"} for v in result["services"].values())
    assert not calls


async def test_dockq_preflight_uses_configured_python_and_preserves_actual_error(round_config, tmp_path, monkeypatch):
    commands = []

    async def process(*args, **kwargs):
        commands.append(args)

        class Failed:
            returncode = 1

            async def communicate(self):
                return b"", b"ModuleNotFoundError: No module named 'numpy'"

        return Failed()

    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.initial_round.runner.asyncio.create_subprocess_exec", process
    )
    with pytest.raises(RuntimeError, match="DockQ.DockQ"):
        await InitialRoundRunner(round_config).run(check_only=True)
    assert commands[0][0] == str(round_config.evaluation.dockq_python)
    assert commands[0][1].endswith("/initial_round/dockq.py") and commands[0][2] == "--check"
    log = tmp_path / round_config.name / "dockq-check.log"
    assert "No module named 'numpy'" in log.read_text()


@pytest.mark.parametrize("threshold,expected", [(0.85, 2), (0.9, 0)])
async def test_torchscore_is_an_independent_initial_round_selection_backend(
    round_config, fake_services, threshold, expected
):
    transport, calls = fake_services
    evaluation = round_config.evaluation
    evaluation.scoring_backend = "torchscore"
    evaluation.torchscore = Service(url="https://torchscore.test")
    evaluation.af3score = None
    evaluation.min_torchscore_iptm = threshold
    result = await InitialRoundRunner(round_config, transport=transport).run()
    assert result["accepted"] == expected
    assert calls[("torchscore", "POST")] == 2
    assert calls[("af3score", "POST")] == 0
    assert calls[("torchfold", "POST")] == expected
    root = round_config.work_dir / round_config.name
    candidates = json.loads((root / "candidates.json").read_text())
    assert all(c["metrics"]["torchscore_iptm"] == 0.9 and "af3score_iptm" not in c["metrics"] for c in candidates)
    assert "torchscore_iptm" in (root / "ranking.csv").read_text().splitlines()[0]
    before = dict(calls)
    assert (await InitialRoundRunner(round_config, transport=transport).run())["accepted"] == expected
    assert all(calls[key] == value for key, value in before.items() if key[1] == "POST")


def test_default_af3score_serialization_preserves_existing_checkpoint_shape(round_config):
    dumped = round_config.model_dump(mode="json")["evaluation"]
    assert "scoring_backend" not in dumped
    assert not any(key in dumped for key in ("torchscore", "torchscore_options", "min_torchscore_iptm"))
    from opendde_harness.plugin.protein_design.initial_round.config import Evaluation

    assert Evaluation.model_validate(dumped).model_dump(mode="json") == dumped
    with pytest.raises(ValueError, match="torchscore.*required"):
        Evaluation.model_validate({**dumped, "scoring_backend": "torchscore"})


def test_torchscore_round_rejects_msa_search_and_requires_inference(round_config):
    from opendde_harness.plugin.protein_design.initial_round.config import Evaluation

    value = round_config.model_dump(mode="json")["evaluation"]
    value.update(scoring_backend="torchscore", torchscore={"url": "https://score.test"})
    for options in ({"run_data_pipeline": True}, {"run_inference": False}, {"seed": 11}, {"mode": "metrics"}):
        with pytest.raises(ValueError):
            Evaluation.model_validate({**value, "torchscore_options": options})


def autonomous_config(config, **overrides):
    value = config.model_dump(mode="json")
    value["generators"]["ppiflow"]["request"]["specified_hotspots"] = "A:1"
    value["autonomous"] = {"decision_policy": "rules", "max_rounds": 3, "max_samples": 8, "batch_size": 1, **overrides}
    return InitialRoundConfig.model_validate(value)


async def test_autonomous_round_preserves_best_budget_and_resume(round_config, fake_services):
    transport, calls = fake_services
    config = autonomous_config(round_config)
    result = await InitialRoundRunner(config, transport=transport).run()
    assert result["ranking_policy"] == "100*af3score_iptm-rosetta_interface_dg_desc"
    assert result["stop_reason"] == "no_improvement"
    assert result["used_samples"] == 6  # two initial samples and two follow-up batches of two
    assert result["accepted"] == 4
    root = config.work_dir / config.name
    records = json.loads((root / "candidates.json").read_text())
    assert all(c["metrics"]["ranking_score"] == 102 for c in records)
    assert all(c["diagnostics"]["epitope_hit"] is True for c in records)
    assert all("generated.pdb" in c["generated_structure"] for c in records)
    decisions = json.loads((root / "decisions.json").read_text())["decisions"]
    assert [d["source"] for d in decisions] == ["ppiflow", "torchcraft"]
    assert all(d["action"] == "expand" for d in decisions)
    before = {key: count for key, count in calls.items() if key[1] == "POST"}
    assert await InitialRoundRunner(config, transport=transport).run() == result
    assert {key: count for key, count in calls.items() if key[1] == "POST"} == before


async def test_autonomous_sample_budget_counts_initial_jobs(round_config, fake_services):
    config = autonomous_config(round_config, max_samples=2)
    result = await InitialRoundRunner(config, transport=fake_services[0]).run()
    assert result["used_samples"] == 2
    assert result["followup_rounds"] == 0
    assert result["stop_reason"] == "sample_budget"
    with pytest.raises(ValueError, match="exceed"):
        autonomous_config(round_config, max_samples=1)


async def test_autonomous_rescues_low_scores_and_ignores_legacy_refold_gate(round_config, fake_services):
    config = autonomous_config(round_config, min_af3score=0.95, max_samples=2)
    config.evaluation.min_refold_iptm = 0.99
    runner = InitialRoundRunner(config, transport=fake_services[0])
    result = await runner.run()
    assert result["accepted"] == 0
    records = runner.checkpoint.data["candidates"]
    assert all(c["diagnostics"]["diagnosis"] == "low_confidence" for c in records)
    assert all(c["metrics"]["design_refold_dockq"] == 1 for c in records)
    assert "torchopt" in result["unavailable_actions"]


async def test_invalid_agent_choice_falls_back_before_submission(round_config, fake_services, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.autonomous import Choice

    async def bad_choice(context):
        return Choice(option_id=999, reason="invalid selection")

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.autonomous.agent_choice", bad_choice)
    config = autonomous_config(round_config, decision_policy="agent", max_rounds=1, max_samples=3)
    runner = InitialRoundRunner(config, transport=fake_services[0])
    result = await runner.run()
    decision = runner.checkpoint.data["autonomous"]["decisions"][0]
    assert decision["fallback"] and decision["samples"] == 1
    assert result["used_samples"] == 3


async def test_followup_resumes_reserved_decision_without_duplicate_submission(
    round_config, fake_services, monkeypatch
):
    from opendde_harness.plugin.protein_design.initial_round.autonomous import evaluate

    once = True

    async def interrupted_evaluation(runner, candidates):
        nonlocal once
        if any(c.get("round_index") for c in candidates) and once:
            once = False
            raise RuntimeError("worker interrupted after generation")
        return await evaluate(runner, candidates)

    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.initial_round.autonomous.evaluate", interrupted_evaluation
    )
    config = autonomous_config(round_config, max_rounds=1, max_samples=3)
    transport, calls = fake_services
    with pytest.raises(RuntimeError, match="interrupted"):
        await InitialRoundRunner(config, transport=transport).run()
    before = calls[("ppiflow", "POST")]
    result = await InitialRoundRunner(config, transport=transport).run()
    assert calls[("ppiflow", "POST")] == before == 2
    assert result["used_samples"] == 3
    assert result["followup_rounds"] == 1


def test_diagnosis_thresholds_and_missing_evidence_are_explicit():
    from opendde_harness.plugin.protein_design.initial_round.config import AutonomousRound
    from opendde_harness.plugin.protein_design.initial_round.diagnostics import classify

    config = AutonomousRound()
    metrics = {"af3score_iptm": 0.9, "design_refold_dockq": 0.8, "rosetta_interface_dg": -10}
    evidence = {"epitope_hit": True}
    assert classify(metrics, evidence, config) == "hit"
    assert classify({**metrics, "af3score_iptm": 0.8}, evidence, config) == "low_confidence"
    assert classify({**metrics, "design_refold_dockq": 0.49}, evidence, config) == "pose_change"
    assert (
        classify({**metrics, "design_refold_dockq": 0.1}, {**evidence, "local_region": "H3"}, config) == "local_change"
    )
    assert classify(metrics, {"epitope_hit": False}, config) == "off_epitope"
    assert classify(metrics, {"epitope_hit": None}, config) == "uncertain"
    assert classify({**metrics, "rosetta_interface_dg": None}, evidence, config) == "evaluation_incomplete"


def test_structure_diagnosis_maps_chains_and_localizes_one_region(round_config, tmp_path):
    import biotite.structure.io as structure_io

    from opendde_harness.plugin.protein_design.initial_round.artifacts import structure
    from opendde_harness.plugin.protein_design.initial_round.diagnostics import analyze

    config = autonomous_config(round_config)
    config.generators["ppiflow"].request["cdr_position"] = "B5-6"
    runner = InitialRoundRunner(config)
    generated = pdb(tmp_path / "generated.pdb")
    folded = pdb(tmp_path / "folded.pdb", target="X", chain="Y")
    atoms = structure(folded)
    atoms.coord[(atoms.chain_id == "Y") & (atoms.res_id >= 5), 1] += 5
    # A rigid transform of the whole complex must not change the diagnosis.
    atoms.coord += [11, 3, -7]
    structure_io.save_structure(str(folded), atoms)
    candidate = {
        "source": "ppiflow",
        "generated_structure": runner.files.remote(generated),
        "refold_structure": runner.files.remote(folded),
        "chains": {"A": "GGGGGG", "B": "AAAAAA"},
        "metrics": {"af3score_iptm": 0.9, "design_refold_dockq": 0.3, "rosetta_interface_dg": -10},
    }
    evidence = analyze(runner, candidate)
    assert evidence["epitope_hit"] is True
    assert evidence["framework_rmsd"] < 0.01
    assert evidence["region_rmsd"]["B_region_1"] == pytest.approx(5, abs=0.01)
    assert evidence["diagnosis"] == "local_change"
    assert evidence["region_source"] == "registered_cdr_position"
    config.generators["ppiflow"].request.pop("cdr_position")
    automatic = analyze(runner, candidate)
    assert automatic["region_source"] == "python_displaced_loop"
    assert automatic["local_region"] == "B_region_1"
    candidate["chains"]["A"] = "WRONG"
    evidence = analyze(runner, candidate)
    assert evidence["epitope_hit"] is None
    assert evidence["diagnosis"] == "uncertain"


def test_partial_flow_uses_registered_inputs_and_python_region(round_config, tmp_path):
    from opendde_harness.plugin.protein_design.initial_round.autonomous import options_for

    config = autonomous_config(round_config)
    runner = InitialRoundRunner(config)
    runner.root.mkdir()
    runner.checkpoint = Checkpoint(runner.root, {})
    generated = pdb(tmp_path / "parent.pdb")
    candidate = {
        "candidate_id": "parent",
        "source": "ppiflow",
        "status": "rejected",
        "generated_structure": runner.files.remote(generated),
        "chains": {"A": "GGGGGG", "B": "AAAAAA"},
        "metrics": {},
        "diagnostics": {"diagnosis": "local_change", "local_region": "B_region_1"},
    }
    assert all(o["action"] != "partial_flow" for o in options_for(runner, [candidate], 1, 2, []))
    candidate["diagnostics"]["repair_regions"] = {"B_region_1": {"chain": "B", "positions": [5, 6]}}
    options = options_for(runner, [candidate], 1, 2, [])
    repair = options[0]
    assert repair["action"] == "partial_flow" and repair["samples"] == 1
    assert repair["request"]["complex_pdb"] == runner.files.remote(generated)
    assert repair["request"]["fixed_positions"] == "B1-4"
    assert repair["request"]["cdr_position"] == "B5-6"
    assert repair["request"]["start_t"] == 0.01
    assert all(o["action"] != "partial_flow" for o in options_for(runner, [candidate], 2, 2, ["parent"]))
    # Registered fixed positions win even if geometry suggests moving them.
    candidate["generation_request"] = {**config.generators["ppiflow"].request, "fixed_positions": "B5-6"}
    assert all(o["action"] != "partial_flow" for o in options_for(runner, [candidate], 1, 2, []))


def test_default_config_hash_shape_is_preserved(round_config):
    assert "autonomous" not in round_config.model_dump(mode="json")
    value = round_config.model_dump(mode="json")
    value["evaluation"]["rosetta"] = None
    value["autonomous"] = {}
    with pytest.raises(ValueError, match="require a scoring backend and Rosetta"):
        InitialRoundConfig.model_validate(value)


async def test_partial_flow_executes_once_then_rescores_child(round_config, fake_services, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.runner import digest

    transport, calls = fake_services
    config = autonomous_config(round_config, max_rounds=1, max_samples=3)
    config.generators.pop("torchcraft")
    base_generate = InitialRoundRunner.generate
    submitted = []

    async def recorded_generate(self, **kwargs):
        result = await base_generate(self, **kwargs)
        if kwargs.get("round_index"):
            submitted.append(kwargs["source_requests"]["ppiflow"])
        return result

    def diagnosis(runner, candidate):
        return {
            "diagnosis": "hit" if candidate.get("parent_id") else "local_change",
            "local_region": "B_region_1",
            "repair_regions": {"B_region_1": {"chain": "B", "positions": [5, 6]}},
            "epitope_hit": True,
            "flexibility": "unknown",
        }

    monkeypatch.setattr(InitialRoundRunner, "generate", recorded_generate)
    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.autonomous.analyze", diagnosis)
    runner = InitialRoundRunner(config, transport=transport)
    result = await runner.run()
    assert submitted[0]["mode"] == "partial_nanobody"
    assert submitted[0]["samples_per_target"] == 1
    assert result["accepted"] == 1
    assert result["used_samples"] == 2
    assert calls[("af3score", "POST")] == calls[("torchfold", "POST")] == 2
    parent, child = runner.checkpoint.data["candidates"]
    assert child["parent_id"] == parent["candidate_id"]
    assert digest(runner.files.local(parent["generated_structure"])) == parent["generated_sha256"]
    # A completed resume must still reject modified retained evidence.
    runner.files.local(child["refold_structure"]).write_text("changed")
    with pytest.raises(ValueError, match="refold artifact changed"):
        await InitialRoundRunner(config, transport=transport).run()


async def test_agent_decision_reuses_configured_provider_and_json_contract(monkeypatch):
    from types import SimpleNamespace

    from opendde_harness.plugin.protein_design.initial_round.autonomous import agent_choice

    seen = []
    closed = []

    class Provider:
        async def chat_with_retry(self, **kwargs):
            seen.append(kwargs)
            return SimpleNamespace(
                finish_reason="stop", tool_calls=[], content='{"option_id":0,"reason":"localized H3"}'
            )

    monkeypatch.setattr("opendde_harness.cli._helpers.make_provider", lambda config: Provider())
    monkeypatch.setattr("opendde_harness.config.loader.load_config", lambda: object())

    async def shutdown():
        closed.append(True)

    monkeypatch.setattr("opendde_harness.providers.pi_service.shutdown_service", shutdown)
    decision = await agent_choice({"options": [{"option_id": 0, "action": "partial_flow"}]})
    assert decision.option_id == 0 and decision.reason == "localized H3"
    assert seen[0]["max_tokens"] == 1500
    assert closed == [True]


async def test_feedback_repairs_python_loop_without_extra_input(round_config, fake_services, monkeypatch):
    import biotite.structure.io as structure_io

    from opendde_harness.plugin.protein_design.initial_round.artifacts import structure

    transport, calls = fake_services
    base_handler = transport.handler
    config = autonomous_config(round_config, max_rounds=1, max_samples=2)
    config.generators.pop("torchcraft")
    requests = []
    first_refold = True

    def handler(request):
        nonlocal first_refold
        if request.method == "POST" and request.url.host == "ppiflow.test":
            requests.append(json.loads(request.content))
        response = base_handler(request)
        if request.url.host == "torchfold.test" and request.url.path.endswith("/results") and first_refold:
            first_refold = False
            path = SharedFiles(config.path_mappings).local(response.json()["files"][0])
            atoms = structure(path)
            atoms.coord[(atoms.chain_id == "Y") & (atoms.res_id >= 5), 1] += 5
            structure_io.save_structure(str(path), atoms)
        return response

    dockq_calls = 0

    async def dockq(*args, **kwargs):
        nonlocal dockq_calls
        if args[-1] == "--check":
            payload = {"dockq_available": True}
        else:
            dockq_calls += 1
            payload = {"dockq": 0.3 if dockq_calls == 1 else 1.0, "interfaces": []}

        class Process:
            returncode = 0

            async def communicate(self):
                return json.dumps(payload).encode(), b""

        return Process()

    monkeypatch.setattr(
        "opendde_harness.plugin.protein_design.initial_round.runner.asyncio.create_subprocess_exec", dockq
    )
    runner = InitialRoundRunner(config, transport=httpx.MockTransport(handler))
    result = await runner.run()
    assert len(requests) == 2
    assert requests[1]["mode"] == "partial_nanobody"
    assert requests[1]["cdr_position"] == "B5-6"
    assert requests[1]["fixed_positions"] == "B1-4"
    parent, child = runner.checkpoint.data["candidates"]
    assert parent["diagnostics"]["region_source"] == "python_displaced_loop"
    assert "cdr_position" not in parent["generation_request"]
    assert "design_regions" not in parent and "partial_requests" not in parent
    assert child["status"] == "accepted" and result["used_samples"] == 2
    assert calls[("af3score", "POST")] == calls[("torchfold", "POST")] == 2


def test_registered_ranges_preserve_fixed_positions_and_actual_pdb_ids(round_config, tmp_path):
    from opendde_harness.plugin.protein_design.initial_round.artifacts import structure
    from opendde_harness.plugin.protein_design.initial_round.autonomous import _partial_request
    from opendde_harness.plugin.protein_design.initial_round.regions import derive_regions, encode_positions

    config = autonomous_config(round_config)
    runner = InitialRoundRunner(config)
    path = pdb(tmp_path / "numbered.pdb")
    atoms = structure(path)
    atoms.res_id[atoms.chain_id == "B"] += 100
    import biotite.structure.io as structure_io

    structure_io.save_structure(str(path), atoms)
    request = {
        **config.generators["ppiflow"].request,
        "cdr_position": "B104-106",
        "fixed_positions": "B105",
        "start_t": 0.03,
        "num_timesteps": 100,
        "config": "/shared/config.yaml",
        "model_weights": "/shared/weights.ckpt",
    }
    regions, origin = derive_regions(atoms, config.generators["ppiflow"].chains, request, {"B": [0, 0, 0, 4, 4, 4]}, 3)
    assert origin == "registered_cdr_position"
    assert [r.positions for r in regions.values()] == [[4], [6]]
    assert encode_positions(atoms, {"B": {1, 2, 3, 5, 6}}) == "B101-103,B105-106"
    candidate = {
        "source": "ppiflow",
        "generation_request": request,
        "generated_structure": runner.files.remote(path),
        "chains": {"A": "GGGGGG", "B": "AAAAAA"},
        "diagnostics": {"repair_regions": {"B_region_1": regions["B_region_1"].model_dump()}},
    }
    repair = _partial_request(runner, candidate, "B_region_1", "repair")
    assert repair["cdr_position"] == "B104"
    assert repair["fixed_positions"] == "B101-103,B105-106"
    assert repair["start_t"] == 0.03 and repair["num_timesteps"] == 100
    assert repair["config"] == request["config"] and repair["model_weights"] == request["model_weights"]
    with pytest.raises(ValueError, match="missing or ambiguous"):
        derive_regions(atoms, config.generators["ppiflow"].chains, {"cdr_position": "B200-201"}, {"B": [4] * 6}, 3)


@pytest.fixture
def owned_api_config(round_config, tmp_path, monkeypatch):
    names, cancelled = {}, []
    services = [g.service for g in round_config.generators.values()]
    services += [round_config.evaluation.af3score, round_config.evaluation.torchfold, round_config.evaluation.rosetta]
    for index, service in enumerate(services):
        script = tmp_path / f"api-{index}.sbatch"
        script.write_text("#!/bin/bash\nsleep infinity\n")
        service.sbatch_script = script
        service.sbatch_state_file = tmp_path / f"api-{index}.json"
        service.sbatch_resolve_host = True

    async def command(config, *args):
        if args[0] == "sbatch":
            job_id = str(100 + len(names))
            names[job_id] = args[2].split("=", 1)[1]
            return job_id
        if args[0] == "scontrol":
            return httpx.URL(config.url).host
        if args[0] == "scancel":
            cancelled.append(args[1])
            return ""
        if args[0] == "squeue":
            job_id = args[2]
            return "RUNNING|" + (names[job_id] if args[-1] == "--format=%T|%j" else "gpu01")
        return ""

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.api_startup.slurm_command", command)
    return names, cancelled


async def test_completed_round_releases_owned_apis_after_caching_results(round_config, fake_services, owned_api_config):
    names, cancelled = owned_api_config
    config = autonomous_config(round_config, max_samples=2)
    first = await InitialRoundRunner(config, transport=fake_services[0]).run()
    assert set(cancelled) == set(names) and len(cancelled) == 5
    assert all(rows[0]["state"] == "cancel_requested" for rows in first["api_cleanup"].values())
    root = config.work_dir / config.name
    assert json.loads((root / "summary.json").read_text())["api_cleanup"] == first["api_cleanup"]
    second = await InitialRoundRunner(config, transport=fake_services[0]).run()
    assert first == second and len(cancelled) == 5


async def test_round_keeps_apis_when_inference_outcome_is_unresolved(
    round_config, fake_services, owned_api_config, monkeypatch
):
    _, cancelled = owned_api_config
    base_score = InitialRoundRunner.score

    async def interrupted(self, candidate):
        await base_score(self, candidate)
        self.checkpoint.data["jobs"][f"score/{candidate['candidate_id']}"]["state"] = "waiting"
        raise TimeoutError("remote work still pending")

    monkeypatch.setattr(InitialRoundRunner, "score", interrupted)
    config = autonomous_config(round_config, max_samples=2)
    result = await InitialRoundRunner(config, transport=fake_services[0]).run()
    assert not cancelled
    assert result["accepted"] == 0
    assert all(rows[0]["state"] == "retained_active_jobs" for rows in result["api_cleanup"].values())


async def test_low_confidence_candidate_is_optimized_and_fully_reevaluated(round_config, fake_services, tmp_path):
    transport, calls = fake_services
    handler_base = transport.handler
    config = autonomous_config(round_config, max_samples=2, max_rounds=1)
    config.generators.pop("torchcraft")
    config.evaluation.torchopt = Service(url="https://torchopt.test", poll_interval=0.001)
    files = SharedFiles(config.path_mappings)
    final = pdb(tmp_path / "native/cases/case/final/optimized.pdb")
    table = tmp_path / "native/results.csv"
    table.write_text("candidate\noptimized\n")
    identifier = "a" * 32
    submitted = {}
    optimization_calls = []

    def handler(request):
        if request.url.host == "torchopt.test":
            optimization_calls.append(request)
            if request.url.path == "/health":
                return httpx.Response(200, json={"backend_available": True})
            if request.method == "POST":
                submitted.update(json.loads(request.content))
                assert request.url.path == "/v1/optimizations"
                assert submitted["binder_type"] == "vhh"
                assert submitted["binder_chain"] == "B" and submitted["target_chain"] == "A"
                return httpx.Response(201, json={"optimization_id": identifier})
            if request.url.path.endswith("/results"):
                return httpx.Response(
                    200,
                    json={
                        "optimization_id": identifier,
                        "structures": [files.remote(final)],
                        "results": [files.remote(table)],
                    },
                )
            return httpx.Response(
                200,
                json={
                    "optimization_id": identifier,
                    "request": submitted,
                    "state": "SUCCEEDED",
                    "returncode": 0,
                    "runs": {"run": {"native": True}},
                    "submissions": {"run/submission.json": {"slurm_job_id": "123"}},
                    "scheduler_jobs": {"123": {"state": "COMPLETED", "exit_code": "0:0"}},
                },
            )
        response = handler_base(request)
        if request.url.host == "af3score.test" and request.url.path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": [{"iptm": 0.7 if calls[("af3score", "POST")] == 1 else 0.9}]})
        return response

    runner = InitialRoundRunner(config, transport=httpx.MockTransport(handler))
    result = await runner.run()
    decision = runner.checkpoint.data["autonomous"]["decisions"][0]
    assert decision["action"] == "torchopt" and result["used_samples"] == 2
    assert result["accepted"] == 1 and result["unavailable_actions"] == {}
    assert submitted["mcmc_rounds"] == submitted["mcmc_proposals"] == submitted["ddg_rounds"] == 1
    parent, child = runner.checkpoint.data["candidates"]
    assert parent["diagnostics"]["diagnosis"] == "low_confidence"
    assert child["optimized_by"] == "torchopt" and child["parent_id"] == parent["candidate_id"]
    assert child["metrics"]["af3score_iptm"] == 0.9 and child["metrics"]["ranking_score"] == 102
    assert calls[("af3score", "POST")] == calls[("torchfold", "POST")] == calls[("rosetta", "POST")] == 2
    assert sum(request.method == "POST" for request in optimization_calls) == 1
    before = dict(calls)
    assert await InitialRoundRunner(config, transport=httpx.MockTransport(handler)).run() == result
    assert {k: v for k, v in calls.items() if k[1] == "POST"} == {k: v for k, v in before.items() if k[1] == "POST"}
    assert sum(request.method == "POST" for request in optimization_calls) == 1


def test_paired_antibodies_do_not_route_into_unsupported_optimization(round_config, tmp_path):
    from opendde_harness.plugin.protein_design.initial_round.autonomous import options_for

    config = autonomous_config(round_config)
    config.generators.pop("torchcraft")
    config.evaluation.torchopt = Service(url="https://torchopt.test")
    config.generators["ppiflow"].chains.binder.append("L")
    runner = InitialRoundRunner(config)
    runner.root.mkdir()
    runner.checkpoint = Checkpoint(runner.root, {})
    candidate = {
        "candidate_id": "paired",
        "source": "ppiflow",
        "status": "rejected",
        "metrics": {},
        "diagnostics": {"diagnosis": "low_confidence"},
    }
    assert all(option["action"] != "torchopt" for option in options_for(runner, [candidate], 1, 2, []))


def test_registered_optimization_service_is_reused_without_new_scientific_inputs(round_config):
    from opendde_harness.plugin.protein_design.initial_round.automation import prepare_run

    config = autonomous_config(round_config)
    registrations = {"existing_opt": {"kind": "torchopt", "service": {"url": "https://torchopt.test"}}}
    result = prepare_run(config.model_dump(mode="json"), registrations=registrations)
    assert result.evaluation.torchopt.url == "https://torchopt.test"
    assert result.generators == config.generators
    registrations["other_opt"] = {"kind": "torchopt", "service": {"url": "https://other.test"}}
    with pytest.raises(ValueError, match="multiple registered torchopt"):
        prepare_run(config.model_dump(mode="json"), registrations=registrations)


async def test_torchscore_feedback_uses_registered_scorer_for_ranking_and_resume(round_config, fake_services):
    transport, calls = fake_services
    base = transport.handler
    score_requests = []
    config = autonomous_config(round_config, max_rounds=1, max_samples=3)
    config.evaluation.scoring_backend = "torchscore"
    config.evaluation.torchscore = Service(url="https://torchscore.test")
    config.evaluation.af3score = None

    def handler(request):
        assert request.url.host != "af3score.test"
        if request.url.host == "torchscore.test" and request.method == "POST":
            score_requests.append(json.loads(request.content))
        return base(request)

    runner = InitialRoundRunner(config, transport=httpx.MockTransport(handler))
    result = await runner.run()
    assert result["scoring_backend"] == "torchscore"
    assert result["ranking_policy"] == "100*torchscore_iptm-rosetta_interface_dg_desc"
    assert result["accepted"] == 3
    assert calls[("af3score", "POST")] == 0
    assert calls[("torchscore", "POST")] == 3
    for request in score_requests:
        assert request["run_data_pipeline"] is False and request["write_fold_input_json_file"] is False
        assert request["num_samples"] == 1 and request["seed"] == 10
    for candidate in runner.checkpoint.data["candidates"]:
        assert candidate["scoring_backend"] == "torchscore"
        assert candidate["metrics"]["torchscore_iptm"] == 0.9
        assert candidate["metrics"]["ranking_score"] == 102
        assert "af3score_iptm" not in candidate["metrics"]
    csv_header = (runner.root / "ranking.csv").read_text().splitlines()[0]
    assert "torchscore_iptm" in csv_header and "af3score_iptm" not in csv_header
    count = calls[("torchscore", "POST")]
    assert await InitialRoundRunner(config, transport=httpx.MockTransport(handler)).run() == result
    assert calls[("torchscore", "POST")] == count


async def test_selected_torchscore_failure_does_not_fall_back_to_af3score(round_config, fake_services):
    config = autonomous_config(round_config)
    config.evaluation.scoring_backend = "torchscore"
    config.evaluation.torchscore = Service(url="https://torchscore.test")
    base = fake_services[0].handler
    seen = []

    def handler(request):
        seen.append(request)
        assert request.url.host != "af3score.test"
        if request.url.host == "torchscore.test" and request.method == "POST":
            return httpx.Response(500, json={"error": "unavailable"})
        return base(request)

    with pytest.raises(SubmissionUncertainError):
        await InitialRoundRunner(config, transport=httpx.MockTransport(handler)).run()
    assert any(request.url.host == "torchscore.test" for request in seen)


def test_feedback_diagnosis_and_ranking_never_mix_scorers():
    from opendde_harness.plugin.protein_design.initial_round.autonomous import rank_score
    from opendde_harness.plugin.protein_design.initial_round.config import AutonomousRound
    from opendde_harness.plugin.protein_design.initial_round.diagnostics import classify

    metrics = {"af3score_iptm": 0.99, "torchscore_iptm": 0.7, "design_refold_dockq": 0.9, "rosetta_interface_dg": -12}
    assert classify(metrics, {"epitope_hit": True}, AutonomousRound(), "torchscore_iptm") == "low_confidence"
    assert rank_score({"metrics": metrics}, "torchscore_iptm") == 82
    assert rank_score({"metrics": metrics}, "af3score_iptm") == 111


def test_scorer_choice_uses_explicit_configuration_then_registered_kind(round_config):
    import copy

    from opendde_harness.plugin.protein_design.initial_round.automation import prepare_run

    value = autonomous_config(round_config).model_dump(mode="json")
    evaluation = value["evaluation"]
    evaluation.pop("scoring_backend", None)
    evaluation["af3score"] = None
    evaluation.pop("torchscore", None)
    registrations = {"score_alias": {"kind": "torchscore", "service": {"url": "http://torchscore.test:8070"}}}
    before = copy.deepcopy(value)
    selected = prepare_run(value, registrations=registrations)
    assert selected.evaluation.scoring_backend == "torchscore"
    assert selected.evaluation.torchscore.url == "http://torchscore.test:8070"
    assert value == before
    registrations["af3_alias"] = {"kind": "af3score", "service": {"url": "https://af3.test"}}
    assert prepare_run(value, registrations=registrations).evaluation.scoring_backend == "af3score"
    evaluation["scoring_backend"] = "torchscore"
    assert prepare_run(value, registrations=registrations).evaluation.scoring_backend == "torchscore"
    evaluation["scoring_backend"] = "af3score"
    registrations.pop("af3_alias")
    with pytest.raises(ValueError, match="af3score service is required"):
        prepare_run(value, registrations=registrations)


def test_machine_predicates_do_not_confuse_missing_evidence_or_loop_content(round_config, tmp_path):
    from opendde_harness.plugin.protein_design.initial_round.artifacts import structure
    from opendde_harness.plugin.protein_design.initial_round.policy import (
        flexibility,
        generation_order,
        hotspot_hit,
        local_region,
        repair_allowed,
    )
    from opendde_harness.plugin.protein_design.initial_round.regions import check_region_encoding

    assert hotspot_hit([]) is None
    assert hotspot_hit([{"distance": 5.0}]) is False
    assert hotspot_hit([{"distance": 4.99}]) is True
    assert flexibility(0.8, 0.2) == "unknown"  # Loop-rich alone does not establish flexibility.
    assert flexibility(0.8, 3) == "flexible"
    assert flexibility(0.2, 1) == "rigid"
    assert generation_order(["ppiflow", "torchcraft"], "flexible", ["torchcraft"]) == ["torchcraft", "ppiflow"]
    assert generation_order(["ppiflow"], "flexible", []) == ["ppiflow"]
    assert local_region(1.0, {"H3": 4}, 2, 4) is None  # Too few stable residues.
    assert local_region(1.0, {"H1": 4, "H3": 4}, 4, 6) is None
    assert local_region(1.0, {"H3": 4}, 4, 6) == "H3"
    assert not repair_allowed({"candidate_id": "child", "parent_id": "parent"}, [])
    atoms = structure(pdb(tmp_path / "encoding.pdb"))
    atoms.ins_code[(atoms.chain_id == "B") & (atoms.res_id == 5)] = "A"
    encoding = check_region_encoding(atoms, {"B": {5}})
    assert encoding["ok"] is False and encoding["encoded"] is None and encoding["reason"]
