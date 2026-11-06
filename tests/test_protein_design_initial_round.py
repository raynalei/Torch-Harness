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

pytestmark = pytest.mark.in_memory_transport


def pdb(path, *, binder="ALA", target="A", chain="B"):
    lines = []
    serial = 1
    for chain_id, residue, y in [(target, "GLY", 0), (chain, binder, 4)]:
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
                pdb(out / "model.pdb", target="X", chain="Y")
                (out / "summary_confidences.json").write_text(json.dumps({"iptm": 0.95}))
                outputs[job] = {
                    "files": [files.remote(out / "model.pdb"), files.remote(out / "summary_confidences.json")]
                }
                return httpx.Response(202, json={"fold_id": job})
            if source == "rosetta":
                assert body["chain_group_a"] == "Y" and body["chain_group_b"] == "X"
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
    with pytest.raises(RuntimeError, match="both generators are required"):
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
    config = InitialRoundConfig.load(Path(__file__).parents[1] / "docs/examples/initial_round.yaml")
    assert set(config.generators) == {"ppiflow", "torchcraft"}
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
