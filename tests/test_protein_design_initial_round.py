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
        from biotite.sequence import ProteinSequence

        try:
            ProteinSequence.convert_letter_3to1(residue)
            residues = [residue] * 6
        except KeyError:
            residues = [ProteinSequence.convert_letter_1to3(letter) for letter in residue]
        for index, residue in enumerate(residues, 1):
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
                        "samples_per_target": 1,
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


@pytest.mark.parametrize("dialect", ["alphafold3", "TorchFold"])
def test_fold_input_updates_binder_without_reusing_stale_msa(round_config, tmp_path, dialect):
    generator = round_config.generators["ppiflow"]
    template = json.loads(generator.fold_template.read_text())
    template["dialect"] = dialect
    generator.fold_template.write_text(json.dumps(template))
    original = generator.fold_template.read_bytes()
    output = tmp_path / "fold.json"
    prepare_fold(generator.fold_template, {"A": "GGGGGG", "B": "VVVVVV"}, generator.chains, "new", output)
    value = json.loads(output.read_text())
    assert value["dialect"] == "TorchFold"
    assert generator.fold_template.read_bytes() == original
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
                assert prepared["dialect"] == "TorchFold"
                assert prepared["sequences"][1]["protein"]["sequence"] == "AAAAAA"
                ids = [entry["protein"]["id"] for entry in prepared["sequences"]]
                out = tmp_path / job
                pdb(out / "model.pdb", target=ids[0], chain=ids[1], light=ids[2] if len(ids) == 3 else None)
                (out / "summary_confidences.json").write_text(json.dumps({"iptm": 0.95}))
                outputs[job] = {
                    "files": [files.remote(out / "model.pdb"), files.remote(out / "summary_confidences.json")]
                }
                return httpx.Response(202, json={"fold_id": job})
            if source == "rosetta":
                assert body["chain_group_a"] == (
                    "".join(round_config.generators["ppiflow"].chains.binder)
                    if "generated-interface.pdb" in body["input_pdb"]
                    else "YZ"
                    if round_config.generators["ppiflow"].request["mode"] == "antibody"
                    else "Y"
                )
                assert body["chain_group_b"] == ("A" if "generated-interface.pdb" in body["input_pdb"] else "X")
                log = tmp_path / job / "residue.out"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("ResResE Res1 Res2 total\nResResE ---- ---- -----\nResResE ALA_661 GLY_651 -6.0\n")
                outputs[job] = {"metrics": {"dG_separated": -12.0}, "files": [files.remote(log)]}
            else:
                assert body["mode"] == "pdb" and body["num_samples"] == 1
                outputs[job] = {"files": []}
            return httpx.Response(202, json={"task_id": job})
        job = path.split("/")[3]
        if path.endswith("/results"):
            return httpx.Response(200, json=outputs[job])
        if path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": [{"iptm": 0.9, "ptm": 0.95}]})
        return httpx.Response(200, json={"status": "COMPLETED", "exit_code": "0:0"})

    return httpx.MockTransport(handler), calls


@pytest.mark.parametrize("use_identity_refold_map", [False, True])
async def test_both_generators_complete_round_and_resume_without_resubmitting(
    round_config, fake_services, use_identity_refold_map
):
    transport, calls = fake_services
    config = round_config
    if use_identity_refold_map:
        from opendde_harness.plugin.protein_design.initial_round.artifacts import chain_sequences

        value = round_config.model_dump(mode="json")
        for source, generator in value["generators"].items():
            generator["chains"] = {"target": ["D"], "binder": ["H"]}
            generator["fold_template"] = None
            pdb(round_config.work_dir / source / "generated.pdb", target="D", chain="H")
        request = value["generators"]["ppiflow"]["request"]
        request.pop("antigen_chain")
        request.pop("heavy_chain")
        config = InitialRoundConfig.model_validate(value)
        handle_request = transport.handler
        files = SharedFiles(config.path_mappings)

        def verify_user_chains(request):
            source = request.url.host.split(".")[0]
            if request.method != "POST":
                return handle_request(request)
            body = json.loads(request.content)
            if source == "ppiflow":
                assert body["antigen_chain"] == "D" and body["heavy_chain"] == "H"
            elif source == "af3score":
                (candidate,) = files.local(body["input_pdb_dir"]).glob("*.pdb")
                assert set(chain_sequences(candidate)) == {"D", "H"}
            elif source == "torchfold":
                fold_input = json.loads(files.local(body["json_path"]).read_text())
                assert [entry["protein"]["id"] for entry in fold_input["sequences"]] == ["D", "H"]
            elif source == "rosetta":
                assert (body["chain_group_a"], body["chain_group_b"]) == ("H", "D")
                # The existing fixture checks explicit refold names X/Y for legacy configurations.
                calls[(source, request.method)] += 1
                return httpx.Response(202, json={"task_id": "identity-rosetta"})
            return handle_request(request)

        def respond(request):
            if request.url.path.endswith("/identity-rosetta/results"):
                return httpx.Response(200, json={"metrics": {"dG_separated": -12.0}})
            return verify_user_chains(request)

        transport = httpx.MockTransport(respond)
    result = await InitialRoundRunner(config, transport=transport).run()
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
    assert await InitialRoundRunner(config, transport=transport).run() == result
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
        assert content["dialect"] == "TorchFold"
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
    value["evaluation"]["scoring_backend"] = "torchscore"
    value["evaluation"]["torchscore"] = {**value["evaluation"]["af3score"], "url": "https://torchscore.test"}
    value["autonomous"] = {"decision_policy": "rules", "max_rounds": 3, "max_samples": 8, "batch_size": 1, **overrides}
    return InitialRoundConfig.model_validate(value)


async def test_autonomous_round_preserves_best_budget_and_resume(round_config, fake_services):
    transport, calls = fake_services
    config = autonomous_config(round_config)
    result = await InitialRoundRunner(config, transport=transport).run()
    assert result["ranking_policy"] == "torchscore_iptm_desc,ptm_desc,dockq_desc"
    assert result["stop_reason"] == "rules_stop"
    assert result["used_samples"] == result["accepted"] == 2
    root = config.work_dir / config.name
    records = json.loads((root / "candidates.json").read_text())
    assert all(c["metrics"]["ranking_score"] == 90 for c in records)
    assert all(c["diagnostics"]["epitope_hit"] is True for c in records)
    assert all("generated.pdb" in c["generated_structure"] for c in records)
    assert all("rosetta_interface_dg" not in c["metrics"] for c in records)
    assert calls[("rosetta", "POST")] == 0
    decisions = json.loads((root / "decisions.json").read_text())["decisions"]
    assert [d["action"] for d in decisions] == ["stop"]
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


async def test_autonomous_low_iptm_terminates_without_regeneration_or_refold(round_config, fake_services):
    config = autonomous_config(round_config, min_score=0.95, max_samples=2)
    runner = InitialRoundRunner(config, transport=fake_services[0])
    result = await runner.run()
    assert result["accepted"] == 0
    records = runner.checkpoint.data["candidates"]
    assert all(c["diagnostics"]["diagnosis"] == "low_iptm" and c["status"] == "terminated" for c in records)
    assert all("design_refold_dockq" not in c["metrics"] for c in records)
    assert fake_services[1][("torchfold", "POST")] == fake_services[1][("rosetta", "POST")] == 0
    assert fake_services[1][("ppiflow", "POST")] == fake_services[1][("torchcraft", "POST")] == 1


async def test_invalid_agent_choice_falls_back_before_submission(round_config, fake_services, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round.autonomous import Choice

    async def bad_choice(context):
        return Choice(option_id=999, reason="invalid selection")

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.autonomous.agent_choice", bad_choice)
    config = autonomous_config(round_config, decision_policy="agent", max_rounds=1, max_samples=3)
    config.autonomous.min_refold_score = 0.95
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
    config.autonomous.min_refold_score = 0.95
    transport, calls = fake_services
    with pytest.raises(RuntimeError, match="interrupted"):
        await InitialRoundRunner(config, transport=transport).run()
    before = calls[("ppiflow", "POST")]
    result = await InitialRoundRunner(config, transport=transport).run()
    assert calls[("ppiflow", "POST")] == before == 2
    assert result["used_samples"] == 3
    assert result["followup_rounds"] == 1


def test_default_config_hash_shape_is_preserved(round_config):
    assert "autonomous" not in round_config.model_dump(mode="json")
    value = round_config.model_dump(mode="json")
    value["evaluation"]["rosetta"] = None
    value["autonomous"] = {}
    with pytest.raises(ValueError, match="require.*Rosetta"):
        InitialRoundConfig.model_validate(value)


@pytest.mark.parametrize("fail_partial", [False, True])
async def test_partial_flow_executes_once_then_rescores_child(round_config, fake_services, fail_partial):
    from opendde_harness.plugin.protein_design.initial_round.runner import digest

    base, calls = fake_services
    config = autonomous_config(round_config, max_rounds=1, max_samples=3)
    config.generators.pop("torchcraft")
    submitted, failed_ids = [], set()

    def handler(request):
        response = base.handler(request)
        if request.url.host == "torchscore.test" and request.url.path.endswith("/metrics"):
            return httpx.Response(
                200, json={"metrics": [{"iptm": 0.7 if calls[("torchscore", "POST")] == 1 else 0.9, "ptm": 0.9}]}
            )
        if request.method == "POST" and request.url.host == "ppiflow.test":
            payload = json.loads(request.content)
            if payload["mode"].startswith("partial_"):
                submitted.append(payload)
                if fail_partial and not failed_ids:
                    failed_ids.add(response.json()["design_id"])
        if request.method == "GET" and request.url.path.rsplit("/", 1)[-1] in failed_ids:
            return httpx.Response(200, json={"status": "FAILED", "exit_code": "1:0"})
        return response

    transport = httpx.MockTransport(handler)
    runner = InitialRoundRunner(config, transport=transport)
    if fail_partial:
        with pytest.raises(RuntimeError, match="generation failures"):
            await runner.run()
        runner = InitialRoundRunner(config, transport=transport, retry_failed=True)
    result = await runner.run()
    assert calls[("ppiflow", "POST")] == (3 if fail_partial else 2)
    assert submitted[0]["fixed_positions"] == "B1" and submitted[0]["cdr_position"] == "B2-6"
    assert result["accepted"] == 1 and result["used_samples"] == 2
    assert calls[("torchscore", "POST")] == 2
    assert calls[("torchfold", "POST")] == calls[("rosetta", "POST")] == 1
    parent, child = runner.checkpoint.data["candidates"]
    assert child["parent_id"] == parent["candidate_id"]
    assert digest(runner.files.local(parent["generated_structure"])) == parent["generated_sha256"]
    assert child["generation_request"]["complex_pdb"] == parent["generated_structure"]


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
    assert set(cancelled) == set(names) and len(cancelled) == 4
    assert all(rows[0]["state"] == "cancel_requested" for rows in first["api_cleanup"].values())
    root = config.work_dir / config.name
    assert json.loads((root / "summary.json").read_text())["api_cleanup"] == first["api_cleanup"]
    second = await InitialRoundRunner(config, transport=fake_services[0]).run()
    assert first == second and len(cancelled) == 4


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
    runner = InitialRoundRunner(config, transport=fake_services[0])
    with pytest.raises(RuntimeError, match="resume this run_dir"):
        await runner.run()
    assert not cancelled
    assert all(rows[0]["state"] == "retained_active_jobs" for rows in runner.checkpoint.data["api_cleanup"].values())


@pytest.mark.parametrize("fail_optimization", [False, True])
@pytest.mark.parametrize("final_ptm", [0.95, 0.8])
async def test_low_confidence_candidate_is_optimized_and_fully_reevaluated(
    round_config, fake_services, tmp_path, fail_optimization, final_ptm
):
    transport, calls = fake_services
    handler_base = transport.handler
    config = autonomous_config(round_config, max_samples=3, max_rounds=2)
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
        nonlocal identifier
        if request.url.host == "torchopt.test":
            optimization_calls.append(request)
            if request.url.path == "/health":
                return httpx.Response(200, json={"backend_available": True})
            if request.method == "POST":
                identifier = ("a" if sum(r.method == "POST" for r in optimization_calls) == 1 else "b") * 32
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
                    "state": "FAILED" if fail_optimization and identifier == "a" * 32 else "SUCCEEDED",
                    "returncode": 0,
                    "runs": {"run": {"native": True}},
                    "submissions": {"run/submission.json": {"slurm_job_id": "123"}},
                    "scheduler_jobs": {
                        "123": {"state": "FAILED", "exit_code": "1:0"}
                        if fail_optimization and identifier == "a" * 32
                        else {"state": "COMPLETED", "exit_code": "0:0"}
                    },
                },
            )
        response = handler_base(request)
        if request.url.host == "torchscore.test" and request.url.path.endswith("/metrics"):
            return httpx.Response(
                200, json={"metrics": [{"iptm": 0.9, "ptm": 0.7 if calls[("torchscore", "POST")] < 3 else final_ptm}]}
            )
        return response

    runner = InitialRoundRunner(config, transport=httpx.MockTransport(handler))
    if fail_optimization:
        with pytest.raises(RemoteJobFailedError, match="all child jobs terminal"):
            await runner.run()
        runner = InitialRoundRunner(config, transport=httpx.MockTransport(handler), retry_failed=True)
    result = await runner.run()
    decision = runner.checkpoint.data["autonomous"]["decisions"][1]
    assert decision["action"] == "torchopt" and result["used_samples"] == 3
    assert result["accepted"] == int(final_ptm > 0.8) and result["unavailable_actions"] == {}
    assert submitted["mcmc_rounds"] == submitted["mcmc_proposals"] == submitted["ddg_rounds"] == 1
    initial, parent, child = runner.checkpoint.data["candidates"]
    assert parent["diagnostics"]["diagnosis"] == "needs_optimization"
    assert initial["rosetta_structure"] == "generated"
    assert "rosetta_structure" not in child
    assert child["optimized_by"] == "torchopt" and child["parent_id"] == parent["candidate_id"]
    assert child["metrics"]["torchscore_iptm"] == 0.9 and child["metrics"]["ranking_score"] == 90
    assert calls[("torchscore", "POST")] == 3
    assert calls[("torchfold", "POST")] == int(final_ptm > 0.8)
    assert calls[("rosetta", "POST")] == 1
    assert sum(request.method == "POST" for request in optimization_calls) == (2 if fail_optimization else 1)
    before = dict(calls)
    assert await InitialRoundRunner(config, transport=httpx.MockTransport(handler)).run() == result
    assert {k: v for k, v in calls.items() if k[1] == "POST"} == {k: v for k, v in before.items() if k[1] == "POST"}
    assert sum(request.method == "POST" for request in optimization_calls) == (2 if fail_optimization else 1)


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
        "metrics": {"torchscore_iptm": 0.9, "torchscore_ptm": 0.7},
        "generation_request": {"mode": "partial_antibody"},
        "diagnostics": {"diagnosis": "low_ptm", "epitope_hit": True},
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
    assert result["ranking_policy"] == "torchscore_iptm_desc,ptm_desc,dockq_desc"
    assert result["accepted"] == 2
    assert calls[("af3score", "POST")] == 0
    assert calls[("torchscore", "POST")] == 2
    for request in score_requests:
        assert request["run_data_pipeline"] is False and request["write_fold_input_json_file"] is False
        assert request["num_samples"] == 1 and request["seed"] == 10
    for candidate in runner.checkpoint.data["candidates"]:
        assert candidate["scoring_backend"] == "torchscore"
        assert candidate["metrics"]["torchscore_iptm"] == 0.9
        assert candidate["metrics"]["ranking_score"] == 90
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

    metrics = {
        "af3score_iptm": 0.99,
        "torchscore_iptm": 0.4,
        "torchscore_ptm": 0.95,
        "design_refold_dockq": 0.9,
        "rosetta_interface_dg": -12,
    }
    assert classify(metrics, {"epitope_hit": True}, AutonomousRound(), "torchscore_iptm") == "low_iptm"
    assert rank_score({"metrics": metrics}, "torchscore_iptm") == 40
    assert rank_score({"metrics": metrics}, "af3score_iptm") == 99


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
    assert prepare_run(value, registrations=registrations).evaluation.scoring_backend == "torchscore"
    evaluation["scoring_backend"] = "torchscore"
    assert prepare_run(value, registrations=registrations).evaluation.scoring_backend == "torchscore"
    evaluation["scoring_backend"] = "af3score"
    registrations.pop("af3_alias")
    with pytest.raises(ValueError, match="af3score service is required"):
        prepare_run(value, registrations=registrations)


@pytest.mark.parametrize("failed_source", ["ppiflow", "torchcraft", "af3score", "torchscore", "torchfold", "rosetta"])
async def test_resume_retries_only_failed_model_and_reuses_upstream_results(round_config, fake_services, failed_source):
    config = round_config
    if failed_source == "torchscore":
        config.evaluation.scoring_backend = "torchscore"
        config.evaluation.torchscore = Service(url="https://torchscore.test", poll_interval=0.001)
        config.evaluation.af3score = None
    base, calls = fake_services
    failed_ids = set()

    def handler(request):
        response = base.handler(request)
        source = request.url.host.split(".")[0]
        if source == failed_source and request.method == "POST" and not failed_ids:
            submitted = response.json()
            failed_ids.add(next(submitted[key] for key in ("task_id", "design_id", "fold_id") if key in submitted))
        if request.method == "GET" and request.url.path.rsplit("/", 1)[-1] in failed_ids:
            return httpx.Response(200, json={"status": "FAILED", "exit_code": "1:0"})
        return response

    transport = httpx.MockTransport(handler)
    first = InitialRoundRunner(config, transport=transport)
    if failed_source in {"ppiflow", "torchcraft"}:
        with pytest.raises(RuntimeError, match="generation failures"):
            await first.run()
    else:
        assert (await first.run())["status"] == "partial"
    before = dict(calls)
    generated = {c["candidate_id"]: c["generated_sha256"] for c in first.checkpoint.data.get("candidates", [])}
    resumed = InitialRoundRunner(config, transport=transport, retry_failed=True)
    result = await resumed.run()
    assert result["status"] == "completed"
    assert result["accepted"] == 2
    for source in config.generators:
        assert calls[(source, "POST")] == before[(source, "POST")] + (source == failed_source)
    if failed_source not in config.generators:
        assert calls[(failed_source, "POST")] == before[(failed_source, "POST")] + 1
    assert all(
        c["generated_sha256"] == generated.get(c["candidate_id"], c["generated_sha256"])
        for c in resumed.checkpoint.data["candidates"]
    )
    attempts = resumed.checkpoint.data["job_attempts"]
    assert len(attempts) == 1
    key, history = next(iter(attempts.items()))
    assert history[0]["id"] in failed_ids
    assert history[0]["submission_key"] != resumed.checkpoint.data["jobs"][key]["submission_key"]


async def test_autonomous_recovery_pauses_on_scoring_failure_without_consuming_extra_budget(
    round_config, fake_services
):
    config = autonomous_config(round_config, max_samples=2)
    base, calls = fake_services
    failed = set()

    def handler(request):
        response = base.handler(request)
        if request.url.host == "torchscore.test" and request.method == "POST" and not failed:
            failed.add(response.json()["task_id"])
        if request.method == "GET" and request.url.path.rsplit("/", 1)[-1] in failed:
            return httpx.Response(200, json={"status": "FAILED", "exit_code": "1:0"})
        return response

    transport = httpx.MockTransport(handler)
    runner = InitialRoundRunner(config, transport=transport)
    with pytest.raises(RuntimeError, match="resume this run_dir"):
        await runner.run()
    resumed = InitialRoundRunner(config, transport=transport, retry_failed=True)
    result = await resumed.run()
    assert result["accepted"] == 2
    assert result["used_samples"] == 2
    assert calls[("ppiflow", "POST")] == calls[("torchcraft", "POST")] == 1
    assert calls[("torchscore", "POST")] == 3


def test_recovery_does_not_reset_unknown_running_or_unverified_jobs(tmp_path):
    from opendde_harness.plugin.protein_design.initial_round.storage import retry_failed_jobs

    checkpoint = Checkpoint(tmp_path, {})
    checkpoint.data["jobs"] = {
        "unknown": {"state": "submission_uncertain"},
        "running": {"state": "waiting", "id": "running"},
        "unverified": {"state": "failed", "remote_status": "UNKNOWN"},
        "success": {"state": "completed", "results": {"files": []}},
        "failed": {"state": "failed", "remote_status": "FAILED", "id": "old"},
    }
    assert retry_failed_jobs(checkpoint) == ["failed"]
    assert set(checkpoint.data["jobs"]) == {"unknown", "running", "unverified", "success"}
    assert checkpoint.data["job_attempts"]["failed"][0]["id"] == "old"


async def test_resume_rehosts_ended_api_and_queries_original_job_without_posting(tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.initial_round import services
    from opendde_harness.plugin.protein_design.initial_round.api_startup import allocation_record
    from opendde_harness.plugin.protein_design.initial_round.storage import fingerprint

    checkpoint = Checkpoint(tmp_path, {})
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        if request.url.path.endswith("/results"):
            return httpx.Response(200, json={"files": ["/shared/retained.pdb"]})
        return httpx.Response(200, json={"status": "COMPLETED", "exit_code": "0:0"})

    service = SlurmService(Service(url="https://restore.test"), checkpoint, transport=httpx.MockTransport(handler))
    await service.ensure_ready()
    old = allocation_record(service.config)
    checkpoint.data["jobs"]["score/existing"] = {
        "state": "waiting",
        "id": "original-job",
        "submission_key": "a" * 32,
        "request_hash": fingerprint({"url": service.config.url, "route": "/v1/tasks", "payload": {}}),
        "api_url": str(service.client.base_url),
        "api_allocation": old,
    }

    async def ended(config, job_id, state):
        return "COMPLETED", "restore.test"

    monkeypatch.setattr(services, "allocation_status", ended)
    try:
        assert await service.run("score/existing", "/v1/tasks", "task_id", {}) == {"files": ["/shared/retained.pdb"]}
        job = checkpoint.data["jobs"]["score/existing"]
        assert job["id"] == "original-job"
        assert job["api_allocation"]["job_id"] != old["job_id"]
        assert job["api_history"][0]["api_allocation"] == old
        assert all("original-job" in r.url.path for r in requests)
    finally:
        await service.close()


@pytest.mark.parametrize(
    "iptm,ptm,passes",
    [(0.51, 0.81, True), (0.5, 0.9, False), (0.9, 0.8, False), (0.49, 0.95, False), (0.9, 0.79, False)],
)
async def test_torchscore_joint_gate_controls_refold(round_config, fake_services, iptm, ptm, passes):
    round_config.evaluation.scoring_backend = "torchscore"
    round_config.evaluation.torchscore = Service(url="https://torchscore.test", poll_interval=0.001)
    round_config.evaluation.af3score = None
    base, calls = fake_services

    def handler(request):
        response = base.handler(request)
        if request.url.host == "torchscore.test" and request.url.path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": [{"iptm": iptm, "ptm": ptm}]})
        return response

    runner = InitialRoundRunner(round_config, transport=httpx.MockTransport(handler))
    result = await runner.run()
    assert result["accepted"] == (2 if passes else 0)
    assert calls[("torchfold", "POST")] == (2 if passes else 0)
    assert all(c["metrics"]["torchscore_ptm"] == ptm for c in runner.checkpoint.data["candidates"])
    assert "torchscore_ptm" in (runner.root / "ranking.csv").read_text().splitlines()[0]


@pytest.mark.parametrize(
    "ptm,expected",
    [(0.81, "hit"), (0.8, "needs_interface_score"), (0.79, "needs_interface_score"), (None, "evaluation_incomplete")],
)
def test_torchscore_hit_uses_both_metrics_independent_of_af3_threshold(ptm, expected):
    from opendde_harness.plugin.protein_design.initial_round.config import AutonomousRound
    from opendde_harness.plugin.protein_design.initial_round.diagnostics import classify

    metrics = {"torchscore_iptm": 0.9, "torchscore_ptm": ptm, "design_refold_dockq": 0.9}
    assert classify(metrics, {"epitope_hit": True}, AutonomousRound()) == expected


async def test_missing_torchscore_ptm_is_not_assumed_to_pass(round_config, fake_services):
    round_config.evaluation.scoring_backend = "torchscore"
    round_config.evaluation.torchscore = Service(url="https://torchscore.test", poll_interval=0.001)
    round_config.evaluation.af3score = None
    base, calls = fake_services

    def handler(request):
        response = base.handler(request)
        if request.url.host == "torchscore.test" and request.url.path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": [{"iptm": 0.99}]})
        return response

    runner = InitialRoundRunner(round_config, transport=httpx.MockTransport(handler))
    assert (await runner.run())["status"] == "partial"
    assert calls[("torchfold", "POST")] == 0
    assert all("error" in c for c in runner.checkpoint.data["candidates"])


@pytest.mark.parametrize(
    "hotspot,iptm,ptm,dockq,expected",
    [
        (False, 0.99, 0.99, 0.99, ["stop"]),
        (False, None, None, None, []),
        (True, 0.5, 0.99, 0.99, ["score_interface"]),
        (True, 0.49, 0.1, 0.1, ["terminate"]),
        (True, 0.51, 0.8, 0.99, ["score_interface"]),
        (True, 0.9, 0.79, 0.1, ["score_interface"]),
        (True, 0.9, 0.81, 0.49, ["stop"]),
        (True, 0.9, 0.9, 0.48, ["terminate"]),
        (None, 0.9, 0.9, 0.9, ["stop"]),
        (True, float("nan"), 0.9, 0.9, []),
        (True, 0.9, None, 0.9, []),
        (True, 0.9, 0.9, None, ["refold"]),
    ],
)
def test_staged_table_priorities_and_strict_boundaries(hotspot, iptm, ptm, dockq, expected):
    from opendde_harness.plugin.protein_design.initial_round.policy import rule_actions

    metrics = {"torchscore_iptm": iptm, "torchscore_ptm": ptm, "design_refold_dockq": dockq}
    assert rule_actions(metrics, hotspot) == expected


@pytest.mark.parametrize("role", ["heavy", "light"])
def test_python_cdr_alignment_includes_insertions_and_real_pdb_numbering(tmp_path, role):
    from biotite.structure.io import save_structure

    from opendde_harness.plugin.protein_design.initial_round.artifacts import structure
    from opendde_harness.plugin.protein_design.initial_round.config import Chains
    from opendde_harness.plugin.protein_design.initial_round.regions import (
        CDR_REFERENCES,
        infer_cdr_positions,
        python_partial_positions,
        selection_positions,
    )

    segments = list(CDR_REFERENCES[role])
    segments[5] += "GSGS"  # Additional CDR3 residues must remain mutable.
    sequence = "".join(segments)
    selected = infer_cdr_positions(sequence, role)
    expected, cursor = set(), 0
    for i, segment in enumerate(segments):
        if i % 2:
            expected.update(range(cursor + 1, cursor + len(segment) + 1))
        cursor += len(segment)
    assert selected == expected
    path = pdb(tmp_path / "numbered.pdb", binder=sequence)
    atoms = structure(path)
    atoms.res_id[atoms.chain_id == "B"] += 100
    save_structure(str(path), atoms)
    chains = Chains(target=["A"], binder=["B"], refold_map={"A": "X", "B": "Y"})
    # The paired role helper is tested using a real heavy chain below; direct role mapping validates light CDRs.
    if role == "heavy":
        result = python_partial_positions(atoms, chains, "B")
        mutable = selection_positions(result["cdr_position"], atoms, ["B"])["B"]
        fixed = selection_positions(result["fixed_positions"], atoms, ["B"])["B"]
        assert mutable == expected
        assert not mutable & fixed
        assert mutable | fixed == set(range(1, len(sequence) + 1))
        assert "B1-" not in result["fixed_positions"]
    with pytest.raises(ValueError, match="complete single"):
        infer_cdr_positions("AAAAAA", role)


def test_legal_menu_uses_energy_fixed_points_and_stage_without_cdr_alignment(round_config, tmp_path):
    from opendde_harness.plugin.protein_design.initial_round.autonomous import options_for
    from opendde_harness.plugin.protein_design.initial_round.runner import digest

    config = autonomous_config(round_config)
    config.generators.pop("torchcraft")
    config.evaluation.torchopt = Service(url="https://torchopt.test")
    runner = InitialRoundRunner(config)
    runner.checkpoint = Checkpoint(tmp_path / "menu", {})
    path = pdb(tmp_path / "candidate.pdb")
    candidate = {
        "candidate_id": "parent",
        "source": "ppiflow",
        "status": "rejected",
        "generated_structure": runner.files.remote(path),
        "generated_sha256": digest(path),
        "chains": {"A": "GGGGGG", "B": "AAAAAA"},
        "metrics": {"torchscore_iptm": 0.7, "torchscore_ptm": 0.9},
        "interface_fixed_positions": {"B": [1, 3]},
        "diagnostics": {"epitope_hit": False},
    }
    options = options_for(runner, [candidate], 1, 5, [])
    assert [o["action"] for o in options] == ["partial_flow"]
    assert options[0]["request"]["fixed_positions"] == "B1,B3"
    assert options[0]["request"]["cdr_position"] == "B2,B4-6"
    assert options[0]["region_method"] == "rosetta_interface_residue_energy"
    candidate["generation_request"] = {"mode": "partial_nanobody"}
    assert [o["action"] for o in options_for(runner, [candidate], 1, 5, [])] == ["torchopt"]
    candidate["optimized_by"] = "torchopt"
    assert options_for(runner, [candidate], 1, 5, []) == []


async def test_qwen_can_stop_only_when_all_four_pass_and_without_extra_submission(
    round_config, fake_services, monkeypatch
):
    from opendde_harness.plugin.protein_design.initial_round.autonomous import Choice

    seen = []

    async def choose(context):
        seen.append(context)
        option = next(item for item in context["options"] if item["action"] == "stop")
        assert all("rosetta_interface_dg" not in c["metrics"] for c in context["candidates"])
        return Choice(option_id=option["option_id"], reason="all four criteria pass; keep the current candidates")

    monkeypatch.setattr("opendde_harness.plugin.protein_design.initial_round.autonomous.agent_choice", choose)
    config = autonomous_config(round_config, decision_policy="agent")
    result = await InitialRoundRunner(config, transport=fake_services[0]).run()
    assert result["stop_reason"] == "agent_stop"
    assert result["used_samples"] == 2
    assert result["accepted"] == 2
    assert len(seen) == 1
    assert fake_services[1][("ppiflow", "POST")] == 1
    assert fake_services[1][("rosetta", "POST")] == 0


def test_python_partial_masks_cover_both_antibody_chains_without_manual_regions(tmp_path):
    import biotite.structure as struc

    from opendde_harness.plugin.protein_design.initial_round.artifacts import structure
    from opendde_harness.plugin.protein_design.initial_round.config import Chains
    from opendde_harness.plugin.protein_design.initial_round.regions import (
        CDR_REFERENCES,
        python_partial_positions,
        selection_positions,
    )

    heavy = structure(pdb(tmp_path / "heavy.pdb", binder="".join(CDR_REFERENCES["heavy"])))
    light = structure(pdb(tmp_path / "light.pdb", binder="".join(CDR_REFERENCES["light"])))
    light = light[light.chain_id == "B"]
    light.chain_id[:] = "L"
    light.res_id += 200
    atoms = struc.concatenate([heavy, light])
    chains = Chains(target=["A"], binder=["B", "L"], refold_map={"A": "X", "B": "Y", "L": "Z"})
    result = python_partial_positions(atoms, chains, "B", "L")
    mutable = selection_positions(result["cdr_position"], atoms, chains.binder)
    fixed = selection_positions(result["fixed_positions"], atoms, chains.binder)
    for chain in chains.binder:
        assert mutable[chain]
        assert not mutable[chain] & fixed[chain]
        assert mutable[chain] | fixed[chain] == set(
            range(1, len(struc.get_residues(atoms[atoms.chain_id == chain])[0]) + 1)
        )
    atoms.ins_code[(atoms.chain_id == "B") & (atoms.res_id == 1)] = "A"
    with pytest.raises(ValueError, match="cannot be encoded"):
        python_partial_positions(atoms, chains, "B", "L")


async def test_hotspot_miss_does_not_override_the_confidence_route(round_config, fake_services, monkeypatch):
    from biotite.structure.io import save_structure

    from opendde_harness.plugin.protein_design.initial_round.artifacts import structure

    config = autonomous_config(round_config, max_samples=2, max_rounds=1)
    config.generators.pop("torchcraft")
    generated = config.work_dir / "ppiflow/generated.pdb"
    atoms = structure(generated)
    atoms.coord[atoms.chain_id == "B", 1] += 100
    save_structure(str(generated), atoms)
    observed = []
    base_refold = InitialRoundRunner.refold

    async def capture(self, candidate):
        observed.append(candidate["diagnostics"]["epitope_hit"])
        await base_refold(self, candidate)

    monkeypatch.setattr(InitialRoundRunner, "refold", capture)
    runner = InitialRoundRunner(config, transport=fake_services[0])
    result = await runner.run()
    assert observed == [False]
    assert result["used_samples"] == 1
    assert fake_services[1][("torchfold", "POST")] == 1
    assert fake_services[1][("ppiflow", "POST")] == 1
    assert fake_services[1][("rosetta", "POST")] == 0


async def test_energy_fixed_points_remove_the_need_for_a_cdr_question(round_config, fake_services):
    config = autonomous_config(round_config, max_rounds=1, max_samples=2)
    config.generators.pop("torchcraft")
    base, calls = fake_services

    def handler(request):
        response = base.handler(request)
        if request.url.host == "torchscore.test" and request.url.path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": [{"iptm": 0.7, "ptm": 0.7}]})
        return response

    runner = InitialRoundRunner(config, transport=httpx.MockTransport(handler))
    result = await runner.run()
    assert result["used_samples"] == 2
    assert "pending_regions" not in runner.checkpoint.data
    assert calls[("ppiflow", "POST")] == calls[("torchscore", "POST")] == 2
    assert calls[("rosetta", "POST")] == 1
    assert calls[("torchfold", "POST")] == 0
    decision = runner.checkpoint.data["autonomous"]["decisions"][0]
    assert decision["region_method"] == "rosetta_interface_residue_energy"
    assert decision["request"]["fixed_positions"] == "B1"


def test_direct_refold_ranking_does_not_require_rosetta():
    from opendde_harness.plugin.protein_design.initial_round.autonomous import quality, rank_score
    from opendde_harness.plugin.protein_design.initial_round.policy import rule_actions

    first = {"metrics": {"torchscore_iptm": 0.9, "torchscore_ptm": 0.9, "design_refold_dockq": 0.8}}
    second = {"metrics": {**first["metrics"], "rosetta_interface_score": -20}}
    assert rank_score(first, "torchscore_iptm") == 90
    assert quality(first, "torchscore_iptm") == quality(second, "torchscore_iptm")
    assert rule_actions(first["metrics"]) == rule_actions(second["metrics"]) == ["stop"]


async def test_rosetta_failure_pauses_autonomy_and_retries_only_energy_step(round_config, fake_services):
    config = autonomous_config(round_config, max_samples=2)
    base, calls = fake_services
    failed = set()

    def handler(request):
        response = base.handler(request)
        if request.url.host == "torchscore.test" and request.url.path.endswith("/metrics"):
            return httpx.Response(200, json={"metrics": [{"iptm": 0.9, "ptm": 0.7}]})
        if request.url.host == "rosetta.test" and request.method == "POST" and not failed:
            failed.add(response.json()["task_id"])
        if request.method == "GET" and request.url.path.rsplit("/", 1)[-1] in failed:
            return httpx.Response(200, json={"status": "FAILED", "exit_code": "1:0"})
        return response

    transport = httpx.MockTransport(handler)
    with pytest.raises(RuntimeError, match="resume this run_dir"):
        await InitialRoundRunner(config, transport=transport).run()
    result = await InitialRoundRunner(config, transport=transport, retry_failed=True).run()
    assert result["used_samples"] == 2
    assert calls[("ppiflow", "POST")] == calls[("torchcraft", "POST")] == 1
    assert calls[("torchscore", "POST")] == 2
    assert calls[("torchfold", "POST")] == 0
    assert calls[("rosetta", "POST")] == 3


async def test_completed_refold_is_harvested_after_worker_crash_and_status_profile_fix(round_config, fake_services):
    class WorkerInterruptedError(BaseException):
        pass

    base, calls = fake_services
    interrupted = False
    remote_ids = []

    def handler(request):
        nonlocal interrupted
        response = base.handler(request)
        if request.url.host == "torchfold.test":
            if request.method == "POST":
                remote_ids.append(response.json()["fold_id"])
            elif request.url.path.endswith("/results"):
                if not interrupted:
                    interrupted = True
                    raise WorkerInterruptedError("worker exited after remote completion, before results were saved")
            elif request.url.path.startswith("/v1/folds/"):
                return httpx.Response(200, json={"state": "COMPLETED", "exit_code": 0})
        return response

    transport = httpx.MockTransport(handler)
    first = InitialRoundRunner(round_config, transport=transport)
    with pytest.raises(WorkerInterruptedError):
        await first.run()
    pending = [job for key, job in first.checkpoint.data["jobs"].items() if key.startswith("refold/")]
    assert len(pending) == 1
    assert pending[0]["state"] == "collecting" and pending[0]["completion_verified"]
    assert all(c["status"] == "scored" and "refold_structure" not in c for c in first.checkpoint.data["candidates"])
    original_id = pending[0]["id"]
    # Correcting the response selector must not be treated as a new scientific experiment.
    round_config.evaluation.torchfold.status_field = "state"
    restored = InitialRoundRunner(round_config, transport=transport)
    result = await restored.run()
    assert result["accepted"] == 2
    assert calls[("ppiflow", "POST")] == calls[("torchcraft", "POST")] == 1
    assert calls[("af3score", "POST")] == 2
    assert calls[("torchfold", "POST")] == 2  # One per candidate, never a replacement for the completed job.
    assert original_id in remote_ids
    assert sum(job.get("id") == original_id for job in restored.checkpoint.data["jobs"].values()) == 1
    assert all(
        c.get("refold_structure") and "torchfold_iptm" in c["metrics"] for c in restored.checkpoint.data["candidates"]
    )
    assert (
        restored.checkpoint.data["response_profile_history"][-1]["change"]
        == "compatible snapshot serialization/status_field"
    )


@pytest.mark.parametrize(
    "configured,response,expected",
    [
        ("status", {"state": "COMPLETED"}, "COMPLETED"),
        ("state", {"status": "SUCCEEDED"}, "COMPLETED"),
        ("status", {"status": "SUCCESS", "state": "COMPLETED"}, "COMPLETED"),
        ("status", {}, "UNKNOWN"),
        ("slurm.state", {"slurm": {"state": "RUNNING"}, "status": "COMPLETED"}, "RUNNING"),
    ],
)
def test_response_status_aliases_are_explicit_and_missing_is_not_success(configured, response, expected):
    from opendde_harness.plugin.protein_design.initial_round.storage import response_state

    assert response_state(response, configured) == expected
    with pytest.raises(ValueError, match="conflicting"):
        response_state({"status": "RUNNING", "state": "COMPLETED"})


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_allows_only_proven_status_profile_changes(tmp_path, legacy):
    import copy

    from opendde_harness.plugin.protein_design.initial_round.storage import write_json

    config = {
        "config": {
            "evaluation": {"torchfold": {"url": "https://fold.test", "status_field": "status"}},
            "autonomous": {"min_score": 0.5},
        },
        "inputs": {"input.pdb": "original-hash"},
    }
    checkpoint = Checkpoint(tmp_path, config)
    checkpoint.data["jobs"]["refold/candidate"] = {"id": "original-id", "state": "waiting"}
    checkpoint.save()
    if legacy:
        checkpoint.data.pop("config_snapshot")
        checkpoint.save()
        write_json(tmp_path / "run-config.json", config["config"])
    updated = copy.deepcopy(config)
    updated["config"]["evaluation"]["torchfold"]["status_field"] = "state"
    migrated = Checkpoint(tmp_path, updated)
    assert migrated.data["jobs"]["refold/candidate"]["id"] == "original-id"
    assert len(migrated.data["response_profile_history"]) == 1
    for change in ("url", "threshold", "input"):
        incompatible = copy.deepcopy(updated)
        if change == "url":
            incompatible["config"]["evaluation"]["torchfold"]["url"] = "https://another.test"
        elif change == "threshold":
            incompatible["config"]["autonomous"]["min_score"] = 0.9
        else:
            incompatible["inputs"]["input.pdb"] = "changed-hash"
        with pytest.raises(ValueError, match="configuration changed"):
            Checkpoint(tmp_path, incompatible)


async def test_result_collection_retries_temporary_unavailability_without_new_post(tmp_path):
    checkpoint = Checkpoint(tmp_path, {})
    posts, result_requests = [], []

    def handler(request):
        if request.method == "POST":
            posts.append(request)
            return httpx.Response(202, json={"fold_id": "completed-fold"})
        if request.url.path.endswith("/results"):
            result_requests.append(request)
            return httpx.Response(409 if len(result_requests) == 1 else 200, json={"files": []})
        return httpx.Response(200, json={"state": "COMPLETED", "exit_code": 0})

    service = SlurmService(
        Service(url="https://fold.test", poll_interval=0.001), checkpoint, transport=httpx.MockTransport(handler)
    )
    try:
        assert await service.run("refold/candidate", "/v1/folds", "fold_id", {}) == {"files": []}
        assert len(posts) == 1 and len(result_requests) == 2
        assert checkpoint.data["jobs"]["refold/candidate"]["state"] == "completed"
    finally:
        await service.close()


def test_prefixed_confidence_pairs_with_unique_unprefixed_model_and_rejects_ambiguity(round_config, tmp_path):
    folder = tmp_path / "outputs"
    folder.mkdir()
    model = pdb(folder / "model.pdb", target="X", chain="Y")
    confidence = folder / "ppiflow-candidate_summary_confidences.json"
    confidence.write_text(json.dumps({"iptm": 0.92}))
    files = SharedFiles(round_config.path_mappings)
    result = {"files": [files.remote(model), files.remote(confidence)]}
    service = round_config.evaluation.torchfold
    assert refold_output(result, service, files, "*summary_confidences*.json") == (model, 0.92)
    another = folder / "another_summary_confidences.json"
    another.write_text(json.dumps({"iptm": 0.99}))
    result["files"].append(files.remote(another))
    with pytest.raises(ValueError, match="pair with exactly one"):
        refold_output(result, service, files, "*summary_confidences*.json")


def test_legacy_checkpoint_resume_normalizes_snapshot_without_losing_remote_jobs(round_config, tmp_path):
    from opendde_harness.plugin.protein_design.initial_round.storage import fingerprint, write_json

    legacy = round_config.model_dump(mode="json")
    legacy["generators"]["ppiflow"]["service"].pop("startup_poll_interval")
    legacy["evaluation"]["torchfold"].pop("slurm_command_timeout")
    old_payload = {"config": legacy, "inputs": {"input.pdb": "same-digest"}}
    write_json(tmp_path / "run-config.json", legacy)
    write_json(
        tmp_path / "checkpoint.json",
        {
            "config_hash": fingerprint(old_payload),
            "jobs": {"refold/candidate": {"id": "existing-remote-id", "state": "waiting"}},
            "version": 1,
        },
    )
    new_payload = {"config": round_config.model_dump(mode="json"), "inputs": old_payload["inputs"]}
    restored = Checkpoint(tmp_path, new_payload)
    assert restored.data["jobs"]["refold/candidate"]["id"] == "existing-remote-id"
    assert (
        "config.evaluation.torchfold.slurm_command_timeout"
        in restored.data["response_profile_history"][0]["serialized_differences"]
    )
    incompatible = {**new_payload, "inputs": {"input.pdb": "different-digest"}}
    with pytest.raises(ValueError, match=r"inputs\.input\.pdb"):
        Checkpoint(tmp_path, incompatible)


def test_configuration_identity_normalizes_aliases_but_keeps_scientific_changes(round_config):
    import copy

    from opendde_harness.plugin.protein_design.initial_round.storage import (
        config_difference_paths,
        status_profile_identity,
    )

    config = autonomous_config(round_config).model_dump(mode="json")
    legacy = copy.deepcopy(config)
    legacy["autonomous"]["min_af3score"] = legacy["autonomous"].pop("min_score")
    assert status_profile_identity(legacy) == status_profile_identity(config)
    legacy["autonomous"]["min_af3score"] = 0.95
    assert config_difference_paths(status_profile_identity(legacy), status_profile_identity(config)) == [
        "autonomous.min_score"
    ]


def test_ppiflow_binder_requires_api_chain_and_resolves_explicit_workflow_chain(round_config):
    from opendde_harness.plugin.protein_design.core.model_requests import validate_request

    request = {
        "mode": "binder",
        "name": "test",
        "input_pdb": "/shared/target-only.pdb",
        "target_chain": "A",
        "samples_min_length": 30,
        "samples_max_length": 60,
    }
    with pytest.raises(ValueError, match="binder_chain"):
        validate_request("ppiflow", request)
    value = round_config.model_dump(mode="json")
    value["generators"] = {"ppiflow": value["generators"]["ppiflow"]}
    value["generators"]["ppiflow"]["request"] = request
    config = InitialRoundConfig.model_validate(value)
    assert config.generators["ppiflow"].request["binder_chain"] == "B"

    for field, value_to_reject in (
        ("framework_dir", "/shared/frameworks"),
        ("fixed_positions", "B1"),
        ("num_timesteps", 200),
    ):
        rejected = {**request, field: value_to_reject}
        with pytest.raises(ValueError, match=f"unsupported fields.*{field}"):
            validate_request("ppiflow", {**rejected, "binder_chain": "B"})
        value["generators"]["ppiflow"]["request"] = rejected
        with pytest.raises(ValueError, match=f"unsupported fields.*{field}"):
            InitialRoundConfig.model_validate(value)


@pytest.mark.parametrize(
    ("mode", "request_fields", "chain_fields"),
    [
        (
            "binder",
            {"input_pdb": "/shared/target.pdb", "samples_min_length": 30, "samples_max_length": 60},
            {"target_chain": "C", "binder_chain": "A"},
        ),
        (
            "nanobody",
            {"antigen_pdb": "/shared/target.pdb", "framework_dir": "/shared/frameworks"},
            {"antigen_chain": "C", "heavy_chain": "A"},
        ),
        (
            "antibody",
            {
                "antigen_pdb": "/shared/target.pdb",
                "framework_dir": "/shared/frameworks",
                "cdr_length": "CDRH3,10-20,CDRL3,9-11",
            },
            {"antigen_chain": "C", "heavy_chain": "A", "light_chain": "B"},
        ),
    ],
)
def test_ppiflow_workflow_uses_explicit_request_chains(round_config, mode, request_fields, chain_fields):
    from opendde_harness.plugin.protein_design.initial_round.automation import prepare_run

    value = round_config.model_dump(mode="json")
    ppiflow = value["generators"]["ppiflow"]
    value["generators"] = {"ppiflow": ppiflow}
    ppiflow.pop("chains")
    ppiflow["request"] = {"mode": mode, "name": value["name"], **request_fields, **chain_fields}
    generator = prepare_run(value, registrations={}).generators["ppiflow"]
    assert generator.chains.target == ["C"]
    assert generator.chains.binder == (["A", "B"] if mode == "antibody" else ["A"])
    assert generator.chains.refold_map == {chain: chain for chain in chain_fields.values()}
    assert {field: generator.request[field] for field in chain_fields} == chain_fields
    ppiflow["request"].pop(next(iter(chain_fields)))
    with pytest.raises(ValueError, match="chains.target"):
        prepare_run(value, registrations={})


@pytest.mark.parametrize(
    ("mode", "overrides", "field"),
    [
        ("nanobody", {"light_chain": "L"}, "light_chain"),
        ("antibody", {"fixed_positions": "H1"}, "fixed_positions"),
        ("partial_nanobody", {"fixed_positions": ["H1"]}, "fixed_positions"),
        ("partial_nanobody", {"start_t": 1.01}, "start_t"),
        ("partial_antibody", {"cdr_position": ["H2"]}, "cdr_position"),
        ("partial_antibody", {"framework_dir": "/shared/framework"}, "framework_dir"),
        ("nanobody", {"num_timesteps": 2001}, "num_timesteps"),
    ],
)
def test_ppiflow_request_matches_deployed_mode_schema(mode, overrides, field):
    from opendde_harness.plugin.protein_design.core.model_requests import validate_request

    payload = {"mode": mode, "antigen_chain": "C", "heavy_chain": "H"}
    if mode.startswith("partial_"):
        payload.update(complex_pdb="/shared/complex.pdb", fixed_positions="", cdr_position="H2", start_t=1)
    else:
        payload.update(antigen_pdb="/shared/target.pdb", framework_dir="/shared/framework")
    if mode.endswith("antibody"):
        payload["light_chain"] = "L"
    if mode == "antibody":
        payload["cdr_length"] = "CDRH3,10-20"

    resolved = validate_request("ppiflow", payload)
    assert resolved["name"] == "target"
    assert resolved["samples_per_target"] == 5
    with pytest.raises(ValueError, match=field):
        validate_request("ppiflow", {**payload, **overrides})
