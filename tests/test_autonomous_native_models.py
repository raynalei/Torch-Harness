"""Native optimization artifacts must traverse both scientific stages faithfully."""

import gzip
import json

import httpx
import pytest

from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig
from opendde_harness.plugin.protein_design.initial_round.native_models import validate_sequences
from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner
from opendde_harness.plugin.protein_design.initial_round.task_inputs import prepare_design_task
from opendde_harness.plugin.protein_design.scientific_agent import planner
from tests._slurm_api import mock_slurm_apis  # noqa: F401
from tests.test_persistent_workflow_defaults import settings, task  # noqa: F401
from tests.test_protein_design_initial_round import autonomous_config, fake_services, pdb, round_config  # noqa: F401
from tests.test_scientific_planner import choose

pytestmark = [pytest.mark.in_memory_transport, pytest.mark.usefixtures("mock_slurm_apis")]


def with_native_models(config):
    value = autonomous_config(config, decision_policy="planner", planner_layout="two_stage").model_dump(mode="json")
    for kind in ("rfd4", "proteinmpnn"):
        value["evaluation"][kind] = {"url": f"https://{kind}.test", "poll_interval": 0.001}
    value["evaluation"]["rfd4_input_root"] = "/shared/rfd4-inputs"
    return InitialRoundConfig.model_validate(value)


@pytest.mark.parametrize("action", ["rfd4_partial", "proteinmpnn"])
@pytest.mark.parametrize("optimization_stage", [1, 2])
async def test_native_action_two_stage_resume_and_fresh_evaluation(
    round_config, fake_services, tmp_path, monkeypatch, action, optimization_stage
):
    original, calls = fake_services
    stages, submissions = [], []
    kind = "rfd4" if action == "rfd4_partial" else action
    generated = pdb(tmp_path / "native.pdb")
    fasta = tmp_path / "sequences.fa"
    fasta.write_text(">native, score=1\nAAAAAA\n>sample=1, score=0.1\nAAAAAA\n")

    async def handler(request):
        if request.url.host != f"{kind}.test":
            return await original.handle_async_request(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        if request.method == "POST":
            payload = json.loads(request.content)
            submissions.append(payload)
            if kind == "rfd4":
                assert payload["inputs"].endswith(".cif")
                assert payload["generation"] == "inference_base"
                assert payload["partial_trajectory_fraction_steps"] > 0
            else:
                assert payload["design_chains"] == ["B"]
            return httpx.Response(202, json={"task_id": "a" * 32})
        if request.url.path.endswith("/results"):
            result = (
                {"output_dir": "/shared", "files": [generated.name]}
                if kind == "rfd4"
                else {"fasta_path": "/shared/sequences.fa"}
            )
            return httpx.Response(200, json=result)
        return httpx.Response(200, json={"status": "succeeded", "exit_code": 0})

    async def decide(context, **kwargs):
        stage = context["scientific_decision_stage"]
        stages.append(stage)
        current = context["candidates"][0]
        if stage == optimization_stage and not current.get("parent_id"):
            # Parent lineage is in history/context; use the candidate ID prefix instead.
            if not current["candidate_id"].startswith(action + "-"):
                choice = choose(context, action)
                return choice.model_copy(
                    update={"selection": planner.MutationSelection(mask_positions={"B": [1, 2, 3, 4, 5, 6]})}
                )
        return choose(context, "continue")

    monkeypatch.setattr(planner, "planner_choice", decide)
    config = with_native_models(round_config)
    transport = httpx.MockTransport(handler)
    runner = InitialRoundRunner(config, transport=transport)
    result = await runner.run()
    assert result["accepted"] == 2
    assert len(submissions) == 2
    assert stages == [1, 1, 2, 2]
    children = [c for c in runner.checkpoint.data["candidates"] if c.get("optimized_by") == action]
    assert len(children) == 2
    assert all(c["metrics"]["torchscore_iptm"] == 0.9 for c in children)
    assert all(c["final_validation"]["decision_stage"] == 2 for c in children)
    assert calls[("torchscore", "POST")] == 4
    assert calls[("torchfold", "POST")] == (4 if kind == "proteinmpnn" else 2)
    before = dict(calls)
    await InitialRoundRunner(config, transport=transport).run()
    assert calls == before
    assert len(submissions) == 2


@pytest.mark.parametrize("changed", [{"A": "AAAAAA", "B": "AAAAAA"}, {"A": "GGGGGG", "B": "AAVAAA"}])
def test_native_output_rejects_target_or_fixed_residue_changes(round_config, changed):
    parent = {"chains": {"A": "GGGGGG", "B": "AAAAAA"}}
    partition = {"fixed_positions": {"B": [3]}, "mask_positions": {"B": [1, 2]}}
    with pytest.raises(ValueError):
        validate_sequences(parent, changed, round_config.generators["ppiflow"].chains, partition)


@pytest.mark.parametrize("native_only", [False, True])
async def test_rfd4_initial_generation_enters_both_decisions(
    round_config, fake_services, tmp_path, monkeypatch, native_only
):
    original, calls = fake_services
    config = with_native_models(round_config).model_dump(mode="json")
    config["generators"]["rfd4"] = {
        "service": config["evaluation"]["rfd4"],
        "request": {"inputs": "specs/new-binder.yaml", "num_replicates": 1},
        "chains": config["generators"]["ppiflow"]["chains"],
        "expected_target_sequences": {"A": "GGGGGG"},
    }
    if native_only:
        config["generators"] = {"rfd4": config["generators"]["rfd4"]}
    spec = tmp_path / "rfd4-inputs" / "specs" / "new-binder.yaml"
    spec.parent.mkdir(parents=True)
    spec.write_text("# Native spec fixture; inference is simulated\n")
    from biotite.structure.io import save_structure

    from opendde_harness.plugin.protein_design.initial_round.artifacts import structure

    source = pdb(tmp_path / "rfd-initial.pdb")
    cif = tmp_path / "rfd-initial.cif"
    save_structure(str(cif), structure(source))
    with gzip.open(tmp_path / "rfd-initial.cif.gz", "wb") as compressed:
        compressed.write(cif.read_bytes())
    submissions, stages = [], []

    async def handler(request):
        if request.url.host != "rfd4.test":
            return await original.handle_async_request(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        if request.method == "POST":
            submissions.append(json.loads(request.content))
            assert request.url.path == "/v1/tasks"
            assert "name" not in submissions[-1]
            return httpx.Response(202, json={"task_id": "b" * 32})
        if request.url.path.endswith("/results"):
            return httpx.Response(200, json={"output_dir": "/shared", "files": ["rfd-initial.cif.gz"]})
        return httpx.Response(200, json={"status": "succeeded", "exit_code": 0})

    async def decide(context, **kwargs):
        stages.append(context["scientific_decision_stage"])
        return choose(context, "continue")

    monkeypatch.setattr(planner, "planner_choice", decide)
    runner = InitialRoundRunner(InitialRoundConfig.model_validate(config), transport=httpx.MockTransport(handler))
    result = await runner.run()
    total = 1 if native_only else 3
    assert result["accepted"] == total
    assert stages == [1] * total + [2] * total
    assert len(submissions) == 1
    assert calls[("torchscore", "POST")] == total


def test_rfd4_task_uses_registered_deployment_and_current_target(task, settings):
    settings["model_services"]["rfd4"] = {
        "kind": "rfd4",
        "service": {"url": "https://rfd4.test", "environment": {"RFD4_INPUTS": "/shared/rfd4-inputs"}},
    }
    task.update(
        generators=["rfd4"],
        rfd4_inputs="specs/new-binder.yaml",
        generator_overrides={
            "rfd4": {
                "chains": {
                    "target": ["A"],
                    "binder": ["B"],
                    "refold_map": {"A": "X", "B": "Y"},
                }
            }
        },
    )
    config = prepare_design_task(task, settings, planner_layout="two_stage")
    generator = config.generators["rfd4"]
    assert generator.request["inputs"] == "specs/new-binder.yaml"
    assert generator.request["num_replicates"] == 10
    assert generator.expected_target_sequences == {"A": "GGGGGG"}
    assert generator.scientific_mode == "vhh"
    assert config.evaluation.rfd4_input_root == "/shared/rfd4-inputs"
    assert config.autonomous.planner_layout == "two_stage"


async def test_sequence_realization_keeps_mutations_and_restores_chain_names(round_config, tmp_path):
    from opendde_harness.plugin.protein_design.initial_round.artifacts import chain_sequences
    from opendde_harness.plugin.protein_design.initial_round.native_models import materialize_sequence

    runner = InitialRoundRunner(with_native_models(round_config))
    result_dir = tmp_path / "materialized"
    pdb(result_dir / "model.pdb", target="X", chain="Y", binder="VAAAAA")
    (result_dir / "summary_confidences.json").write_text(json.dumps({"iptm": 0.9}))
    expected = {"A": "GGGGGG", "B": "VAAAAA"}

    async def job(service, key, route, id_field, payload):
        value = json.loads(runner.files.local(payload["json_path"]).read_text())
        proteins = {s["protein"]["id"]: s["protein"]["sequence"] for s in value["sequences"]}
        assert proteins == {"X": "GGGGGG", "Y": "VAAAAA"}
        assert key.startswith("materialize/")
        return {"files": ["/shared/materialized/model.pdb", "/shared/materialized/summary_confidences.json"]}

    runner.job = job
    output = await materialize_sequence(runner, {}, expected, runner.config.generators["ppiflow"], "sequence-test")
    assert chain_sequences(output) == expected
