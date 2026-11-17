"""Planner autonomy, native analysis, constrained masks and checkpoint recovery."""

import json

import pytest

from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner
from opendde_harness.plugin.protein_design.scientific_agent import planner
from opendde_harness.plugin.protein_design.scientific_agent.mutation_selector import select_mutations
from tests._slurm_api import mock_slurm_apis  # noqa: F401
from tests.test_protein_design_initial_round import autonomous_config, fake_services, round_config  # noqa: F401

pytestmark = [pytest.mark.in_memory_transport, pytest.mark.usefixtures("mock_slurm_apis")]


def choose(context, action, **fields):
    for option in context["options"]:
        if option["action"] == action and all(option.get(k) == v for k, v in fields.items()):
            return planner.PlannerChoice(option_id=option["option_id"], reason="measured scientific evidence")
    raise AssertionError(f"missing action {action}: {context['options']}")


@pytest.mark.parametrize("selection", [{"fixed_positions": {"B": [2]}}, {"mask_positions": {"B": [3, 4]}}])
def test_mutation_selector_partitions_and_preserves_required_fixed(selection):
    result = select_mutations({"B": "ACDE"}, selection, required_fixed={"B": [1]})
    assert 1 in result["fixed_positions"]["B"]
    assert set(result["fixed_positions"]["B"]) | set(result["mask_positions"]["B"]) == {1, 2, 3, 4}
    assert not set(result["fixed_positions"]["B"]) & set(result["mask_positions"]["B"])


@pytest.mark.parametrize(
    "selection",
    [
        {"mask_positions": {"B": [0]}},
        {"mask_positions": {"B": [True]}},
        {"mask_positions": {"A": [1]}},
        {"mask_positions": {"B": [1]}},
        {"mask_positions": {"B": [2, 2]}},
        {"mask_positions": {"B": [4]}},
    ],
)
def test_mutation_selector_rejects_invalid_or_out_of_scope_masks(selection):
    with pytest.raises(ValueError):
        select_mutations({"B": "ACDE"}, selection, required_fixed={"B": [1]}, allowed_positions={"B": [2, 3]})


async def test_planner_runs_analysis_then_reconsiders_all_metrics(round_config, fake_services, monkeypatch):
    transport, calls = fake_services
    contexts = []

    async def decide(context, **kwargs):
        contexts.append(context)
        assert context["decision_mode"] == "qwen_planner"
        assert context["evaluation_skills"]["developability-filter"]["source"].endswith("SKILL.md")
        parent = context["options"][0]["parent_id"]
        record = next(c for c in context["candidates"] if c["candidate_id"] == parent)
        if "developability-filter" not in record["completed_analysis"]:
            return choose(context, "analysis", evaluation_skill="developability-filter", parent_id=parent)
        assert record["developability"]["ptm_risks"]["available"] is True
        assert record["developability"]["immunogenicity"]["available"] is False
        if any(
            o["action"] == "continue" and o.get("phase") == "refold" and o["parent_id"] == parent
            for o in context["options"]
        ):
            return choose(context, "continue", phase="refold", parent_id=parent)
        return choose(context, "continue", phase="complete", parent_id=parent)

    monkeypatch.setattr(planner, "planner_choice", decide)
    config = autonomous_config(round_config, decision_policy="planner")
    runner = InitialRoundRunner(config, transport=transport)
    result = await runner.run()
    assert result["accepted"] == 2
    assert calls[("rosetta", "POST")] == 0
    assert len(contexts) == 6
    before = dict(calls)
    await InitialRoundRunner(config, transport=transport).run()
    assert dict(calls) == before
    assert len(contexts) == 6


async def test_planner_can_optimize_high_confidence_without_rosetta_gate(round_config, fake_services, monkeypatch):
    transport, calls = fake_services

    async def decide(context, **kwargs):
        choice = choose(context, "partial_flow")
        return choice.model_copy(update={"selection": planner.MutationSelection(fixed_positions={"B": [1]})})

    monkeypatch.setattr(planner, "planner_choice", decide)
    config = autonomous_config(round_config, decision_policy="planner")
    result = await InitialRoundRunner(config, transport=transport).run()
    assert calls[("rosetta", "POST")] == 0
    assert result["accepted"] == 0
    decisions = json.loads((config.work_dir / config.name / "decisions.json").read_text())["decisions"]
    assert all(d["action"] == "partial_flow" for d in decisions)
    assert all(d["mutation_selector"]["tool"] == "protein_design_mutation_selector" for d in decisions)
    assert all(d["request"]["fixed_positions"] == "B1" for d in decisions)


async def test_unavailable_planner_uses_recorded_rule_fallback(round_config, fake_services, monkeypatch):
    transport, calls = fake_services

    async def unavailable(*args, **kwargs):
        raise TimeoutError("Qwen unavailable")

    monkeypatch.setattr(planner, "planner_choice", unavailable)
    config = autonomous_config(round_config, decision_policy="planner")
    result = await InitialRoundRunner(config, transport=transport).run()
    assert result["accepted"] == 2
    assert calls[("rosetta", "POST")] == 0
    decisions = json.loads((config.work_dir / config.name / "decisions.json").read_text())["decisions"]
    assert all(d["policy"] == "rules_fallback" for d in decisions)
    assert all("Qwen unavailable" in d["fallback"] for d in decisions)


async def test_low_confidence_is_presented_to_planner_before_stop(round_config, fake_services, monkeypatch):
    transport, calls = fake_services
    # This configured threshold would terminate immediately in rule mode.
    seen = []

    async def decide(context, **kwargs):
        seen.append(context)
        assert any(c["rule_recommendation"] == ["terminate"] for c in context["candidates"])
        assert any(o["action"] == "partial_flow" for o in context["options"])
        return choose(context, "stop")

    monkeypatch.setattr(planner, "planner_choice", decide)
    config = autonomous_config(round_config, decision_policy="planner", min_score=0.99)
    result = await InitialRoundRunner(config, transport=transport).run()
    assert len(seen) == 2
    assert result["accepted"] == 0
    assert calls[("torchfold", "POST")] == 0


@pytest.mark.parametrize("mutated_position,accepted", [(1, False), (2, True)])
async def test_optimizer_outputs_enforce_selector_mask(round_config, tmp_path, monkeypatch, mutated_position, accepted):
    from types import SimpleNamespace

    from biotite.structure.io.pdb import PDBFile

    from opendde_harness.plugin.protein_design.initial_round.artifacts import chain_sequences
    from tests.test_protein_design_initial_round import pdb

    original = pdb(tmp_path / "parent.pdb")
    atoms = PDBFile.read(original).get_structure(model=1)
    atoms.res_name[(atoms.chain_id == "B") & (atoms.res_id == mutated_position)] = "LYS"
    output = tmp_path / "optimized.pdb"
    writer = PDBFile()
    writer.set_structure(atoms)
    writer.write(output)
    parent = {
        "candidate_id": "parent",
        "source": "ppiflow",
        "chains": chain_sequences(original),
        "generated_structure": "/shared/parent.pdb",
    }
    runner = InitialRoundRunner(autonomous_config(round_config, decision_policy="planner"))
    runner.root.mkdir(parents=True, exist_ok=True)
    runner.checkpoint = SimpleNamespace(data={"candidates": [parent]})
    monkeypatch.setattr(runner, "save_candidates", lambda: None)

    class Service:
        async def run_optimization(self, *args, **kwargs):
            return {"structures": ["/shared/optimized.pdb"], "results": []}

    runner.services["torchopt"] = Service()
    decision = {
        "parent_id": "parent",
        "request": {},
        "mutation_constraints": {"fixed_positions": {"B": [1, 3, 4, 5, 6]}, "mask_positions": {"B": [2]}},
    }
    if accepted:
        children = await runner.optimize(decision, 1)
        assert children[0]["chains"]["B"] == "AKAAAA"
        assert children[0]["mutation_constraints"] == decision["mutation_constraints"]
    else:
        with pytest.raises(ValueError, match="preserves target"):
            await runner.optimize(decision, 1)
        assert runner.checkpoint.data["candidates"] == [parent]


async def test_invalid_planner_selection_falls_back_without_submitting_partial(
    round_config, fake_services, monkeypatch
):
    transport, calls = fake_services

    async def decide(context, **kwargs):
        choice = choose(context, "partial_flow")
        return choice.model_copy(update={"selection": planner.MutationSelection(mask_positions={"B": [1000]})})

    monkeypatch.setattr(planner, "planner_choice", decide)
    config = autonomous_config(round_config, decision_policy="planner")
    result = await InitialRoundRunner(config, transport=transport).run()
    assert result["accepted"] == 2
    assert calls[("ppiflow", "POST")] == 1
    decisions = json.loads((config.work_dir / config.name / "decisions.json").read_text())["decisions"]
    assert all(d["policy"] == "rules_fallback" for d in decisions)


async def test_failed_execution_resumes_same_planned_action_without_replanning(
    round_config, fake_services, monkeypatch
):
    transport, _ = fake_services
    selected = []
    fail_once = [True]
    real_analysis = planner.run_analysis

    async def analysis(*args):
        if fail_once.pop() if fail_once else False:
            raise RuntimeError("temporary analysis failure")
        return await real_analysis(*args)

    async def decide(context, **kwargs):
        parent = context["options"][0]["parent_id"]
        record = next(c for c in context["candidates"] if c["candidate_id"] == parent)
        selected.append(parent)
        if "developability-filter" not in record["completed_analysis"]:
            return choose(context, "analysis", evaluation_skill="developability-filter", parent_id=parent)
        return choose(context, "stop", parent_id=parent)

    monkeypatch.setattr(planner, "run_analysis", analysis)
    monkeypatch.setattr(planner, "planner_choice", decide)
    config = autonomous_config(round_config, decision_policy="planner")
    with pytest.raises(RuntimeError, match="temporary analysis"):
        await InitialRoundRunner(config, transport=transport).run()
    assert len(selected) == 1
    await InitialRoundRunner(config, transport=transport).run()
    assert len(selected) == 4
    state = json.loads((config.work_dir / config.name / "decisions.json").read_text())
    assert len(state["decisions"]) == 4
    assert all(d["state"] == "completed" and not d["fallback"] for d in state["decisions"])


async def test_rule_fallback_runs_rosetta_before_energy_fixed_partial(round_config, fake_services, monkeypatch):
    transport, calls = fake_services

    async def unavailable(*args, **kwargs):
        raise TimeoutError("planner offline")

    monkeypatch.setattr(planner, "planner_choice", unavailable)
    config = autonomous_config(round_config, decision_policy="planner", min_refold_score=0.99)
    await InitialRoundRunner(config, transport=transport).run()
    decisions = json.loads((config.work_dir / config.name / "decisions.json").read_text())["decisions"]
    assert calls[("rosetta", "POST")] == 2
    assert [d["action"] for d in decisions].count("partial_flow") == 2
    assert all(d["request"]["fixed_positions"] == "B1" for d in decisions if d["action"] == "partial_flow")


def test_default_policy_is_planner_and_rosetta_is_optional(round_config):
    from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig

    value = autonomous_config(round_config).model_dump(mode="json")
    value["autonomous"] = {}
    value["evaluation"]["rosetta"] = None
    config = InitialRoundConfig.model_validate(value)
    assert config.autonomous.decision_policy == "planner"
    assert "planner_model" not in config.autonomous.model_dump()
    value["autonomous"] = {"decision_policy": "rules"}
    with pytest.raises(ValueError, match="require.*Rosetta"):
        InitialRoundConfig.model_validate(value)


async def test_mutation_selector_is_registered_and_validates_requests(tmp_path):
    from opendde_harness.plugin import active_registry
    from opendde_harness.plugin.context import ServiceLocator

    tool = active_registry().build_tool(
        "protein_design_mutation_selector",
        config={},
        services=ServiceLocator(workspace=tmp_path, user_id="test", agent_id="test"),
    )
    result = await tool.execute(sequences={"B": "ACDE"}, selection={"mask_positions": {"B": [3]}})
    assert json.loads(result.model_text)["fixed_positions"] == {"B": [1, 2, 4]}


def test_planner_context_prioritizes_chemical_liabilities_beyond_first_residues():
    from opendde_harness.plugin.protein_design.scientific_agent.context_builder import planner_developability

    rows = [{"chain": "B", "sequence_position": i, "hydropathy": 1.8, "surface_exposed": False} for i in range(1, 41)]
    candidate = {
        "metadata": {
            "developability_evidence": {
                "ptm_risks": {"sites": [{"chain": "B", "position": 40}]},
                "surface_patches": {"residues": rows},
            }
        }
    }
    context = planner_developability(candidate)
    assert context["surface_patches"]["residues"][0]["sequence_position"] == 40
    assert context["surface_patches"]["residues_omitted"] == 28


async def test_initial_analysis_can_lead_directly_to_torchopt_and_fresh_child_score(
    round_config, fake_services, monkeypatch
):
    from opendde_harness.plugin.protein_design.initial_round.artifacts import chain_sequences
    from opendde_harness.plugin.protein_design.initial_round.config import Service
    from opendde_harness.plugin.protein_design.initial_round.runner import digest
    from tests.test_protein_design_initial_round import pdb

    transport, calls = fake_services
    submitted = []

    async def decide(context, **kwargs):
        parent = context["options"][0]["parent_id"]
        record = next(c for c in context["candidates"] if c["candidate_id"] == parent)
        if record["stage"] == "torchopt":
            assert record["metrics"]["torchscore_iptm"] == 0.9
            return choose(context, "stop", parent_id=parent)
        if "developability-filter" not in record["completed_analysis"]:
            return choose(context, "analysis", evaluation_skill="developability-filter", parent_id=parent)
        choice = choose(context, "torchopt", parent_id=parent)
        return choice.model_copy(update={"selection": planner.MutationSelection(mask_positions={"B": [2]})})

    async def optimize(self, decision, round_index):
        submitted.append(decision)
        parent = next(c for c in self.checkpoint.data["candidates"] if c["candidate_id"] == decision["parent_id"])
        path = pdb(self.root / f"child-{round_index}.pdb", binder="ACAAAA")
        child = {
            "candidate_id": f"child-{round_index}",
            "source": parent["source"],
            "parent_id": parent["candidate_id"],
            "round_index": round_index,
            "optimized_by": "torchopt",
            "generated_structure": self.files.remote(path),
            "generated_sha256": digest(path),
            "sequence": "ACAAAA",
            "chains": chain_sequences(path),
            "metrics": {},
            "status": "generated",
        }
        self.checkpoint.data["candidates"].append(child)
        return [child]

    monkeypatch.setattr(planner, "planner_choice", decide)
    monkeypatch.setattr(InitialRoundRunner, "optimize", optimize)
    config = autonomous_config(round_config, decision_policy="planner")
    config.evaluation.torchopt = Service(url="https://torchopt.test")
    await InitialRoundRunner(config, transport=transport).run()
    assert len(submitted) == 2
    assert all(d["mutation_constraints"]["mask_positions"] == {"B": [2]} for d in submitted)
    assert all("fixed_positions" not in d["request"] for d in submitted)
    assert calls[("torchscore", "POST")] == 4
    assert calls[("rosetta", "POST")] == 0
