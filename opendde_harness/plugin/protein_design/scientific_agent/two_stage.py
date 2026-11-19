"""Two scientific decision stages, with strictly separated evaluation evidence."""

from __future__ import annotations

import asyncio

from ..initial_round.autonomous import rank_score, sample_count
from ..initial_round.native_models import OPTIMIZATION_ACTIONS, optimize_native
from ..initial_round.policy import candidate_stage, rule_actions
from ..initial_round.storage import write_json
from . import planner
from .context_builder import context_for_runner


def active_candidates(candidates):
    return [c for c in candidates if not c.get("planner_closed")]


async def evaluate_stage(runner, candidate, stage):
    ready = candidate.setdefault("scientific_evaluations", {})
    if str(stage) in ready:
        return
    # Both scores are mandatory before either decision. Same-structure cached
    # measurements are reusable; descendants never inherit their parent's scores.
    await runner.score(candidate)
    await runner.score_interface(candidate)
    options = runner.config.autonomous
    candidate["diagnostics"] = {
        "stage": candidate_stage(candidate),
        "diagnosis": "first_stage_screen" if stage == 1 else "second_stage_evaluation",
        "allowed_actions": rule_actions(
            candidate["metrics"],
            score_metric=runner.config.evaluation.score_metric,
            min_iptm=options.min_score,
            min_ptm=options.min_ptm,
            min_dockq=options.min_dockq,
            min_refold_iptm=options.min_refold_score,
            stage=candidate_stage(candidate),
            fixed_positions=candidate.get("interface_fixed_positions"),
        ),
    }
    ranking = rank_score(candidate, runner.config.evaluation.score_metric)
    if ranking is None:
        candidate["metrics"].pop("ranking_score", None)
    else:
        candidate["metrics"]["ranking_score"] = ranking
    evaluators = ["torchscore", "rosetta-interface"]
    if stage == 2:
        for skill in planner.EVALUATION_SKILLS:
            await planner.run_analysis(runner, candidate, skill)
        evaluators.extend(planner.EVALUATION_SKILLS)
    ready[str(stage)] = {"evaluators": evaluators, "structure_sha256": candidate["generated_sha256"]}
    candidate["status"] = "scored"
    runner.save_candidates()


def stage_options(runner, candidates, stage):
    result = []
    for option in planner.planner_options(runner, candidates):
        if option["action"] == "analysis":
            continue
        option = dict(option)
        if option["action"] == "continue":
            option["phase"] = "advance" if stage == 1 else "final_validation"
            if stage == 1:
                result.append({**option, "action": "analysis", "source": "stage_two_evaluation"})
        option["scientific_decision_stage"] = stage
        result.append(option)
    return result


async def final_validation(runner, candidate, reason):
    # This is validation, not a third planner decision or an inherited pass.
    await runner.score(candidate)
    ranking = rank_score(candidate, runner.config.evaluation.score_metric)
    if ranking is None:
        candidate["metrics"].pop("ranking_score", None)
    else:
        candidate["metrics"]["ranking_score"] = ranking
    await runner.refold(candidate)
    candidate.update(
        planner_closed=True, planner_reason=reason, scientific_phase="final_validation", scientific_decision_stage=2
    )
    candidate["final_validation"] = {
        "decision_stage": 2,
        "scoring_backend": "torchscore",
        "fold_backend": "torchfold",
        "dockq_reference": "generated_design_pose",
        "status": candidate["status"],
    }
    runner.save_candidates()


async def execute_stage_decision(runner, decision, index):
    candidate = next(c for c in runner.checkpoint.data["candidates"] if c["candidate_id"] == decision["parent_id"])
    stage, action = decision["scientific_decision_stage"], decision["action"]
    from ..initial_round.runner import digest

    if digest(runner.files.local(candidate["generated_structure"])) != candidate["generated_sha256"]:
        raise ValueError("candidate structure changed after stage evaluation")
    if action == "stop":
        candidate.update(planner_closed=True, status="terminated", planner_reason=decision["reason"])
    elif action in {"continue", "analysis"}:
        if stage == 1:
            candidate["scientific_decision_stage"] = 2
        else:
            await final_validation(runner, candidate, decision["reason"])
    else:
        state = runner.checkpoint.data["autonomous"]
        if not decision.get("sample_recorded"):
            state["used_samples"] += decision["samples"]
            decision["sample_recorded"] = True
            runner.save_candidates()
        if action in {"rfd4_partial", "proteinmpnn"}:
            batch = await optimize_native(runner, decision, index)
        elif action == "torchopt":
            batch = await runner.optimize(decision, index)
        elif action == "partial_flow":
            batch = await runner.generate(
                source_requests={"ppiflow": decision["request"]}, round_index=index, parent_id=candidate["candidate_id"]
            )
        else:
            raise ValueError("unsupported two-stage planner action")
        if not batch:
            raise ValueError("optimization did not produce a candidate")
        for child in batch:
            child["mutation_constraints"] = decision["mutation_constraints"]
            child["scientific_decision_stage"] = 2
            child["scientific_lineage_id"] = candidate.get("scientific_lineage_id", candidate["candidate_id"])
            runner.save_candidates()
            if stage == 2:
                await final_validation(runner, child, decision["reason"])
        candidate.update(planner_closed=True, status="superseded")
    runner.save_candidates()


async def run_two_stage(runner):
    state = runner.checkpoint.data.setdefault(
        "autonomous",
        {
            "decisions": [],
            "used_samples": sum(sample_count(s, g.request) for s, g in runner.config.generators.items()),
            "repaired": [],
        },
    )
    candidates = runner.checkpoint.data["candidates"]
    state["planner_layout"] = "two_stage"
    if state.get("stop_reason"):
        return
    for candidate in candidates:
        candidate.setdefault("scientific_decision_stage", 1)
        candidate.setdefault("scientific_lineage_id", candidate["candidate_id"])
    index = 0
    while True:
        index += 1
        decisions = state["decisions"]
        decision = decisions[index - 1] if len(decisions) >= index else None
        if decision and decision["state"] == "completed":
            continue
        if decision is None:
            active = active_candidates(candidates)
            if not active:
                state["stop_reason"] = "two_stage_completed"
                runner.save_candidates()
                break
            # Complete first-stage decisions across the initial batch before
            # launching second-stage evaluation and planning.
            stage = min(c["scientific_decision_stage"] for c in active)
            current = [c for c in active if c["scientific_decision_stage"] == stage]
            for candidate in current:
                await evaluate_stage(runner, candidate, stage)
            options = stage_options(runner, current, stage)
            fallback = None
            try:
                context = context_for_runner(runner, options, index, evaluation_stage=stage)
                choice = await asyncio.wait_for(
                    planner.planner_choice(context, model=runner.config.autonomous.planner_model),
                    runner.config.autonomous.decision_timeout,
                )
                if choice.option_id not in {o["option_id"] for o in context["options"]}:
                    raise ValueError("planner selected an option outside the visible stage menu")
                option = options[choice.option_id]
                if option["action"] in OPTIMIZATION_ACTIONS:
                    parent = next(c for c in current if c["candidate_id"] == option["parent_id"])
                    option = planner.prepare_optimization(runner, parent, option, choice, index)
                elif choice.selection is not None:
                    raise ValueError("residue selection is valid only for optimization")
            except Exception as exc:
                fallback = f"{type(exc).__name__}: {exc}"
                state["last_planner_failure"] = {"round": index, "scientific_decision_stage": stage, "error": fallback}
                runner.save_candidates()
                choice = planner.fallback_choice(runner, options)
                option = options[choice.option_id]
                if option["action"] in OPTIMIZATION_ACTIONS:
                    parent = next(c for c in current if c["candidate_id"] == option["parent_id"])
                    option = planner.prepare_optimization(runner, parent, option, choice, index)
            decision = {
                **option,
                "round": index,
                "scientific_decision_stage": stage,
                "policy": "rules_fallback" if fallback else "planner",
                "reason": choice.reason,
                "fallback": fallback,
                "state": "planned",
                "sample_recorded": False,
            }
            decisions.append(decision)
            runner.save_candidates()
        try:
            await execute_stage_decision(runner, decision, index)
        except Exception as exc:
            decision["execution_error"] = str(exc)
            runner.save_candidates()
            raise
        decision.pop("execution_error", None)
        decision["state"] = "completed"
        runner.save_candidates()
    write_json(runner.root / "decisions.json", state)
