"""Torch-Harness workflows over registered model APIs and native evidence functions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from opendde_harness.cli.protein_model_commands import models_app
from opendde_harness.plugin.protein_design.initial_round.automation import (
    list_rounds,
    resume_round,
    round_status,
    start_round,
)
from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner
from opendde_harness.plugin.protein_design.tools.workflow import EvidenceRequest, evidence, prepare_workflow

protein_design_app = typer.Typer(
    help="Run Torch-Harness model workflows and native scientific analysis.", no_args_is_help=True
)
protein_design_app.add_typer(models_app, name="models")


def _output(function, *args, **kwargs):
    try:
        value = function(*args, **kwargs)
        typer.echo(json.dumps(value, ensure_ascii=False, allow_nan=False))
        return value
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@protein_design_app.command("start")
def start_task(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    check: bool = typer.Option(False, "--check"),
    detach: bool = typer.Option(True, "--detach/--foreground"),
    retry_failed: bool = typer.Option(False, "--retry-failed"),
    attach_job: list[str] | None = typer.Option(None, "--attach-job"),
    pdb: Path | None = typer.Option(None, "--pdb", exists=True, dir_okay=False),
    name: str | None = typer.Option(None, "--name"),
):
    """Run the configured autonomous model workflow."""

    def start():
        workflow = prepare_workflow(config, pdb_path=pdb, name=name)
        attachments = {}
        for item in attach_job or []:
            key, separator, remote_id = item.partition("=")
            if not separator or not key or not remote_id or key in attachments:
                raise ValueError("--attach-job requires distinct checkpoint-job-key=remote-id entries")
            attachments[key] = remote_id
        if check or not detach:
            return asyncio.run(
                InitialRoundRunner(workflow, retry_failed=retry_failed, attach_jobs=attachments).run(check_only=check)
            )
        return start_round(workflow, retry_failed=retry_failed, **({"attach_jobs": attachments} if attachments else {}))

    result = _output(start)
    if result.get("status") == "partial":
        raise typer.Exit(2)


@protein_design_app.command("validate")
def validate_config(config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False)):
    _output(lambda: {"valid": True, "workflow": prepare_workflow(config).model_dump(mode="json")})


@protein_design_app.command("status")
def task_status(run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False)):
    _output(round_status, run_dir)


@protein_design_app.command("resume")
def resume(run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False)):
    _output(resume_round, run_dir)


@protein_design_app.command("list")
def workflows():
    _output(list_rounds)


@protein_design_app.command("candidates")
def candidates(run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False)):
    _output(lambda: json.loads((run_dir / "candidates.json").read_text()))


@protein_design_app.command("analyze")
def analyze(request: Path = typer.Option(..., "--request", exists=True, dir_okay=False)):
    """Evaluate JSON native-analysis inputs without model execution or an RPC service."""
    _output(lambda: evidence(EvidenceRequest.model_validate_json(request.read_text())))


@protein_design_app.command("context")
def show_context():
    from opendde_harness.cli._helpers import load_runtime_config

    config = load_runtime_config()
    plugin = config.plugins.config.get("protein-design", {})
    _output(
        lambda: {
            "compute_service_required": False,
            "models": list(plugin.get("model_services", {})),
            "workflow_config": plugin.get("workflow_config"),
            "analysis": ["developability", "structure", "epitope", "evolution", "mutations"],
        }
    )


@protein_design_app.command("regions")
def supply_regions(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
    question_id: str = typer.Option(..., "--question-id"),
    cdr_position: str | None = typer.Option(None, "--cdr-position"),
    fixed_positions: str | None = typer.Option(None, "--fixed-positions"),
):
    """Supply requested redesign regions and resume the saved workflow."""
    from opendde_harness.plugin.protein_design.initial_round.region_input import supply_regions as supply

    def update():
        supply(run_dir, question_id, cdr_position, fixed_positions)
        return resume_round(run_dir)

    _output(update)
