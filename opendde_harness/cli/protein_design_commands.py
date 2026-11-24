"""CLI commands for validating, launching and inspecting protein-design tasks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console

from opendde_harness.cli.protein_model_commands import models_app

protein_design_app = typer.Typer(
    help="Run and inspect Initial Round workflows and registered model requests.",
    no_args_is_help=True,
)
error_console = Console(stderr=True)
protein_design_app.add_typer(models_app, name="models")


@protein_design_app.command("initial-round")
def initial_round(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, resolve_path=True),
    check: bool = typer.Option(False, "--check", help="Validate shared inputs and API health without submitting jobs."),
    pdb: Path | None = typer.Option(
        None,
        "--pdb",
        exists=True,
        dir_okay=False,
        help="Mounted input PDB; overrides the profile's generator input path.",
    ),
    name: str | None = typer.Option(
        None, "--name", help="Experiment name; use a new name for a different input/config."
    ),
    detach: bool = typer.Option(
        False, "--detach", help="Launch a background worker that polls and advances stages automatically."
    ),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Retry confirmed failed steps in the same run; reuse successful results."
    ),
    attach_job: list[str] | None = typer.Option(
        None, "--attach-job", help="Reconcile an uncertain submission: checkpoint-job-key=remote-job-id. Repeatable."
    ),
) -> None:
    """Run or resume a PPIFlow initial round, optionally with compatible TorchCraft generation."""
    from opendde_harness.plugin.protein_design.initial_round.automation import prepare_run, start_round
    from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig
    from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner

    try:
        attachments = {}
        for item in attach_job or []:
            key, separator, value = item.partition("=")
            if not separator or not key or not value or key in attachments:
                raise ValueError("--attach-job requires distinct checkpoint-job-key=remote-job-id entries")
            attachments[key] = value
        workflow = (
            prepare_run(config, pdb_path=pdb, name=name)
            if pdb is not None or name is not None
            else InitialRoundConfig.load(config)
        )
        if detach:
            if check:
                raise ValueError("--detach cannot be combined with --check")
            options = {"attach_jobs": attachments} if attachments else {}
            if retry_failed:
                options["retry_failed"] = True
            result = start_round(workflow, **options)
        else:
            result = asyncio.run(
                InitialRoundRunner(
                    workflow, attach_jobs=attachments, **({"retry_failed": True} if retry_failed else {})
                ).run(check_only=check)
            )
    except Exception as exc:
        error_console.print(f"Initial round failed: {exc}", markup=False)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result.get("status") == "partial":
        raise typer.Exit(2)


@protein_design_app.command("initial-round-regions")
def initial_round_regions(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, resolve_path=True),
    question_id: str = typer.Option(..., "--question-id"),
    cdr_position: str = typer.Option(..., "--cdr-position"),
    fixed_positions: str | None = typer.Option(None, "--fixed-positions"),
):
    """Supply a user CDR answer for a paused workflow and resume saved steps."""
    from opendde_harness.plugin.protein_design.initial_round.automation import resume_round
    from opendde_harness.plugin.protein_design.initial_round.region_input import supply_regions

    try:
        supply_regions(run_dir, question_id, cdr_position, fixed_positions)
        result = resume_round(run_dir)
    except Exception as exc:
        error_console.print(f"Region resolution failed: {exc}", markup=False)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, ensure_ascii=False, allow_nan=False))


@protein_design_app.command("initial-round-resume")
def initial_round_resume(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, resolve_path=True),
):
    """Resume the saved workflow and retry only confirmed failed model steps."""
    from opendde_harness.plugin.protein_design.initial_round.automation import resume_round

    try:
        result = resume_round(run_dir)
    except Exception as exc:
        error_console.print(f"Initial round recovery failed: {exc}", markup=False)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, ensure_ascii=False, allow_nan=False))


@protein_design_app.command("initial-round-status")
def initial_round_status(run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False)):
    """Read saved background pipeline progress without polling the remote APIs."""
    from opendde_harness.plugin.protein_design.initial_round.automation import round_status

    try:
        typer.echo(json.dumps(round_status(run_dir), ensure_ascii=False))
    except Exception as exc:
        typer.echo(f"Initial round status failed: {exc}", err=True)
        raise typer.Exit(1) from exc


@protein_design_app.command("initial-round-list")
def initial_round_list(
    work_dir: Path | None = typer.Option(None, "--work-dir", exists=True, file_okay=False, resolve_path=True),
) -> None:
    """Discover saved runs in the instance index or a shared work directory."""
    from opendde_harness.plugin.protein_design.initial_round.automation import list_rounds

    try:
        typer.echo(json.dumps(list_rounds(work_dir), ensure_ascii=False))
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(f"Initial round listing failed: {exc}", err=True)
        raise typer.Exit(1) from exc
