"""Register external model endpoints in the actual Harness configuration."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import typer

from opendde_harness.config.loader import get_config_path
from opendde_harness.plugin.protein_design.core.model_registry import ProteinModelRegistry
from opendde_harness.plugin.protein_design.initial_round.defaults import (
    SETTINGS_KEYS,
    merge_defaults,
    validate_settings,
)
from opendde_harness.plugin.protein_design.initial_round.storage import write_json
from opendde_harness.utils.portable_lock import file_lock

models_app = typer.Typer(help="Register and inspect external protein-model services.", no_args_is_help=True)


def _object_file(path: Path):
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object; existing file was left unchanged")
    return value


def _section(data, *, create=False):
    current = data
    for key in ("plugins", "config", "protein-design"):
        if key not in current:
            if not create:
                return {}
            current[key] = {}
        if not isinstance(current[key], dict):
            raise ValueError(f"configuration section {key} must be an object")
        current = current[key]
    return current


def local_defaults():
    ports = {
        "ppiflow": 8010,
        "torchcraft": 8020,
        "rosetta": 8030,
        "af3score": 8040,
        "torchfold": 8050,
        "torchscore": 8070,
        "torchopt": 8060,
    }
    return {
        kind: {
            "kind": kind,
            "service": {
                "url": f"http://slurm-api.invalid:{port}",
                "health_path": "/openapi.json" if kind in {"ppiflow", "torchcraft"} else "/health",
                **({"status_field": "state"} if kind == "torchopt" else {}),
            },
            "max_samples": 20,
        }
        for kind, port in ports.items()
    }


def register_services(
    path: Path, registrations: dict, *, replace_existing=False, fill_missing_kinds=False, workflow_settings=None
):
    # Validate every alias/profile before touching the user's provider settings.
    ProteinModelRegistry(registrations, path.parent / "unused")
    if workflow_settings is not None:
        validate_settings(workflow_settings)
    with file_lock(path.parent / ".lock" / (path.name + ".lock")):
        original = _object_file(path)
        data = json.loads(json.dumps(original))
        section = _section(data, create=True)
        disabled = data["plugins"].get("disabled", [])
        if not isinstance(disabled, list) or any(not isinstance(item, str) for item in disabled):
            raise ValueError("plugins.disabled must be a list of plugin names")
        existing = section.get("model_services", {})
        if not isinstance(existing, dict):
            raise ValueError("model_services must be an object")
        existing_registry = ProteinModelRegistry(existing, path.parent / "unused")
        kinds = {value.kind for value in existing_registry.models.values()}
        added = []
        for alias, value in registrations.items():
            if fill_missing_kinds and value["kind"] in kinds:
                continue
            if alias in existing and existing[alias] != value and not replace_existing:
                raise ValueError(
                    f"alias {alias} already has a different registration; use --replace-existing only when intended"
                )
            if existing.get(alias) != value:
                existing[alias] = value
                added.append(alias)
                kinds.add(value["kind"])
        section["model_services"] = existing
        changed_settings = []
        for key, value in (workflow_settings or {}).items():
            if key not in SETTINGS_KEYS:
                raise ValueError("unsupported persistent workflow setting")
            merged = merge_defaults(section.get(key, {}), value)
            if key in section and merge_defaults(value, section[key]) != merged and not replace_existing:
                raise ValueError("persistent defaults conflict; use --replace-existing when intended")
            if section.get(key) != merged:
                section[key] = merged
                changed_settings.append(key)
        validate_settings({key: section[key] for key in SETTINGS_KEYS if key in section})
        backup = None
        if data != original:
            if path.exists():
                backup = path.with_name(f"{path.name}.models-backup-{uuid.uuid4().hex}.json")
                write_json(backup, original)
            write_json(path, data)
        return {
            "config_path": str(path),
            "registered_aliases": sorted(existing),
            "changed_aliases": sorted(added),
            "backup_path": str(backup) if backup else None,
            "plugin_enabled": "protein-design" not in disabled,
            "restart_required": bool(added or changed_settings),
            "changed_settings": changed_settings,
            "inference_verified": False,
        }


@models_app.command("register")
def register_models(
    local_defaults_flag: bool = typer.Option(
        False,
        "--local-defaults",
        help="Register example service profiles (TorchOpt 8060, TorchScore 8070); verify deployment addresses first.",
    ),
    services_file: Path | None = typer.Option(
        None,
        "--services-file",
        exists=True,
        dir_okay=False,
        help="A model_services map, {model_services: ...}, or full plugins.config.protein-design fragment.",
    ),
    opendde_config: Path | None = typer.Option(
        None,
        "--opendde-config",
        dir_okay=False,
        help="Actual Harness config.json; defaults to the active/default configuration path.",
    ),
    replace_existing: bool = typer.Option(
        False, "--replace-existing", help="Allow explicitly imported registrations to replace existing aliases."
    ),
):
    """Persist registrations while preserving Qwen/provider and unrelated settings."""
    try:
        if local_defaults_flag == (services_file is not None):
            raise ValueError("choose exactly one of --local-defaults or --services-file")
        path = (opendde_config or get_config_path()).expanduser().resolve()
        if services_file:
            values = _object_file(services_file)
            section = _section(values) if "plugins" in values else values
            settings = {key: section[key] for key in SETTINGS_KEYS if key in section}
            registrations = section.get("model_services", {} if settings else section)
            if not isinstance(registrations, dict) or not registrations and not settings:
                raise ValueError("services file contains no model registrations or workflow defaults")
        else:
            registrations = local_defaults()
            settings = {}
        result = register_services(
            path,
            registrations,
            replace_existing=replace_existing,
            fill_missing_kinds=local_defaults_flag,
            workflow_settings=settings,
        )
    except Exception as exc:
        # Do not echo raw configuration/credential values in parse errors.
        typer.echo(
            f"Model registration failed ({type(exc).__name__}); configuration was not replaced. Check JSON shape, aliases and service profiles.",
            err=True,
        )
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, ensure_ascii=False))


@models_app.command("list")
def list_models(
    opendde_config: Path | None = typer.Option(None, "--opendde-config", dir_okay=False),
    check_health: bool = typer.Option(False, "--check-health", help="Probe APIs without submitting jobs."),
):
    """Show which actual configuration file supplies registered endpoints."""
    try:
        path = (opendde_config or get_config_path()).expanduser().resolve()
        data = _object_file(path)
        registry = ProteinModelRegistry(
            _section(data).get("model_services", {}),
            path.parent / "unused",
            config_path=path,
            workflow_settings=_section(data),
        )
        value = asyncio.run(registry.catalogue_with_health()) if check_health else registry.catalogue()
        value["plugin_enabled"] = "protein-design" not in data.get("plugins", {}).get("disabled", [])
        value["endpoints"] = {name: item.service.url for name, item in registry.models.items()}
    except Exception as exc:
        typer.echo(
            f"Cannot inspect model registrations ({type(exc).__name__}); check the selected configuration.", err=True
        )
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(value, ensure_ascii=False))


@models_app.command("defaults")
def import_workflow_defaults(
    from_workflow: Path = typer.Option(..., "--from-workflow", exists=True, dir_okay=False),
    design_type: str = typer.Option(..., "--design-type"),
    torchcraft_hotspot_numbering: str | None = typer.Option(None, "--torchcraft-hotspot-numbering"),
    csv_id_column: str = typer.Option("id", "--csv-id-column"),
    csv_sequence_column: str = typer.Option("sequence", "--csv-sequence-column"),
    opendde_config: Path | None = typer.Option(None, "--opendde-config", dir_okay=False),
    replace_existing: bool = typer.Option(False, "--replace-existing"),
):
    """Import fixed settings once from an existing task, removing old target inputs."""
    import yaml

    from opendde_harness.plugin.protein_design.core.model_requests import TARGET_REQUEST_FIELDS, TASK_DESIGN_FIELDS
    from opendde_harness.plugin.protein_design.initial_round.task_inputs import TorchCraftAdapter

    try:
        if design_type not in {"vhh", "antibody", "minibinder"}:
            raise ValueError("design type must be vhh, antibody or minibinder")
        source = yaml.safe_load(from_workflow.read_text())
        path = (opendde_config or get_config_path()).expanduser().resolve()
        existing = _section(_object_file(path))
        defaults = {k: source[k] for k in ("work_dir", "path_mappings", "evaluation", "autonomous") if k in source}
        defaults["autonomous"] = defaults.get("autonomous") or {}
        defaults.setdefault("evaluation", {})["scoring_backend"] = "torchscore"
        generators = json.loads(json.dumps(source["generators"]))
        known = existing.get("model_services", {})
        for kind, generator in generators.items():
            if sum(model.get("kind") == kind for model in known.values()) == 1:
                generator.pop("service", None)
            for key in TARGET_REQUEST_FIELDS | TASK_DESIGN_FIELDS | {"output_dir"}:
                generator["request"].pop(key, None)
            generator["fold_template"] = None
        for kind in ("af3score", "torchscore", "torchfold", "rosetta", "torchopt"):
            if sum(model.get("kind") == kind for model in known.values()) == 1:
                defaults["evaluation"].pop(kind, None)
        settings = {"workflow_defaults": defaults, "design_profiles": {design_type: {"generators": generators}}}
        if "torchcraft" in generators:
            adapter = existing.get("target_adapters", {}).get("torchcraft")
            if torchcraft_hotspot_numbering:
                adapter = TorchCraftAdapter(
                    hotspot_numbering=torchcraft_hotspot_numbering,
                    csv_id_column=csv_id_column,
                    csv_sequence_column=csv_sequence_column,
                ).model_dump()
            if adapter is None:
                raise ValueError("set --torchcraft-hotspot-numbering once; the API numbering must not be guessed")
            settings["target_adapters"] = {"torchcraft": adapter}
        result = register_services(path, {}, workflow_settings=settings, replace_existing=replace_existing)
    except Exception as exc:
        typer.echo(f"Cannot import workflow defaults: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, ensure_ascii=False))
