"""Persistent static defaults, separate from each experiment's target inputs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from ..core.model_requests import TASK_DESIGN_FIELDS, ppiflow_defaults_for_mode, validate_request_defaults
from .config import InitialRoundConfig

SETTINGS_KEYS = ("workflow_defaults", "design_profiles", "target_adapters", "deployment_config", "public_lookup_url")
DESIGN_MODES = {"nanobody": "vhh", "antibody": "antibody", "binder": "minibinder"}
BUILTIN_NATIVE_MODELS = frozenset({"rfd4", "proteinmpnn"})


def merge_defaults(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key not in {"chains", "refold_map", "service"}
            and isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = merge_defaults(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def deployment_settings(settings):
    """Ignore legacy persisted experimental choices for new tasks."""
    settings = copy.deepcopy(settings)
    for profile in [settings.get("workflow_defaults", {}), *settings.get("design_profiles", {}).values()]:
        for generator in profile.get("generators", {}).values():
            generator.pop("partial_flow", None)
            for field in TASK_DESIGN_FIELDS:
                generator.get("request", {}).pop(field, None)
    for model in settings.get("model_services", {}).values():
        for field in TASK_DESIGN_FIELDS:
            model.get("request_defaults", {}).pop(field, None)
    return settings


def workflow_values(value, settings):
    settings = deployment_settings(settings)
    value = copy.deepcopy(value)
    evaluation = value.get("evaluation", {})
    dockq = evaluation.get("dockq_python")
    if dockq is None or (isinstance(dockq, str) and not dockq.strip()):
        evaluation.pop("dockq_python", None)
    mode = value.get("generators", {}).get("ppiflow", {}).get("request", {}).get("mode")
    design = value.get("design_type") or DESIGN_MODES.get(mode)
    base = merge_defaults(settings.get("workflow_defaults", {}), settings.get("design_profiles", {}).get(design, {}))
    flow = base.get("generators", {}).get("ppiflow")
    if flow and "request" in flow:
        flow["request"] = ppiflow_defaults_for_mode(flow["request"], mode or flow["request"].get("mode"))
    result = merge_defaults(base, value)
    if "generators" in value:
        result["generators"] = {
            key: merge_defaults(base.get("generators", {}).get(key, {}), generator)
            for key, generator in value["generators"].items()
        }
    for kind in ("af3score", "torchscore", "torchfold", "rosetta", "torchopt", "rfd4", "proteinmpnn"):
        if kind in value.get("evaluation", {}):
            result["evaluation"][kind] = copy.deepcopy(value["evaluation"][kind])
    result.pop("design_type", None)
    return result


def validate_settings(settings):
    if settings.get("deployment_config") and not Path(settings["deployment_config"]).is_absolute():
        raise ValueError("deployment_config must be an absolute YAML path")
    if settings.get("public_lookup_url") is not None:
        from ..scientific_agent.backends.public_lookup import validate_lookup_url

        validate_lookup_url(settings["public_lookup_url"])
    allowed = set(InitialRoundConfig.model_fields) | {"model_aliases"}
    for profile in [settings.get("workflow_defaults", {}), *settings.get("design_profiles", {}).values()]:
        if not isinstance(profile, dict) or set(profile) - allowed:
            raise ValueError("workflow defaults contain unknown configuration fields")
        if "name" in profile:
            raise ValueError("experiment names are per-task inputs, not persistent defaults")
        aliases = profile.get("model_aliases", {})
        if not isinstance(aliases, dict) or any(
            kind
            not in {
                "ppiflow",
                "torchcraft",
                "af3score",
                "torchscore",
                "torchfold",
                "torchopt",
                "rosetta",
                "rfd4",
                "proteinmpnn",
            }
            or not isinstance(alias, str)
            or not alias
            for kind, alias in aliases.items()
        ):
            raise ValueError("model_aliases must map supported model kinds to registration names")
        for kind, generator in profile.get("generators", {}).items():
            validate_request_defaults(kind, generator.get("request", {}))
    if set(settings.get("design_profiles", {})) - {"vhh", "antibody", "minibinder"}:
        raise ValueError("design_profiles keys must be vhh, antibody or minibinder")
    adapters = settings.get("target_adapters", {})
    if set(adapters) - {"torchcraft"}:
        raise ValueError("unsupported target adapter")
    if "torchcraft" in adapters:
        from .task_inputs import TorchCraftAdapter

        TorchCraftAdapter.model_validate(adapters["torchcraft"])


# Scientific shape/chain requirements are not model tuning defaults.
SCIENTIFIC_GENERATOR_FIELDS = frozenset(
    {
        "mode",
        "antigen_chain",
        "target_chain",
        "binder_chain",
        "heavy_chain",
        "light_chain",
        "cdr_length",
        "samples_min_length",
        "samples_max_length",
        "design_length",
    }
)


def server_default_settings(settings):
    settings = deployment_settings(settings)
    for model in settings.get("model_services", {}).values():
        model["request_defaults"] = {
            key: value for key, value in model.get("request_defaults", {}).items() if key in SCIENTIFIC_GENERATOR_FIELDS
        }
    for profile in [settings.get("workflow_defaults", {}), *settings.get("design_profiles", {}).values()]:
        for generator in profile.get("generators", {}).values():
            generator["request"] = {
                key: value for key, value in generator.get("request", {}).items() if key in SCIENTIFIC_GENERATOR_FIELDS
            }
        for key in list(profile.get("evaluation", {})):
            if key.endswith("_options"):
                profile["evaluation"].pop(key)
    return settings


DEFAULT_DEPLOYMENT_CONFIG = Path(__file__).with_name("cluster_deployment.yaml")
if not DEFAULT_DEPLOYMENT_CONFIG.is_file():
    # Editable source checkout; wheel builds copy this exact file beside defaults.py.
    DEFAULT_DEPLOYMENT_CONFIG = Path(__file__).resolve().parents[4] / "docs/examples/cluster_deployment.yaml"


def load_deployment_defaults(settings):
    """Use only the dedicated deployment YAML; never inspect historical tasks."""
    path = Path(settings.get("deployment_config") or DEFAULT_DEPLOYMENT_CONFIG).expanduser()
    if not path.is_file():
        raise ValueError(f"deployment configuration not found: {path}")
    defaults = yaml.safe_load(path.read_text())
    if not isinstance(defaults, dict) or set(defaults) - {"model_services", *SETTINGS_KEYS}:
        raise ValueError("deployment YAML must contain deployment sections, not a historical task run_config")
    settings = copy.deepcopy(settings)
    # Retired API deployment defaults must not restart the Rosetta API for new tasks.
    legacy_rosetta = settings.get("workflow_defaults", {}).get("evaluation", {}).get("rosetta")
    if isinstance(legacy_rosetta, dict) and "backend" not in legacy_rosetta:
        settings["workflow_defaults"]["evaluation"].pop("rosetta")
    result = merge_defaults(defaults, settings)
    result["deployment_config"] = str(path.resolve())
    # Explicit registrations take precedence over deployment entries.
    for key in ("model_services", "target_adapters"):
        if settings.get(key):
            result[key] = copy.deepcopy(settings[key])
    models = result.setdefault("model_services", {})
    configured_kinds = {model.get("kind") for model in models.values()}
    if BUILTIN_NATIVE_MODELS - configured_kinds:
        bundled = (
            defaults
            if path.resolve() == DEFAULT_DEPLOYMENT_CONFIG.resolve()
            else yaml.safe_load(DEFAULT_DEPLOYMENT_CONFIG.read_text())
        )
        for source in (defaults, bundled):
            for alias, model in source.get("model_services", {}).items():
                kind = model.get("kind")
                if kind not in BUILTIN_NATIVE_MODELS or kind in configured_kinds:
                    continue
                if alias in models:
                    raise ValueError(f"model alias {alias!r} conflicts with the automatic {kind} registration")
                models[alias] = copy.deepcopy(model)
                configured_kinds.add(kind)
    for name, profile in settings.get("design_profiles", {}).items():
        if "generators" in profile:
            result["design_profiles"][name]["generators"] = copy.deepcopy(profile["generators"])
    return result
