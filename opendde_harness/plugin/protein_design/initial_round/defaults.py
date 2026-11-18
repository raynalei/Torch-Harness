"""Persistent static defaults, separate from each experiment's target inputs."""

from __future__ import annotations

import copy

from ..core.model_requests import TASK_DESIGN_FIELDS, validate_request_defaults
from .config import InitialRoundConfig

SETTINGS_KEYS = ("workflow_defaults", "design_profiles", "target_adapters")
DESIGN_MODES = {"nanobody": "vhh", "antibody": "antibody", "binder": "minibinder"}


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
            for field in TASK_DESIGN_FIELDS:
                generator.get("request", {}).pop(field, None)
    for model in settings.get("model_services", {}).values():
        for field in TASK_DESIGN_FIELDS:
            model.get("request_defaults", {}).pop(field, None)
    return settings


def workflow_values(value, settings):
    settings = deployment_settings(settings)
    mode = value.get("generators", {}).get("ppiflow", {}).get("request", {}).get("mode")
    design = value.get("design_type") or DESIGN_MODES.get(mode)
    base = merge_defaults(settings.get("workflow_defaults", {}), settings.get("design_profiles", {}).get(design, {}))
    result = merge_defaults(base, value)
    if "generators" in value:
        result["generators"] = {
            key: merge_defaults(base.get("generators", {}).get(key, {}), generator)
            for key, generator in value["generators"].items()
        }
    for kind in ("af3score", "torchscore", "torchfold", "rosetta", "torchopt"):
        if kind in value.get("evaluation", {}):
            result["evaluation"][kind] = copy.deepcopy(value["evaluation"][kind])
    result.pop("design_type", None)
    return result


def validate_settings(settings):
    allowed = set(InitialRoundConfig.model_fields) | {"model_aliases"}
    for profile in [settings.get("workflow_defaults", {}), *settings.get("design_profiles", {}).values()]:
        if not isinstance(profile, dict) or set(profile) - allowed:
            raise ValueError("workflow defaults contain unknown configuration fields")
        if "name" in profile:
            raise ValueError("experiment names are per-task inputs, not persistent defaults")
        aliases = profile.get("model_aliases", {})
        if not isinstance(aliases, dict) or any(
            kind not in {"ppiflow", "torchcraft", "af3score", "torchscore", "torchfold", "torchopt", "rosetta"}
            or not isinstance(alias, str)
            or not alias
            for kind, alias in aliases.items()
        ):
            raise ValueError("model_aliases must map supported model kinds to registration names")
        aliases = profile.get("model_aliases", {})
        kinds = {"ppiflow", "torchcraft", "af3score", "torchscore", "torchopt", "torchfold", "rosetta"}
        if (
            not isinstance(aliases, dict)
            or set(aliases) - kinds
            or any(not isinstance(alias, str) or not alias for alias in aliases.values())
        ):
            raise ValueError("model_aliases must map known model kinds to nonempty registration names")
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
