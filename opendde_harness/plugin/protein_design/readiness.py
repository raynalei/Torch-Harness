"""Readiness for registered Torch-Harness model workflows."""

from __future__ import annotations

from typing import Any


def models_configured(config: dict[str, Any]) -> bool:
    return bool(config.get("model_services"))


def compute_configured(config: dict[str, Any]) -> bool:
    """Compatibility import only; legacy compute URLs no longer establish readiness."""
    return models_configured(config)
