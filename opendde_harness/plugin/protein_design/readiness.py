"""Setup readiness for the protein-design plugin."""

from __future__ import annotations

from typing import Any


def models_configured(config: dict[str, Any]) -> bool:
    """Return whether registered model services or a workflow profile are configured."""
    return bool(config.get("model_services") or config.get("initial_round_config") or config.get("deployment_config"))
