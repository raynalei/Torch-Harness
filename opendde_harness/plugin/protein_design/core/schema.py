"""Strict schema base for protein-design configuration and requests."""

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    # Scientific runs must fail on misspelled or stale fields rather than
    # silently running with a default configuration.
    model_config = ConfigDict(extra="forbid")
