"""Validate planner-selected residue masks without inferring scientific scores."""

from __future__ import annotations

import json

from pydantic import Field, StrictInt

from opendde_harness.agent.tools.base import Tool, ToolResult

from ..initial_round.config import RoundModel
from .backends.developability import validate_sequences


class MutationSelection(RoundModel):
    fixed_positions: dict[str, list[StrictInt]] = Field(default_factory=dict)
    mask_positions: dict[str, list[StrictInt]] = Field(default_factory=dict)


class SelectorRequest(RoundModel):
    sequences: dict[str, str]
    selection: MutationSelection
    required_fixed: dict[str, list[StrictInt]] = Field(default_factory=dict)
    allowed_positions: dict[str, list[StrictInt]] | None = None


def select_mutations(sequences, selection, *, required_fixed=None, allowed_positions=None):
    """Partition observed chain sequence indices; never relax inherited constraints."""
    sequences = validate_sequences(sequences)
    selection = MutationSelection.model_validate(selection)
    universe = {c: set(range(1, len(s) + 1)) for c, s in sequences.items()}

    def validated(value):
        if set(value) - set(universe):
            raise ValueError("mutation selections must refer only to binder chains")
        result = {}
        for chain, positions in value.items():
            if (
                any(type(p) is not int for p in positions)
                or len(positions) != len(set(positions))
                or not set(positions) <= universe[chain]
            ):
                raise ValueError("mutation positions must be distinct one-based observed sequence indices")
            result[chain] = set(positions)
        return result

    required = validated(required_fixed or {})
    allowed = validated(allowed_positions) if allowed_positions is not None else universe
    fixed = validated(selection.fixed_positions)
    selected = validated(selection.mask_positions)
    if not any(fixed.values()) and not any(selected.values()):
        raise ValueError("select explicit fixed_positions or mask_positions")
    masks, locked = {}, {}
    for chain, positions in universe.items():
        frozen = required.get(chain, set()) | fixed.get(chain, set())
        mutable = allowed.get(chain, set()) - frozen
        if selection.mask_positions:
            mask = selected.get(chain, set())
            if not mask <= mutable:
                raise ValueError("mask changes a fixed residue or exceeds the authorized mutable region")
        else:
            mask = mutable
        if selected.get(chain, set()) & fixed.get(chain, set()):
            raise ValueError("fixed_positions and mask_positions overlap")
        masks[chain] = sorted(mask)
        locked[chain] = sorted(positions - mask)
    if not any(masks.values()):
        raise ValueError("mutation selection contains no mutable residues")
    return {
        "method": "planner_validated_residue_partition",
        "position_semantics": "one-based observed binder sequence",
        "fixed_positions": locked,
        "mask_positions": masks,
    }


class MutationSelectorTool(Tool):
    name = "protein_design_mutation_selector"
    description = "Validate explicit binder fixed points or mutation masks using one-based observed sequence positions, preserving required_fixed and allowed_positions. This selects a residue partition, not mutations or affinity scores. The workflow independently enforces saved task constraints."

    @property
    def parameters(self):
        return SelectorRequest.model_json_schema()

    async def execute(self, **kwargs):
        try:
            request = SelectorRequest.model_validate(kwargs)
            result = select_mutations(
                request.sequences,
                request.selection,
                required_fixed=request.required_fixed,
                allowed_positions=request.allowed_positions,
            )
            return ToolResult(json.dumps(result), "Validated mutation selection", retryable=False)
        except Exception as exc:
            return ToolResult(json.dumps({"error": str(exc)}), "Mutation selection failed", retryable=False)


def make_mutation_selector_tool(ctx):
    return MutationSelectorTool()
