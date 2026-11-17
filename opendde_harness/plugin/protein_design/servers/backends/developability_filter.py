"""Compatibility function for the native Torch-Harness developability backend."""

from __future__ import annotations

from opendde_harness.plugin.protein_design.scientific_agent.backends.developability import assess_developability


def check_antibody_developability(
    heavy_chain: str,
    name="candidate",
    output_dir=None,
    structure_path="",
    heavy_chain_id="",
    light_chain="",
    light_chain_id="",
):
    if structure_path and (not heavy_chain_id or (light_chain and not light_chain_id)):
        raise ValueError("explicit structural chain IDs are required")
    sequences = {heavy_chain_id or "H": heavy_chain}
    if light_chain:
        if light_chain_id and light_chain_id == heavy_chain_id:
            raise ValueError("heavy and light chains must have distinct IDs")
        sequences[light_chain_id or "L"] = light_chain
    return assess_developability(
        sequences, structure_path=structure_path or None, binder_chains=list(sequences) if structure_path else None
    )
