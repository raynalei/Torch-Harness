"""Binder InterfaceAnalyzerMover dG_separated without an HTTP Rosetta service.

The score is computed on the generated complex using a local PyRosetta worker.
It is reported as diagnostic metadata; ranking remains selected scorer iPTM.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def interface_spec(binder_chains: list[str], target_chains: list[str]) -> str:
    """Produce Rosetta InterfaceAnalyzer chain-group spec (e.g. B_A)."""
    if not binder_chains or not target_chains:
        raise ValueError("Both binder and target chain groups must be non-empty")
    combined = binder_chains + target_chains
    if any(len(chain) != 1 or not chain.isalnum() for chain in combined) or len(set(combined)) != len(combined):
        raise ValueError("InterfaceAnalyzer expects distinct single-character chain IDs")
    return "".join(binder_chains) + "_" + "".join(target_chains)



def make_interface_analyzer(pyrosetta, spec: str):
    """Create InterfaceAnalyzerMover across PyRosetta 4 binding versions.

    Older bindings reject string constructors and string set_interface.
    Use core.pose.DockingPartners explicitly when available; never substitute
    a default jump number, which may score the wrong interface.
    """
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    partners_type = getattr(pyrosetta.rosetta.core.pose, "DockingPartners", None)
    errors = []
    iam = InterfaceAnalyzerMover()
    if partners_type is not None:
        try:
            partners = partners_type.docking_partners_from_string(spec)
            iam.set_interface(partners)
            return iam
        except (TypeError, AttributeError, ValueError) as exc:
            errors.append(f"DockingPartners: {exc}")
    # Recent PyRosetta builds offer a convenience string setter, but PyRosetta
    # 4 (2026.29) may expose only the DockingPartners overload.
    try:
        iam.set_interface(spec)
        return iam
    except (TypeError, AttributeError, ValueError) as exc:
        errors.append(f"string setter: {exc}")
    raise RuntimeError(
        f"PyRosetta InterfaceAnalyzer cannot configure interface {spec!r}; "
        + "; ".join(errors)
    )


def evaluate(request: dict) -> dict:
    import pyrosetta
    source = Path(request["input_pdb"]).resolve(strict=True)
    pyrosetta.init(request.get("init_options", "-ignore_zero_occupancy false"))
    pose = pyrosetta.pose_from_pdb(str(source))
    spec = interface_spec(request["binder_chains"], request["target_chains"])
    iam = make_interface_analyzer(pyrosetta, spec)
    iam.set_scorefunction(pyrosetta.get_fa_scorefxn())
    iam.set_pack_input(bool(request.get("pack_input", False)))
    iam.set_pack_separated(bool(request.get("pack_separated", True)))
    iam.apply(pose)
    dg = float(iam.get_interface_dG())
    if not math.isfinite(dg):
        raise ValueError("PyRosetta InterfaceAnalyzer produced non-finite dG_separated")
    return {
        "dG_separated": dg,
        "interface": spec,
        "pack_input": bool(request.get("pack_input", False)),
        "pack_separated": bool(request.get("pack_separated", True)),
        "backend": "pyrosetta_interface_analyzer",
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: pyrosetta_interface.py <request.json>")
    request = json.loads(Path(sys.argv[1]).read_text())
    result = evaluate(request)
    dest = Path(request["result_path"])
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(dest)


if __name__ == "__main__":
    main()
