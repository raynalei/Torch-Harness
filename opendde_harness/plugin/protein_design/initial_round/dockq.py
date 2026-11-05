"""Isolated DockQ worker: score target/binder interfaces against the design pose."""

from __future__ import annotations

import contextlib
import json
import math
import sys


def evaluate(model: str, native: str, chain_map: dict[str, str], targets: list[str], binders: list[str]) -> dict:
    # DockQ's Python API takes native:model, unlike its CLI's model:native.
    from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

    with contextlib.redirect_stdout(sys.stderr):
        scores, _ = run_on_all_native_interfaces(load_PDB(model), load_PDB(native), chain_map=chain_map)
    interfaces = []
    for pair, metrics in scores.items():
        a, b = pair
        if (a in targets and b in binders) or (b in targets and a in binders):
            value = float(metrics["DockQ"])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid DockQ result")
            interfaces.append({"native_chains": list(pair), "dockq": value})
    if not interfaces:
        raise ValueError("no target/binder reference interface was scored by DockQ")
    return {"dockq": min(row["dockq"] for row in interfaces), "interfaces": interfaces, "reference_kind": "design_pose"}


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        import importlib
        from importlib.metadata import version

        importlib.import_module("DockQ.DockQ")
        print(
            json.dumps({"dockq_available": True, "dockq_version": version("DockQ"), "numpy_version": version("numpy")})
        )
    else:
        args = json.loads(sys.argv[1])
        print(json.dumps(evaluate(**args), allow_nan=False))
