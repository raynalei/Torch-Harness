"""Background worker entry point; no model/LLM requests are made here."""

import asyncio
import json
import sys
from pathlib import Path

from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig
from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner
from opendde_harness.plugin.protein_design.initial_round.storage import write_json


def run(root: Path):
    status = root / "worker-status.json"
    write_json(status, {"state": "running"})
    try:
        config = InitialRoundConfig.model_validate(json.loads((root / "worker-config.json").read_text()))
        result = asyncio.run(InitialRoundRunner(config).run())
        write_json(status, {"state": "partial" if result["status"] == "partial" else "completed", "summary": result})
    except Exception as exc:
        write_json(status, {"state": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    run(Path(sys.argv[1]))
