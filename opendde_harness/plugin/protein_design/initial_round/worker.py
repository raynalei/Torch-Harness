"""Background worker; optional bounded feedback uses the existing chat provider."""

import argparse
import asyncio
import json
from pathlib import Path

from opendde_harness.plugin.protein_design.initial_round.config import InitialRoundConfig
from opendde_harness.plugin.protein_design.initial_round.runner import InitialRoundRunner
from opendde_harness.plugin.protein_design.initial_round.services import SubmissionUncertainError
from opendde_harness.plugin.protein_design.initial_round.storage import write_json


def run(root: Path, *, attach_jobs: dict[str, str] | None = None, retry_failed: bool = False):
    status = root / "worker-status.json"
    write_json(status, {"state": "running"})
    try:
        config = InitialRoundConfig.model_validate(json.loads((root / "worker-config.json").read_text()))
        result = asyncio.run(
            InitialRoundRunner(
                config, attach_jobs=attach_jobs, **({"retry_failed": True} if retry_failed else {})
            ).run()
        )
        write_json(status, {"state": "partial" if result["status"] == "partial" else "completed", "summary": result})
    except SubmissionUncertainError as exc:
        write_json(status, {"state": "submission_uncertain", "error": str(exc)})
        raise
    except Exception as exc:
        write_json(status, {"state": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--attach-jobs", type=json.loads)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    run(args.root, attach_jobs=args.attach_jobs, retry_failed=args.retry_failed)
