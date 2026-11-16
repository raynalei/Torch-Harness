"""Ask the user only when a selected Partial Flow action has unidentifiable CDRs."""

import hashlib
import json
from pathlib import Path

from opendde_harness.utils.portable_lock import file_lock

from .artifacts import structure
from .config import InitialRoundConfig
from .diagnostics import _ca
from .regions import check_region_encoding, selection_positions
from .storage import SharedFiles, fingerprint, write_json


class RegionIdentificationError(ValueError):
    pass


class AwaitingRegionsError(RuntimeError):
    def __init__(self, pending):
        self.pending = pending
        super().__init__(pending["question"])


def request_regions(runner, candidate, decision, reason):
    chains = runner.config.generators[candidate["source"]].chains
    question_id = fingerprint(
        {
            "run_dir": str(runner.root.resolve()),
            "round": decision["round"],
            "candidate_id": candidate["candidate_id"],
            "sha256": candidate["generated_sha256"],
        }
    )
    pending = {
        "question_id": question_id,
        "candidate_id": candidate["candidate_id"],
        "generated_sha256": candidate["generated_sha256"],
        "round": decision["round"],
        "structure": candidate["generated_structure"],
        "binder_chains": chains.binder,
        "reason": reason,
        "metrics": candidate["metrics"],
        "question": f"候选 {candidate['candidate_id']} 已选中 Partial Flow，但无法可靠识别 CDR。请根据该候选 PDB 提供 cdr_position（链名和真实残基编号）；fixed_positions 可省略，由 Python 固定其余 binder 残基。",
    }
    decision["state"] = "awaiting_regions"
    runner.checkpoint.data["pending_regions"] = pending
    runner.save_candidates()
    write_json(runner.root / "pending-regions.json", pending)
    raise AwaitingRegionsError(pending)


def supply_regions(root: Path, question_id: str, cdr_position: str, fixed_positions: str | None = None):
    root = root.expanduser().resolve()
    with file_lock(root / "run.lock", blocking=False):
        status_path = root / "worker-status.json"
        if status_path.is_file() and json.loads(status_path.read_text()).get("state") in {"starting", "running"}:
            raise ValueError("worker is still finishing the pause; query status again")
        path = root / "checkpoint.json"
        checkpoint = json.loads(path.read_text())
        pending = checkpoint.get("pending_regions")
        if not pending or pending["question_id"] != question_id:
            raise ValueError("no matching pending CDR question; inspect the current run status")
        snapshot = root / "worker-config.json"
        if not snapshot.is_file():
            snapshot = root / "run-config.json"
        config = InitialRoundConfig.model_validate(json.loads(snapshot.read_text()))
        candidate = next(c for c in checkpoint["candidates"] if c["candidate_id"] == pending["candidate_id"])
        local = SharedFiles(config.path_mappings).local(candidate["generated_structure"])
        if hashlib.sha256(local.read_bytes()).hexdigest() != pending["generated_sha256"]:
            raise ValueError("candidate structure changed since the CDR question")
        atoms = structure(local)
        chains = config.generators[candidate["source"]].chains
        mutable = selection_positions(cdr_position, atoms, chains.binder)
        fixed = {
            chain: set(range(1, len(_ca(atoms, chain)) + 1)) - mutable.get(chain, set()) for chain in chains.binder
        }
        if fixed_positions is not None:
            supplied = selection_positions(fixed_positions, atoms, chains.binder)
            if any(supplied.get(chain, set()) != positions for chain, positions in fixed.items()):
                raise ValueError("fixed_positions must be exactly the binder complement of cdr_position")
        cdr, frozen = check_region_encoding(atoms, mutable), check_region_encoding(atoms, fixed)
        if not cdr["ok"] or not frozen["ok"]:
            raise ValueError("CDR/fixed selections are empty or cannot be encoded for this PDB")
        answer = {
            "question_id": question_id,
            "generated_sha256": pending["generated_sha256"],
            "cdr_position": cdr["encoded"],
            "fixed_positions": frozen["encoded"],
            "mutable_positions": {chain: sorted(positions) for chain, positions in mutable.items()},
            "method": "user_supplied_after_alignment_failure",
        }
        checkpoint.setdefault("region_resolutions", {})[candidate["candidate_id"]] = answer
        checkpoint.pop("pending_regions", None)
        write_json(path, checkpoint)
        (root / "pending-regions.json").unlink(missing_ok=True)
