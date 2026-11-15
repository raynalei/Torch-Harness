"""Durable per-action user approval for refinement experiments."""

import json
from pathlib import Path

from opendde_harness.utils.portable_lock import file_lock

from .storage import fingerprint, write_json


class AwaitingActionApprovalError(RuntimeError):
    def __init__(self, pending):
        self.pending = pending
        super().__init__(pending["question"])


def action_id(root, decision):
    return fingerprint(
        {
            "run_dir": str(Path(root).resolve()),
            **{key: decision[key] for key in ("round", "action", "source", "parent_id", "request", "samples")},
        }
    )


def require_approval(runner, decision):
    identifier = action_id(runner.root, decision)
    if decision.get("approved_id") == identifier:
        return
    decision.update(state="awaiting_approval", approval_id=identifier)
    pending = {
        key: decision[key]
        for key in ("approval_id", "round", "action", "source", "parent_id", "request", "samples", "basis", "reason")
    }
    parent = next(c for c in runner.checkpoint.data["candidates"] if c["candidate_id"] == decision["parent_id"])
    pending["parent_metrics"] = parent["metrics"]
    pending["diagnostics"] = parent.get("diagnostics")
    pending["question"] = (
        f"是否同意第 {decision['round']} 轮对候选 {decision['parent_id']} 执行 {decision['action']}？请核对请求参数和依据。"
    )
    runner.checkpoint.data["pending_approval"] = pending
    runner.save_candidates()
    write_json(runner.root / "pending-approval.json", pending)
    raise AwaitingActionApprovalError(pending)


def approve_action(root: Path, approval_id: str):
    root = root.expanduser().resolve()
    with file_lock(root / "run.lock", blocking=False):
        worker_status = root / "worker-status.json"
        if worker_status.is_file() and json.loads(worker_status.read_text()).get("state") in {"starting", "running"}:
            raise ValueError("worker is still finishing the pause; query status before approving")
        path = root / "checkpoint.json"
        checkpoint = json.loads(path.read_text())
        pending = checkpoint.get("pending_approval")
        if not pending or pending["approval_id"] != approval_id:
            raise ValueError("no matching pending action; inspect current workflow status")
        decisions = checkpoint["autonomous"]["decisions"]
        decision = next((item for item in decisions if item.get("approval_id") == approval_id), None)
        if decision is None or decision["state"] != "awaiting_approval" or action_id(root, decision) != approval_id:
            raise ValueError("pending action changed or was already handled; request fresh user approval")
        decision.update(approved_id=approval_id, state="planned")
        checkpoint.pop("pending_approval", None)
        checkpoint.setdefault("approval_history", []).append({"approval_id": approval_id, "approved": True})
        write_json(path, checkpoint)
        (root / "pending-approval.json").unlink(missing_ok=True)
