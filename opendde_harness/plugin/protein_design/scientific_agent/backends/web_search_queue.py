"""Network-free, persistent shared-file search queue used by Torch-Harness tools."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

SOURCES = ("rcsb", "uniprot", "pubmed", "sabdab")
DEFAULT_ROOT = "/lustre/grp/cmclab/leiy/harness_web"


def queue_root() -> Path:
    return Path(os.environ.get("HARNESS_WEB_ROOT", DEFAULT_ROOT)).expanduser().resolve()


def _validate(source: str, query: str, limit: int) -> None:
    if source not in SOURCES:
        raise ValueError("source must be rcsb, uniprot, pubmed or sabdab")
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 300:
        raise ValueError("query must be 1-300 characters")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 20:
        raise ValueError("limit must be 1-20")


def _atomic_write(destination: Path, value: dict) -> None:
    temporary = destination.parent / ("." + destination.stem + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def search_queue(*, source: str | None = None, query: str | None = None,
                 limit: int = 10, request_id: str | None = None,
                 wait_seconds: int = 0, root: Path | None = None) -> dict:
    """Submit once, then poll by request_id. Never resubmit on a timeout."""
    if not isinstance(wait_seconds, int) or not 0 <= wait_seconds <= 30:
        raise ValueError("wait_seconds must be 0-30")
    base = queue_root() if root is None else Path(root)
    request_dir, result_dir = base / "requests", base / "results"
    for directory in (request_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if request_id is None:
        if source is None or query is None:
            raise ValueError("source and query are required for a new search")
        _validate(source, query, limit)
        request_id = uuid.uuid4().hex
        _atomic_write(request_dir / (request_id + ".json"),
                      {"request_id": request_id, "source": source,
                       "query": query.strip(), "limit": limit})
    else:
        if not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise ValueError("invalid request_id")
        if source is not None or query is not None:
            raise ValueError("poll by request_id only; omit source and query")
        if not ((request_dir / (request_id + ".json")).exists()
                or (base / "processing" / (request_id + ".json")).exists()
                or (result_dir / (request_id + ".json")).exists()):
            raise ValueError("request_id not found in search queue")
    result_path = result_dir / (request_id + ".json")
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            result = None
        if result is not None:
            if result.get("request_id") != request_id:
                raise ValueError("search queue result request_id mismatch")
            # Keep a bounded model context, including structured source evidence.
            raw = json.dumps(result, ensure_ascii=False)
            if len(raw) > 100_000:
                return {"request_id": request_id, "status": "error",
                        "error": "search result too large; inspect queue artifact instead"}
            return result
        if time.monotonic() >= deadline:
            return {"request_id": request_id, "status": "pending",
                    "message": "Worker has not finished. Poll the SAME request_id; do not resubmit."}
        time.sleep(0.5)
