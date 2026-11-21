#!/usr/bin/env python3
"""Restricted official-source search worker. Run ONLY on permitted outbound-access node."""
from __future__ import annotations
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(os.environ.get("HARNESS_WEB_ROOT", "/lustre/grp/cmclab/leiy/harness_web")).resolve()
REQ, PROCESSING, RESULTS = (ROOT / part for part in ("requests", "processing", "results"))


def fetch(url, *, body=None):
    headers = {"User-Agent": "Torch-Harness-ScientificSearch/1.0", "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    else:
        data = None
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=25) as response:
        if "json" not in response.headers.get("Content-Type", "").lower():
            raise ValueError("expected JSON from official upstream")
        return json.load(response)


def sabdab_id(raw):
    raw = raw.strip().lower()
    if re.fullmatch(r"[a-z0-9]{4}", raw):
        return "pdb_" + raw.rjust(8, "0")
    if re.fullmatch(r"pdb_[a-z0-9]{8}", raw):
        return raw
    raise ValueError("invalid SAbDab PDB identifier")


def search(source, query, limit):
    if source not in {"rcsb", "uniprot", "pubmed", "sabdab"}:
        raise ValueError("unsupported source")
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 300:
        raise ValueError("query must be 1-300 characters")
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("limit must be 1-20")
    if source == "rcsb":
        payload = {"query": {"type": "terminal", "service": "full_text", "parameters": {"value": query}},
                   "return_type": "entry", "request_options": {"paginate": {"start": 0, "rows": limit}}}
        try:
            data = fetch("https://search.rcsb.org/rcsbsearch/v2/query", body=payload)
        except urllib.error.HTTPError as e:
            if e.code == 204:
                return {"total_count": 0, "hits": []}
            raise
        return {"total_count": data.get("total_count", 0),
                "hits": [{"id": r["identifier"],
                          "url": "https://www.rcsb.org/structure/" + urllib.parse.quote(r["identifier"])}
                         for r in data.get("result_set", [])[:limit]]}
    if source == "uniprot":
        endpoint = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {"query": query, "size": limit, "format": "json"})
        data = fetch(endpoint)
        return {"hits": [{"id": r.get("primaryAccession"),
                          "protein": (r.get("proteinDescription") or {}).get("recommendedName", {}).get("fullName", {}).get("value"),
                          "url": "https://www.uniprot.org/uniprotkb/" + (r.get("primaryAccession") or "")}
                         for r in data.get("results", [])[:limit]]}
    if source == "pubmed":
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        params = urllib.parse.urlencode({"db": "pubmed", "term": query, "retmax": limit, "retmode": "json", "tool": "torch_harness"})
        found = fetch(base + "esearch.fcgi?" + params)["esearchresult"]
        ids = found.get("idlist", [])
        if not ids:
            return {"total_count": int(found.get("count", 0)), "hits": []}
        params = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "json", "tool": "torch_harness"})
        details = fetch(base + "esummary.fcgi?" + params)["result"]
        return {"total_count": int(found.get("count", 0)),
                "hits": [{"id": pmid, "title": details.get(pmid, {}).get("title"),
                          "pubdate": details.get(pmid, {}).get("pubdate"),
                          "url": "https://pubmed.ncbi.nlm.nih.gov/" + pmid + "/"} for pmid in ids]}
    api = "https://sabdab.opig.stats.ox.ac.uk/api"
    if query.lower().startswith("pdb:") or re.fullmatch(r"(?:[a-z0-9]{4}|pdb_[a-z0-9]{8})", query, re.I):
        key = sabdab_id(query[4:] if query.lower().startswith("pdb:") else query)
        result = fetch(api + "/pdb/" + urllib.parse.quote(key, safe=""))
        return {"mode": "pdb", "hits": [{"id": key, "data": result,
                "url": "https://sabdab.opig.stats.ox.ac.uk/pdb/" + key}]}
    if len(query) > 25:
        raise ValueError("SAbDab antigen name filter permits at most 25 characters")
    url = api + "/antibody-instances?" + urllib.parse.urlencode(
        {"binds_to_antigen": query, "limit": limit, "offset": 0})
    page = fetch(url)
    if not isinstance(page, dict) or not isinstance(page.get("results"), list):
        raise ValueError("unexpected SAbDab2 response")
    return {"mode": "antigen", "total_count": page.get("total"),
            "hits": [{"id": item.get("id"), "data": item,
                      "url": "https://sabdab.opig.stats.ox.ac.uk/search/"}
                     for item in page["results"][:limit] if isinstance(item, dict)],
            "note": "Antigen name filters do not necessarily match gene aliases"}


def atomic_write(path, payload):
    tmp = path.parent / ("." + path.stem + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def tick():
    for file in sorted(REQ.glob("*.json")):
        if not re.fullmatch(r"[0-9a-f]{32}", file.stem):
            continue
        claimed = PROCESSING / file.name
        try:
            os.replace(file, claimed)
        except FileNotFoundError:
            continue
        try:
            request = json.loads(claimed.read_text(encoding="utf-8"))
            if request.get("request_id") != file.stem:
                raise ValueError("request_id mismatch")
            output = {"request_id": file.stem, "status": "completed",
                      "source": request["source"], "query": request["query"],
                      "data": search(request["source"], request["query"], request.get("limit", 10))}
        except Exception as exc:
            output = {"request_id": file.stem, "status": "failed", "error": str(exc)}
        atomic_write(RESULTS / file.name, output)
        claimed.unlink(missing_ok=True)
        print(file.stem, output["status"], flush=True)


def main():
    os.umask(0o077)
    for path in (REQ, PROCESSING, RESULTS):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    print("search worker watching", ROOT, flush=True)
    while True:
        tick()
        time.sleep(2)


if __name__ == "__main__":
    main()
