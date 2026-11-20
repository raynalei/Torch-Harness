"""Login-node API for fixed-source public database metadata."""

from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException

from .public_lookup import lookup_public, validate_identifier

app = FastAPI(title="Torch-Harness Public Lookup")


@app.get("/health")
def health():
    return {"ok": True, "service": "public-lookup"}


@app.get("/v1/lookup/{source}/{identifier}")
def lookup(source: Literal["pdb", "uniprot", "pubmed"], identifier: str):
    try:
        validate_identifier(source, identifier)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    try:
        return lookup_public(source, identifier)
    except httpx.TimeoutException as exc:
        raise HTTPException(504, f"{source} lookup timed out") from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 404:
            raise HTTPException(404, f"{source} identifier not found") from exc
        if status == 429:
            retry_after = exc.response.headers.get("Retry-After")
            headers = {"Retry-After": retry_after} if retry_after else None
            raise HTTPException(429, f"{source} is rate limiting requests", headers=headers) from exc
        raise HTTPException(502, f"{source} returned an upstream error") from exc
    except (httpx.RequestError, ValueError) as exc:
        raise HTTPException(502, f"{source} lookup failed") from exc
