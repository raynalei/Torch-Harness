"""Read-only metadata lookup through the login-node API."""

from __future__ import annotations

import os
import re

import httpx

IDENTIFIER_PATTERNS = {
    "pdb": (r"[0-9A-Za-z]{4}", "expected a four-character PDB identifier"),
    "uniprot": (r"[A-Z0-9]{6,10}(?:-[0-9]+)?", "expected a UniProt accession"),
    "pubmed": (r"[0-9]+", "expected a numeric PMID"),
}


def validate_identifier(source, identifier):
    if source not in IDENTIFIER_PATTERNS:
        raise ValueError("unsupported public source")
    pattern, message = IDENTIFIER_PATTERNS[source]
    if not re.fullmatch(pattern, identifier):
        raise ValueError(message)


def validate_lookup_url(value):
    if not isinstance(value, str):
        raise ValueError("public_lookup_url must be an HTTP(S) URL")
    url = httpx.URL(value)
    if url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.query or url.fragment:
        raise ValueError("public_lookup_url must be an HTTP(S) base URL without credentials, query or fragment")
    return str(url).rstrip("/")


def public_lookup_url(config):
    value = os.environ.get("TORCH_HARNESS_PUBLIC_LOOKUP_URL") or config.get("public_lookup_url")
    return validate_lookup_url(value) if value else None


def lookup_via_api(source, identifier, *, base_url, transport=None):
    validate_identifier(source, identifier)
    if not base_url:
        raise ValueError("set public_lookup_url or TORCH_HARNESS_PUBLIC_LOOKUP_URL to the login-node lookup API")
    endpoint = f"{validate_lookup_url(base_url)}/v1/lookup/{source}/{identifier}"
    with httpx.Client(timeout=30, transport=transport, trust_env=False, follow_redirects=False) as client:
        response = client.get(endpoint)
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, dict) or data.get("source") != source or data.get("identifier") != identifier:
        raise ValueError("login-node lookup API returned a mismatched result")
    return data


def lookup_public(source, identifier, *, transport=None):
    """Fetch from a fixed upstream URL on the login-node API host."""
    validate_identifier(source, identifier)
    if source == "pdb":
        url = "https://data.rcsb.org/rest/v1/core/entry/" + identifier.upper()
    elif source == "uniprot":
        url = "https://rest.uniprot.org/uniprotkb/" + identifier + ".json"
    else:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id=" + identifier
    with httpx.Client(timeout=20, transport=transport, follow_redirects=False) as client:
        response = client.get(url)
        response.raise_for_status()
        return {"available": True, "source": source, "identifier": identifier, "data": response.json()}
