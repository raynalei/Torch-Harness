"""Read-only public metadata lookup, independent of model and compute services."""

from __future__ import annotations

import re

import httpx


def lookup_public(source, identifier, *, transport=None):
    if source == "pdb":
        if not re.fullmatch(r"[0-9A-Za-z]{4}", identifier):
            raise ValueError("expected a four-character PDB identifier")
        url = "https://data.rcsb.org/rest/v1/core/entry/" + identifier.upper()
    elif source == "uniprot":
        if not re.fullmatch(r"[A-Z0-9]{6,10}(?:-[0-9]+)?", identifier):
            raise ValueError("expected a UniProt accession")
        url = "https://rest.uniprot.org/uniprotkb/" + identifier + ".json"
    elif source == "pubmed":
        if not re.fullmatch(r"[0-9]+", identifier):
            raise ValueError("expected a numeric PMID")
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id=" + identifier
    else:
        raise ValueError("unsupported public source")
    with httpx.Client(timeout=20, transport=transport, follow_redirects=False) as client:
        response = client.get(url)
        response.raise_for_status()
        return {"available": True, "source": source, "identifier": identifier, "data": response.json()}
