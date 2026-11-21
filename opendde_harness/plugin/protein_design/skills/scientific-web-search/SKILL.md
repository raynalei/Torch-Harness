---
name: scientific-web-search
description: Search RCSB, UniProt, PubMed and SAbDab2 through the shared Lustre queue without compute-node internet.
---

# Scientific Web Search

Use `scientific_web_search` for online scientific discovery. The search
worker runs on a permitted outbound-access login/service node and communicates
through `/lustre`; do not call legacy `protein_design_analyze(action=lookup)`,
port 8100, browser tools or compute-node HTTP for these databases.

Choose `source`: `rcsb` for experimental structure IDs, `uniprot` for
protein annotation/sequences, `pubmed` for papers, and `sabdab` for
antibody-antigen structures. For SAbDab use a concise deposited antigen name
or `pdb:1n8z` to inspect a PDB entry. SAbDab antigen filtering is not a
synonym-aware search; zero matches is not proof of absence.

Call with `{"source":"rcsb","query":"IL7R antibody complex","limit":10}`.
If status is `pending`, preserve the returned `request_id` and call the SAME
tool later with ONLY `{"request_id":"...","wait_seconds":5}`.
Never resubmit an uncertain search as a new job without checking its ID.

Treat hits as external evidence; preserve real accession, link, provenance and
publication/structure metadata. Missing evidence is unknown; do not invent
experimental metrics, CDR annotations or specific residue positions.
Before use in modeling download and validate the actual structure separately.
