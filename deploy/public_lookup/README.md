# Public database lookup on the login node

Run this API on a login node with outbound access to RCSB, UniProt and NCBI.
It needs Python with FastAPI, Uvicorn and HTTPX, and access to this checkout.
From the repository root, run in that Python environment:

```bash
PYTHONPATH="$PWD" python -m uvicorn \
  opendde_harness.plugin.protein_design.scientific_agent.backends.public_lookup_api:app \
  --host 0.0.0.0 --port 8100
```

Use your cluster's internal hostname for that login node in the environment that
starts Harness:

```bash
export TORCH_HARNESS_PUBLIC_LOOKUP_URL=http://LOGIN_NODE_HOSTNAME:8100
```

Alternatively, set `public_lookup_url` to that URL under
`plugins.config.protein-design` in the active Harness configuration, or define it
as a top-level deployment YAML setting. The environment variable takes
precedence. Restart Harness after changing its configuration.

From a compute node, check the connection and a sample lookup:

```bash
curl -fsS http://LOGIN_NODE_HOSTNAME:8100/health
curl -fsS http://LOGIN_NODE_HOSTNAME:8100/v1/lookup/pdb/1ABC
```

The same route supports `uniprot` with an accession or `pubmed` with a PMID.
Harness uses these routes for `protein_design_analyze` with action `lookup`.
The API accepts database identifiers and contacts fixed upstream metadata
endpoints. Harness reports API errors without trying a direct internet lookup
from the compute node.
