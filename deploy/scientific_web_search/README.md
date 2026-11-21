# Torch-Harness scientific web search via /lustre

This replaces the protein-design plugin's old PDB/UniProt/PubMed HTTP-based
lookup skills/tool action. Registered Qwen tool: `scientific_web_search`.
Supported sources: RCSB, UniProt, NCBI PubMed, SAbDab2.

## Login/service node (requires outbound network access)

Use a node where your cluster policy permits this lightweight persistent
process. No GPU or HTTP listener is necessary.

```bash
cd /lustre/grp/cmclab/leiy/Torch-Harness
export HARNESS_WEB_ROOT=/lustre/grp/cmclab/leiy/harness_web
mkdir -p "$HARNESS_WEB_ROOT"/{requests,processing,results}
chmod 700 "$HARNESS_WEB_ROOT" "$HARNESS_WEB_ROOT"/{requests,processing,results}
python3 deploy/scientific_web_search/worker.py
```

## Compute node / TUI

Set exactly the same root in the environment that runs Harness, then restart
the TUI so it loads the new registered tool and skill:

```bash
export HARNESS_WEB_ROOT=/lustre/grp/cmclab/leiy/harness_web
uv run --frozen ddeharness tui
```

Ask Qwen: "Use scientific_web_search to find IL7R antibody structures in RCSB,
then search SAbDab for the same antigen." Or submit through the tool directly
with `source=pubmed`, `query=IL7R antibody`.

New searches return `pending` with an ID if the worker has not completed
within `wait_seconds` (default 5, max 30). Call again with ONLY `request_id`
to poll. The queue is append-only and crash-recoverable; a worker crash may leave
a request in `processing`; operators should check that artifact before
requeueing it. Stale results should be pruned administratively after archiving.

An API search result is **metadata**, not a downloaded PDB, a verified antibody
complex, or experimental validation. On compute nodes no external network access,
no SSH tunnel, no 8100 port, no OpenDDE compute service is required.
