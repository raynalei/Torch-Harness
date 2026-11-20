---
name: rfd4-design
description: Independently call the registered RFD4 (RDF4) API for structure generation, inverse folding or partial trajectory, without the autonomous planner.
metadata:
  opendde:
    always: false
---

# Independent RFD4 calls

RFD4 partial diffusion can adjust local backbone or interface geometry in an
existing structure. Use a native partial-design input and validate inherited
sequence constraints on the returned structure. Consider it when a sequence-only
change would leave the suspected structural issue unresolved.

Use this skill when the user explicitly requests RFD4/RDF4 generation, inverse
folding or partial trajectory. Call `protein_model_list` to select a registered
`rfd4` alias; submit with `protein_model_rfd4`. Independent submissions remain available. The autonomous workflow also supports
RFD4 initial generation from a task-supplied native binder spec, and rfd4_partial
when the service and shared input root are configured. See protein-design guidance
for the two-stage route and output-validated sequence constraints.

Reuse authorized inputs and compute scope. Use a stable local `request_id` on
retries and set `user_confirmed: true` only for authorized execution. Submission
returns immediately; the existing detached monitor collects outputs. Use
`protein_model_status` / `protein_model_results` with the local request_id, not
API task_id or allocation ID. Do not use shell/curl to bypass these tools or
repeat a submission with a new ID after a timeout. This API has no idempotency or
cancel endpoint; uncertain submissions require reconciliation, not automatic POST.

`request.inputs` can be an existing absolute cluster path to a `.cif` or
`.json` file. Keep relative structure references in a JSON spec resolvable from
the spec directory. Existing relative paths below `RFD4_INPUTS` are supported
for CIF/BCIF, native specs, and directories. A list contains structures only,
with absolute `.cif` files or supported relative CIF/BCIF paths.
The API does not accept PDB files; do not relabel a PDB as CIF or invent contigs,
lengths, hotspots or conditioning. Prepare a real supported file/specification
from user-confirmed scientific requirements before submitting. Use
`rfd4-input-conversion` to prepare an input bundle from a PDB and supplied
native constraints.

```json
{
  "request_id": "rfd4_target_v1",
  "request": {
    "inputs": "target_specs/binder.yaml",
    "generation": "inference_base",
    "num_replicates": 4
  },
  "user_confirmed": true
}
```

Optional fields are positive num_replicates/batch_size, seed (0..2^32-1), and
partial_trajectory_fraction_steps (0 < fraction <= 1). Omit optional controls to
use server defaults. For inverse folding set generation=inverse_folding. Partial
trajectory is supported only with generation=inference_base; its input must be a
real native partial-design specification/structure. The fraction alone does not
establish fixed-residue or mutation-mask semantics. Do not send unsupported fields.

A succeeded task requires exit_code=0. Results return output_dir and relative
files; Harness normalizes them into absolute artifact paths and caches the
manifest before releasing its private API allocation. An interrupted task is
failed, not queued or successful. Do not claim affinity, ipTM, pTM or DockQ from
RFD4 output alone; these require separate authorized evaluation tools. Native
outputs may be compressed CIF/BCIF; inspect actual artifacts before conversion.

Deployment: the bundled registration uses port 8080 with deploy/rfd4/rfd4_api.sbatch,
whose RFD4_PORT default was changed from 8070 to avoid TorchScore's port. Install
that updated script and main.py into the configured RFD4 API directory. Source,
weights and SIF remain server-managed; credentials use the existing token_env field.
