---
name: proteinmpnn-design
description: Independently design sequences on existing PDB chains using the registered ProteinMPNN API and retrieve native FASTA outputs.
metadata:
  opendde:
    always: false
---

# Independent ProteinMPNN calls

Call `protein_model_list` for a configured proteinmpnn alias, then use
`protein_model_proteinmpnn`. Independent calls return FASTA. The autonomous planner also offers proteinmpnn
when compatible; it validates sequence masks on outputs and realizes an admitted
sequence with TorchFold before TorchScore. It is independent of the retired OpenDDE
compute service and its legacy SolubleMPNN tool.

Required inputs are an absolute cluster pdb_path ending in .pdb, and design_chains
containing unique one-character chain IDs already present in the input's ATOM
records. These are existing chains whose sequences will be designed, not names
for a new generated chain. Confirm which chains are designable; preserve target
chains by not listing them. The server enforces allowed input roots and existence.

```json
{
  "request_id": "mpnn_binder_v1",
  "request": {
    "pdb_path": "/lustre/grp/cmclab/leiy/inputs/complex.pdb",
    "design_chains": ["B"],
    "num_seq_per_target": 8
  },
  "user_confirmed": true
}
```

Reuse the same local request_id after reconnects/timeouts. user_confirmed=true
requires existing authorization of inputs, parameters and compute scope. Optional
controls are num_seq_per_target (1..512), sampling_temp (0 < temperature <= 2),
and seed (1..2^32-1). Omit controls to use server defaults. Inspect the active
registration's per-request sample cap if the user requests a larger batch.

This exact API supports chain-level design only: it has no fixed_positions,
mask_positions, framework-preservation, partial-residue mask or structure-refinement
request field. Explain an unsupported constraint rather than pretending it will
be enforced or forwarding unsupported parameters. The uploaded server launcher
uses --use_soluble_model and its configured weights; do not silently change models.

Outputs are FASTA sequences, not newly folded or docked PDBs. Headers and chain
separators are preserved; native reference records may be present alongside design
samples. Do not treat every FASTA record as a newly designed binder or invent
quality metrics. Re-fold/re-score sequences with separately authorized tools if
structure, ipTM/pTM, DockQ or interface energy are needed.

Submission returns immediately. A detached monitor verifies completion and caches
native results plus FASTA text/records before releasing the owned API allocation.
Use protein_model_status and protein_model_results with the local request_id.
API task_id, inference Slurm job_id, and the API allocation ID are distinct. There
is no cancellation endpoint or request idempotency key; preserve uncertain IDs and
never blindly POST again. Do not poll continuously through repeated chat turns.

Deployment uses port 8090. Install deploy/proteinmpnn/main.py and
proteinmpnn_api.sbatch in the configured API directory. sbatch_chdir is required
because the supplied script changes into SLURM_SUBMIT_DIR. Conda/API Python,
container, model weights and compute resources remain server-managed.
