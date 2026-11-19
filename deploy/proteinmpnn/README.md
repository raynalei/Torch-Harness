# ProteinMPNN standalone API

Install the uploaded `main.py` and `proteinmpnn_api.sbatch` into
`/lustre/grp/cmclab/leiy/proteinmpnn_api` (confirmed). The API listens on 8090.
Harness sets sbatch_chdir to this directory: the supplied launcher explicitly
uses SLURM_SUBMIT_DIR, so running sbatch from an arbitrary Harness directory
would import the wrong main.py or fail.

Source, SIF, Conda API Python, weights and GPU-job resource defaults remain in
the service environment. The launcher defaults are included in the deployment
YAML. The uploaded adapter uses --use_soluble_model and the configured weight
file. It submits one inference Slurm child job and returns its task_id/job_id.

Use skill `proteinmpnn-design` and tool `protein_model_proteinmpnn`. Required:
absolute existing PDB and design_chains containing existing unique chain IDs.
There are no residue-level fixed/mask request fields. Results are FASTA sequences,
not folded structures. Harness collects JSON and the /fasta response before
requesting cleanup of its private API allocation. Headers/chain separators and
native reference records are preserved in the cached FASTA.

No cancel/idempotency endpoint exists. An uncertain request is not automatically
resubmitted. ProteinMPNN is standalone, not an optimizer option in the two-stage
planner. Independently evaluate generated sequences when structural metrics are
required; do not infer ipTM/pTM/interface energies from FASTA headers.
