# RFD4 standalone API

`main.py` is the uploaded ASGI adapter. Install it and `rfd4_api.sbatch` into
`/lustre/grp/cmclab/leiy/rfd4_api` (the confirmed deployment directory).
The updated script defaults to **8080**, configurable through RFD4_PORT; the
uploaded script used 8070, already assigned to TorchScore. Update the installed
script together with Harness's service registration. Do not point Harness at
8080 while still running the unmodified 8070 launcher.

The bundled registration sets sbatch_chdir so SLURM_SUBMIT_DIR/main.py resolve in
the correct directory. RFD4_API_DIR, SIF/weights/input/job roots are configured
once in cluster_deployment.yaml. Server weights must include config.json and
model.safetensors, as checked by the launch script. No per-task weight overrides
are accepted. Optional RFD4_API_TOKEN must match the registration's token_env;
never store literal tokens in YAML.

Use skill `rfd4-design` and tool `protein_model_rfd4`, then shared status/results
tools. Inputs are relative to RFD4_INPUTS. Generation=inference_base supports
optional partial_trajectory_fraction_steps; generation=inverse_folding does not.
Structure lists accept CIF/BCIF (including compressed files); PDB is unsupported.

This API runs inference in a single background worker inside its sbatch GPU
allocation. The allocation stays active until the task is terminal and Harness
has cached its artifact manifest. interrupted/failed is not successful completion.
No metrics/cancel/idempotency endpoint exists. This integration is standalone and
does not add RFD4 to the two-stage autonomous generator/optimizer menu.
