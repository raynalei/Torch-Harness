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

## Convert a PDB and constraints

From the Torch-Harness checkout, use the `rfd4-input-conversion` skill or run its
script in the same RFD4-Proteina image as the API. This CPU-only step uses the
native condition-spec writer. Set `RFD4_SIF` and `RFD4_INPUTS` to the values
used by the API launcher.

```bash
SCRIPT="$PWD/opendde_harness/plugin/protein_design/skills/rfd4-input-conversion/scripts/convert_rfd4_inputs.py"
STAGING=/path/to/directory/containing/target-and-constraints

apptainer exec --bind "$SCRIPT:/convert_rfd4_inputs.py:ro" \
  --bind "$STAGING:/staging:ro" --bind "$RFD4_INPUTS:/inputs:rw" \
  "$RFD4_SIF" python /convert_rfd4_inputs.py pack \
  --pdb /staging/target.pdb --constraints /staging/constraints.yaml \
  --out-dir /inputs/target_bundle

apptainer exec --bind "$SCRIPT:/convert_rfd4_inputs.py:ro" \
  --bind "$RFD4_INPUTS:/inputs:rw" "$RFD4_SIF" \
  python /convert_rfd4_inputs.py unpack --spec /inputs/target_bundle/spec.yaml \
  --out-dir /inputs/target_editable
```

The YAML/JSON constraint file uses the native
[condition-spec format](https://rfd4-proteina.org/latest/reference/yaml-json.html)
with a single design key and its `contig` and/or `conditions` fields. The script
adds `input: source.cif`. Submit `target_bundle/spec.yaml` or
`target_bundle/annotated.cif` as the relative `request.inputs` value. Keep
`source.cif` with the spec for unpacking and replay.

Conversion supports a single-model PDB and an unpackable spec with one plain
source CIF. If PDB cannot preserve the source identifiers, use the native CIF
directly. PDB has limited bond-order and charge information. For ligands or
modified residues, verify those details against the original mmCIF before using
a converted file. The output directory must be new for either command.
