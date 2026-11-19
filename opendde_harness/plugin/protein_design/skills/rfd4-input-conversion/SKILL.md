---
name: rfd4-input-conversion
description: Convert a PDB and supplied RFD4-Proteina constraints into an annotated CIF and native condition spec, or export the spec and its source CIF back to PDB and YAML constraints.
metadata:
  opendde:
    always: false
---

# RFD4 input conversion

Use the bundled `scripts/convert_rfd4_inputs.py` in the RFD4-Proteina Python
environment. The tool uses the upstream AtomWorks parser and condition-spec
writer. It runs on CPU without model weights. On a cluster, use the RFD4 image
and bind the script, inputs and writable output parent. See the deployment
README for the Apptainer command.

## Pack

Use the caller's PDB and YAML/JSON constraints. The constraint file names one
design and carries that design's native `contig` and/or `conditions` fields.
Use an empty mapping for a design only if the caller explicitly requests no
constraints. Ask for the native constraints when they are missing. Preserve
chain and residue references as supplied.

```bash
python "$SCRIPT" pack --pdb target.pdb --constraints constraints.yaml --out-dir bundle
```

The bundle contains `source.cif`, `spec.yaml` and `annotated.cif`.
`spec.yaml` references its sibling `source.cif` by relative path.
`annotated.cif` embeds the applied conditions. For an authorized submission,
load `rfd4-design` and pass the relative path to `spec.yaml` or `annotated.cif`
under `RFD4_INPUTS`. A conversion request authorizes file preparation only.

## Unpack

```bash
python "$SCRIPT" unpack --spec bundle/spec.yaml --out-dir editable
```

This writes `source.pdb` and `constraints.yaml` from the spec and its original
source CIF. Keep the source CIF and spec together. An annotated CIF alone can
contain cropped or renumbered residues, so it cannot recover the original
structure and authoring constraints. Use a single model and one plain source
CIF with PDB-compatible IDs. If conversion reports changed or incompatible
identifiers, keep the native CIF and inspect the identifiers before editing
any selection.
