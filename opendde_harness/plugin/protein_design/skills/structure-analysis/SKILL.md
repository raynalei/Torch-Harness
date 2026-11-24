---
name: structure-analysis
description: |
  Analyze protein-ligand or protein-protein interactions from a single structure
  file with a locally installed PLIP executable.
  Use this skill when you need to profile interaction types such as hydrogen
  bonds, hydrophobic contacts, pi-stacking, salt bridges, or water bridges.
---

# Structure Analysis Skill

## Standalone Usage

```bash
python -m opendde_harness.plugin.protein_design.analysis.plip_runner \
    outputs/.../some_complex_structure.cif \
    --cyc-num 1 \
    --name cycle_1_mut_001 \
    --binder-chain <binder_chain_ids> \
    -o outputs/.../some_structure_interactions.txt
```

Pass a SINGLE predicted complex structure file. The file should already contain both target and binder chains. For protein-protein complexes, provide the binder chain IDs explicitly with `--binder-chain`.

Converts `.cif`/`.mmcif` to `.pdb` automatically when needed, runs PLIP on the given structure, and writes the interaction report to a txt file.

Do not split the complex into separate binder/target files. This wrapper expects one complex file and optional binder chain IDs.

## Interpreting Results

Focus on the interaction summary PLIP reports:

- **H-bonds / salt bridges** — often strongest interaction signals
- **Hydrophobic contacts** — useful for interface packing
- **Pi-stacking** — notable for aromatic residues

Output is a text report with interaction type counts, residue identities, and distances per binding site.

Use interaction claims only for the exact analyzed complex. Do not infer
hydrogen bonds, salt bridges, pi-stacking, or residue contacts from confidence
scores alone.

## Dependencies

Install PLIP in the local environment and ensure its command is on PATH.
Use the optional `protein-design` dependencies for local structure parsing.
