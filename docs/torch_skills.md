# Torch-Harness skill backends

The active protein-design plugin uses registered model APIs and native analysis.
It does not register the old compute control tools, expose `ddeharness compute`,
or probe/start OpenDDE compute during onboarding or doctor checks. Old compute
modules and examples remain as inactive compatibility sources; they are not a
fallback. Configure the actual model services with `protein-design models`.

Use `ddeharness protein-design start --config workflow.yaml`; this enables the
existing autonomous router by default. `--check` validates availability without
claiming inference. Recover with `status`, `resume`, `list`, `candidates`, and
`regions --run-dir ... --question-id ... --cdr-position ...` for pending user
region answers. The internal `initial_round` package and saved checkpoint names
remain compatible; there is no separate public initial-round command/tool.

## Installed skill mapping

| Skill | Executable backend |
| --- | --- |
| protein-design | Generic workflow tools and registered PPIFlow/TorchCraft/TorchScore/TorchFold/Rosetta/TorchOptimization APIs |
| cdr-full-redesign, minibinder-full-redesign | Registered compatible generators; partial/optimization actions only when supported by the current router and model mode |
| cdr-point-mutation, minibinder-point-mutation | Validated substitutions with fixed-position protection; generated sequences require real rescoring |
| developability-filter | Native chemical motif scan, native surface descriptors, reserved EpiMatrix adapter |
| structure-analysis, epitope-analysis | Biotite structures and minimum heavy-atom contacts with explicit chain roles and PDB hotspot IDs |
| evolutionary-analysis | Recorded candidate parent edges; no invented phylogeny |
| post-filter | Measured confidence ranking; incomplete candidates stay unranked; router acceptance unchanged |
| pdb-lookup, uniprot-lookup, pubmed-lookup | Read-only fixed public metadata APIs |
| antibody-inverse-folding, minibinder-inverse-folding, esm2-guided-mutation | Reference skills only; no corresponding registered backend, explicitly unavailable |

Native evidence is exposed as `protein_design_analyze`, or CLI
`ddeharness protein-design analyze --request request.json`. For example:

```json
{"action":"developability","sequences":{"B":"ACDEFGHIKLMNPQRSTVWY"}}
```

For surface analysis also supply `structure_path` and `binder_chains`; chain IDs
and complete sequences must agree. Native output contains `ptm_risks`,
`surface_patches`, and `immunogenicity` independently. Motifs are potential
chemical liabilities, not modification rates. Chemical PTM is distinct from
TorchScore structural pTM. `torch_surface_patch_v1` defines its exposure, distance,
hydrophobicity and charge calculations explicitly; its PSH/PNC/PSC values are not
the official TAP implementation or calibrated TAP pass/fail thresholds.

EpiMatrix is intentionally not connected. `EpiMatrixAdapter.predict(sequences)`
is the trusted Python integration point for a future licensed backend. Until it
exists, immunogenicity returns `available=false, scores=null`. Preserve method,
version, panel, units and provenance when implementing that adapter. No alternative
predictor or LLM score may be labelled EpiMatrix.

The context builder accepts these domains in Candidate `developability` or
`metadata.developability_evidence`, compresses them into Qwen's decision context,
and preserves unknown measurements. Analysis calls return evidence; they do not
silently rewrite checkpoints or change the score-based routing rules.


Surface output also includes a `residues` array for all observed binder residues,
including buried ones. Each row retains chain, PDB residue_id/insertion_code,
one-based observed sequence_position, amino_acid, Kyte-Doolittle hydropathy and
hydropathy_class, Black-Mould patch_hydrophobicity, sasa_angstrom2, relative_asa,
surface_exposed and charge_class. Positive/negative Kyte-Doolittle values label
hydrophobic/hydrophilic residues; this is separate from exposure and from the
Black-Mould PSH weighting scale. Structure-missing residues are not inferred.
