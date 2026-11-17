---
name: developability-filter
description: Compute chemical PTM motif risks and native surface PSH/PNC/PSC descriptors; report EpiMatrix immunogenicity only through an installed licensed adapter.
---

# Developability filter

Use `protein_design_analyze` with action `developability`. Supply a chain-to-sequence
mapping and, for surface analysis, a real binder structure and explicit chain IDs.
The backend is native Python/Biotite and never calls OpenDDE compute. Reuse existing
computed candidate evidence instead of recomputing it without a reason.

## 1. PTM / chemical modification risk

PTM here means post-translational/chemical modification, **not structural pTM**.
The installed sequence scanner reports N-glycosylation sequons N-X-S/T (X != P),
deamidation N-G/S/T, Asp isomerization D-G/S/T, Met/Trp oxidation sites, DP cleavage
motifs and cysteine sites. Positions are one-based within the named sequence.
These are potential liabilities, not modification probabilities, reaction rates
or evidence that a cysteine is free. No LLM numerical substitution is permitted.

## 2. Surface hydrophobic and charge patches: PSH, PNC, PSC

The installed `torch_surface_patch_v1` backend computes SASA on isolated binder
chains, selects residues with relative ASA >=0.075 (Tien maximum ASA values), and
uses minimum sidechain-heavy-atom distance (CA fallback for Gly). Surface pairs
within 7.5 Angstrom contribute H_i*H_j/d^2 to PSH (Black-Mould hydrophobicities),
1/d^2 for D/E pairs to PNC and 1/d^2 for K/R pairs to PSC. Histidine charge is not
assumed. Structure and sequence mappings must match.

Always report the method, exposure/distance settings and structure basis with the
scores. **These are explicit native descriptors, not the official TAP algorithm,
clinical thresholds or a validated TAP pass/fail result.** Do not compare methods
as if calibrated identically. Missing structures make this domain unavailable.

The `residues` array reports every observed binder residue, including buried
ones: chain, one-based observed sequence_position, PDB residue_id/insertion_code,
amino_acid, Kyte-Doolittle hydropathy, hydropathy_class (positive hydrophobic,
negative hydrophilic), Black-Mould patch_hydrophobicity, sasa_angstrom2,
relative_asa, surface_exposed and charge_class. These scales differ: aromatic
residues such as W/Y have negative Kyte-Doolittle values but high Black-Mould
patch weights. Always retain scale labels. Hydropathy describes residue chemistry;
surface_exposed is separate structure evidence. Neither predicts binding affinity.
Histidine charge remains unassigned. Sequence positions refer to observed residues,
not missing coordinates or antibody numbering schemes.

## 3. Immunogenicity: EpiMatrix

EpiMatrix is not installed/configured in this deployment. The backend exposes an
`EpiMatrixAdapter` protocol for an operator-supplied licensed integration. Until
then report `available=false`, `scores=null`, and the reason. Never substitute
NetMHC, a generic motif scanner, an LLM opinion or a zero score under the EpiMatrix
name. Real adapters must retain their version, panel, units and provenance.

## Evidence contract

Return domains independently: `ptm_risks`, `surface_patches`, `immunogenicity`.
A successful motif scan does not mean all domains are available or the candidate
passes developability. Keep uncertain/uncomputed domains visible. These findings
are advisory evidence and do not silently change the autonomous router thresholds.
