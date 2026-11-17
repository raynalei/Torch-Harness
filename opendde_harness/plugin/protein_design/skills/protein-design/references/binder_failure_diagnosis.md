# Binder failure diagnosis

Separate off-epitope contact evidence from structural confidence. Both can coexist. Missing or ambiguous evidence is unknown, not a pass or failure. Follow actual tool capabilities and task constraints; Qwen chooses scientific actions while rules remain fallback advice.

## Off-epitope

Use mapped hotspot contacts on the recorded structure basis. `epitope_hit=false`
means the evaluated pose misses all specified hotspots at the configured contact
cutoff, not that experimental binding is absent. `null` means mapping/contact
assessment is unknown; inspect the mapping reason before describing a miss.
Check input versus candidate chain IDs, target cropping and true PDB numbering.
Generated and refolded structures can give different contacts; report which was
measured. Hotspot evidence is diagnostic under the current autonomous policy.

## Low structural confidence

Use the selected scorer's complex ipTM and pTM independently. Low ipTM concerns
confidence in relative chain placement; pTM concerns predicted fold/topology.
pLDDT is local confidence and PAE describes positional uncertainty; neither is an
experimental affinity measurement. Do not substitute generic or AF3Score values
for missing TorchScore fields. Preserve metric units/provenance when available.
A high-confidence pose may be off-epitope, and a contacted hotspot may still have
low structural confidence. Neither diagnosis implies the other.

## Interface and developability

Rosetta contact energies identify favorable residue contributions, not affinity.
DockQ measures consistency with the generated reference pose. Developability
metrics may help compare legal candidates only when actually computed. Read chemical PTM motifs, method-labelled surface PSH/PNC/PSC descriptors and
EpiMatrix availability from native developability evidence. EpiMatrix is not
connected. Missing measurements remain unavailable; sequence prose or a
Qwen opinion must not become a fabricated numerical measurement or routing gate.
