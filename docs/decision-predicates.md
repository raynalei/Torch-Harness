# Executable autonomous predicates

All scientific decisions use `policy.rule_actions(metrics, hotspot, score_metric,
min_iptm=0.5, min_ptm=0.8, min_dockq=0.49)`:

```python
if hotspot is False:
    return ["regenerate"]
if hotspot is not True or iptm_is_unknown:
    return []
if iptm <= 0.5:
    return ["regenerate"]
if ptm_is_unknown:
    return []
if ptm <= 0.8:
    return ["torchopt"]
if dockq_is_unknown:
    return []
if dockq <= 0.49:
    return ["partial_flow"]
return ["partial_flow", "expand", "stop"]
```

| Function | Purpose |
| --- | --- |
| `diagnostics.analyze` | Maps hotspot IDs and computes minimum heavy-atom contact distances; records which structure was used |
| `policy.hotspot_hit` | At least one specified hotspot closer than 5 Å; missing evidence is unknown |
| `autonomous.evaluate` | Scores every candidate, deduplicates, refolds those passing both confidence gates, computes DockQ and records the rule classification |
| `autonomous.options_for` | Builds only rule-permitted, capability-valid options for Qwen |
| `regions.infer_cdr_positions` | Python framework alignment estimates three CDRs and includes inserted query residues |
| `regions.python_partial_positions` | Encodes CDR/fixed complementary masks from actual candidate PDB numbering |
| `policy.torchopt_supported` | Requires a registered optimizer, compatible single-chain mode and valid chain IDs |
| `policy.repair_allowed` | Avoids repeating refinement of the same parent; children remain eligible |
| `policy.stop_reason` | Enforces extra-round cap, total requested-sample budget and consecutive no-improvement cap |

No flexibility/local-problem RMSD gate is used. Rosetta interface_dg participates
in Qwen comparison and 100*ipTM-interface_dg ranking without changing rule branches.
If a selected Partial Flow has unreliable automatic CDR identification,
`region_input.request_regions` persists a user question and pauses execution;
`region_input.supply_regions` validates the actual user ranges, computes their fixed
complement and resumes. No general action approval is added. Unsupported model modes
remain unavailable; residue IDs are never fabricated.
See [autonomous feedback](autonomous-feedback.md) for alignment assumptions and limits.
