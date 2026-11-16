# Executable autonomous predicates

`policy.rule_actions` applies the [staged autonomous routes](autonomous-feedback.md).

```python
if iptm_or_ptm_is_unknown_or_invalid:
    return []
if stage == 'initial' and iptm < 0.5:
    return ['terminate']
if iptm > 0.8 and ptm > 0.8:
    if dockq_is_not_yet_available:
        return ['refold']
    return ['stop'] if dockq >= min_dockq else ['terminate']
if stage == 'partial_flow':
    return ['torchopt']
if stage == 'torchopt':
    return ['terminate']
if interface_fixed_positions_have_not_been_computed:
    return ['score_interface']
return ['partial_flow'] if any_fixed_points else ['terminate']
```

| Function | Purpose |
| --- | --- |
| `policy.candidate_stage` | Derives initial/Partial Flow/optimization stage from checkpointed lineage |
| `diagnostics.analyze` | Records hotspot contacts and the stage's permitted next operations |
| `autonomous.evaluate` | Scores each child; submits only the required refold or Rosetta operation |
| `interface_energy.interface_energies` | Applies the uploaded script's negative-contact energy sums per binder residue |
| `interface_energy.partial_positions` | Encodes energy-selected fixed points and their mutable complement in real PDB numbering |
| `autonomous.options_for` | Builds capability-valid Partial Flow/optimization/stop options |
| `policy.torchopt_supported` | Requires a registered optimizer and a compatible single-chain mode |
| `policy.repair_allowed` | Prevents repeating a refinement of the same parent |
| `policy.stop_reason` | Enforces round, sample and no-improvement caps |

Contacts are diagnostic. Missing scientific outputs are errors; a scalar interface
energy cannot replace a residue-energy table. See the workflow guide for the
required Rosetta log format, strict boundaries and supported Partial Flow modes.
