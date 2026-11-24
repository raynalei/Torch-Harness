---
name: evolutionary-analysis
description: |
  Analyze saved candidate lineages, supplied Newick trees and alignments.
---

# Evolutionary Analysis

## Standalone Usage

Population-only analysis:

```bash
python -m opendde_harness.plugin.protein_design.analysis.evolution_analysis \
    --candidates_json_path ~/.opendde_harness/protein_design/TASK_ID/search_history.json
```

Pass existing Newick or alignment files using `--tree-path CHAIN=PATH` or
`--alignment-path CHAIN=PATH`. The script summarizes the files locally.

## Input

The candidate path may be `search_history.json`, `population.json`, or a task
directory containing one of those files.

OpenDDE Harness's canonical fields are `candidate_id`, `parent_id`, `cycle`, `objective`,
`objective_key`, `minimize`, `metrics`, `mutations`, `skill_id`, gate evidence,
and the population action. Unknown or obsolete external field names are not
accepted; producers must normalize data at OpenDDE Harness's task boundary.

## Report Sections

- known parent-child search lineages
- summary for each supplied Newick tree
- objective-aware root-to-leaf trajectories
- mutation recurrence across candidates and independent parents
- sequence conservation for each supplied alignment

Do not interpret candidate count alone as convergence. Require recurrence
across independent parents and positive objective-aware improvement evidence.
Report candidate IDs, mutations and objective values from the saved lineage.
Base structural clusters and conservation on the supplied trees and alignments.

## Component CLIs

For a smaller report, use:

```bash
python -m opendde_harness.plugin.protein_design.analysis.lineage_analysis \
    --candidates_json_path ~/.opendde_harness/protein_design/TASK_ID/search_history.json
```
