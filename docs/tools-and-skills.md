# Original tools and skills, and the added model registry

This inventory uses the checkout's original `fdab4f0` baseline (before our
initial-round/model-tool changes). It describes this repository snapshot,
not every upstream release or a user's private plugins/MCP servers.

**Tools** are callable functions. **Skills** are instruction documents loaded on
demand; installing a skill does not install its models or make an API reachable.
Tool visibility depends on disabled plugins, blacklists, progressive disclosure,
optional dependencies and the selected workflow-agent role.

## Original main-assistant tools

| Name | Function |
| --- | --- |
| `read` | Read files |
| `write` | Write files |
| `edit` | Apply file edits |
| `ls` | List directories |
| `grep` | Search file content |
| `find` | Find files |
| `bash` | Execute permitted shell commands |
| `web_search` | Search the web using configured search support |
| `web_fetch` | Fetch web content |
| `spawn` | Start a configured subagent |
| `ask_user` | Ask structured user questions |
| `use_skill` | Load a selected skill body |
| `tool_search` | Discover tools when progressive disclosure is enabled |
| `tool_call` | Invoke a discovered tool in that mode |
| `understand_media` | Optional long-term-memory plugin tool for PDF/image/audio/etc. understanding |

## Original Protein Design task-control tools

| Name | Function |
| --- | --- |
| `protein_design_context` | Preparation defaults, examples and compute configuration |
| `protein_design_start` | Launch the legacy evolutionary design task |
| `protein_design_status` | Query legacy task progress |
| `protein_design_adjust` | Adjust an existing task |
| `protein_design_stop` | Stop an existing task |
| `protein_design_candidates` | Read candidate results |
| `protein_design_search_target_msa` | Search target-chain MSA through compute |

## Original workflow-internal compute tools

These are declared by `ProteinDesignToolRegistry` and granted to particular
Analyze/Design/Quality/Reflection/PostFilter agents through their role profiles.
They are **not all automatically direct tools of the outer Qwen chat agent**.
They require the corresponding configured Harness compute service/backends.

| Name | Function |
| --- | --- |
| `protrek_sequence_search` | ProTrek sequence-neighbour search |
| `protrek_structure_search` | ProTrek structure-neighbour search |
| `generate_soluble_mpnn` | Structure-conditioned SolubleMPNN sequence proposals |
| `generate_esm2_guided` | ESM-2-guided mutation proposals |
| `analyze_epitope` | Epitope contacts and coverage |
| `analyze_structure` | Structure/interface analysis |
| `analyze_current_cycle_structure` | Structure analysis scoped to the current cycle |
| `analyze_evolution_tree` | Candidate lineage/evolution analysis |
| `check_antibody_developability` | Candidate developability assessment |

There are 15 general/optional tools, 7 original task-control tools, and 9 internal
compute tool definitions in these groups. This is not a promise that all 31 are
available simultaneously to a given conversation. User-defined MCP tools and
plugins add their own runtime-specific names and are not listed here.

## Original bundled skills (17)

| Skill | Purpose |
| --- | --- |
| `protein-design` | Overall design preparation, task execution and result interpretation |
| `cdr-point-mutation` | Local antibody CDR mutations |
| `cdr-full-redesign` | Full approved CDR-region redesign |
| `antibody-inverse-folding` | Structure-conditioned antibody sequence design |
| `minibinder-point-mutation` | Local minibinder sequence mutations |
| `minibinder-full-redesign` | Full mutable minibinder-sequence redesign |
| `minibinder-inverse-folding` | Structure-conditioned minibinder sequence design |
| `esm2-guided-mutation` | ESM-2 likelihood-guided antibody mutation proposals |
| `epitope-analysis` | Epitope/interface contacts and hotspot coverage |
| `structure-analysis` | Protein interaction/structure analysis |
| `developability-filter` | Developability and sequence liability filtering |
| `evolutionary-analysis` | Search lineages and structure/evolution comparison |
| `post-filter` | Evidence-based candidate Top-K selection |
| `pdb-lookup` | RCSB PDB structure lookup |
| `uniprot-lookup` | UniProt sequence/annotation lookup |
| `pubmed-lookup` | Biomedical literature lookup |
| `weather` | Weather lookup (general bundled skill) |

The current `protein-design` skill has been extended with the external model
workflow; it was not a newly created skill. User workspace/library skills can
extend or override this bundled set.

## Added external-model tools

All five documented services now have dedicated main-assistant tools:
`protein_model_ppiflow`, `protein_model_torchcraft`, `protein_model_torchfold`,
`protein_model_af3score`, and `protein_model_rosetta`. Shared tools are
`protein_model_list`, `protein_model_submit`, `protein_model_status`,
`protein_model_results`, and `protein_model_cancel` (AF3Score cancellation only).

Supported protocol modes from the supplied documentation/examples:

- PPIFlow: `antibody`, `nanobody`, `binder`, `partial_antibody`, `partial_nanobody`.
- TorchCraft: `mini_binder`, `vhh`, **`monomer_unconditional`**.
- TorchFold: prediction/refold through `/v1/folds` and prepared JSON input.
- AF3Score: `pdb`, `prepared`, `metrics`, including fast scoring, MSA search and
  pipeline-only behavior through its controls.
- Rosetta: `structure_relax`, `constrained_relax`, `interface_analyze`,
  `structure_score`, `interface_local_relax`, `mutation_binding_ddg`.

Unconditional TorchCraft monomer generation needs `json_path` and `design_length`,
not target/hotspot/framework fields. It is a standalone model-tool operation;
`initial-round` remains the bounded target-binding VHH/minibinder workflow.
Unknown additional service modes require their documented request contract;
the client does not bypass validation or invent unsupported modes.

## Why a registered tool can return an empty models list

The tool function is installed through the plugin manifest. Its **service aliases
and URLs** come from `plugins.config.protein-design.model_services` in the active
instance's config. Editing `docs/examples/model_services.json` alone does not
change the active config. The user's pasted trace showed the function existed,
but no service bindings were loaded.

First inspect the exact file selected by the CLI:

```bash
uv run --frozen ddeharness protein-design models list
```

If all five services are on the **same host as Harness**, explicitly register
the default endpoints (8010/8020/8030/8040/8050):

```bash
uv run --frozen ddeharness protein-design models register --local-defaults
uv run --frozen ddeharness protein-design models list --check-health
```

For a custom instance, append `--opendde-config /actual/path/config.json` to each
command. Compare that file with `protein_model_list.active_config_path` in the
running conversation. Different files mean different instances/configurations.

For different hosts/ports, edit a copy of the service fragment and import it:

```bash
uv run --frozen ddeharness protein-design models register \
  --services-file /actual/path/model_services.json \
  --opendde-config /actual/path/config.json
```

The command preserves Qwen/provider and unrelated settings, makes a private
backup before changing an existing file, and does not submit inference jobs.
Default registration fills missing model kinds without replacing an existing
custom alias/endpoint. Import conflicts need an explicit `--replace-existing`.
A disabled protein-design plugin remains disabled and is reported in the output.
Restart Harness after registration so the conversation loads the updated tools.
A registered URL is not proof of reachability or successful inference.

For request arguments and conversation examples, see
[protein-model tools](protein-model-tools.md).
