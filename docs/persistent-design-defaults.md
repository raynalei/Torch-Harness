# Configure once, then describe each new protein

The complete workflow tool now accepts a compact `task`. Deployment information
is loaded from the active Harness instance config, not recalled from conversation
history. Existing `run_config`, YAML and checkpoint entry points remain supported.
The current prototype routing rules and budgets are unchanged.

## One-time setup using an existing working configuration

Keep the existing seven registrations in `plugins.config.protein-design.model_services`.
Import a known task once to save its fixed deployment and workflow settings:

```bash
ddeharness protein-design models defaults \
  --from-workflow /actual/path/to/existing-task.yaml \
  --design-type vhh \
  --opendde-config /actual/path/to/config.json
```

If that profile includes TorchCraft, explicitly supply its actual hotspot numbering
convention once. For example, **only if the deployed API uses zero-based observed
sequence indices**:

```bash
ddeharness protein-design models defaults \
  --from-workflow /actual/path/to/existing-task.yaml \
  --design-type vhh \
  --torchcraft-hotspot-numbering sequence_0_based \
  --csv-id-column id --csv-sequence-column sequence \
  --opendde-config /actual/path/to/config.json
```

Other supported conventions are `sequence_1_based` and `pdb_residue_id`. CSV column
names must match the actual TorchCraft deployment; Harness does not infer that
contract from hotspot numbers. Configure `target_adapters.torchcraft.csv_extra_columns`
if the native reader requires additional fixed CSV columns.

The importer removes the previous target PDB, hotspots, target CSV, task name and
output folder. Old fold templates are cleared because they can contain another
antigen's MSAs/features. It removes framework and fixed/designable residue requirements as task inputs. It keeps weights/request settings, output root,
shared mappings and DockQ environment. Services uniquely registered globally are
reused rather than copied into the new profile. The imported common evaluation
selects TorchScore and enables autonomous; check the resulting configuration before
first use. Repeat once for each needed `design_type`: `vhh`, `antibody`, `minibinder`.
Import conflicts require explicit `--replace-existing`; provider credentials and
unrelated settings are preserved with a private backup.

## Where defaults live

Under `plugins.config.protein-design` in the active `config.json`:

| Setting | Purpose |
| --- | --- |
| `model_services.<alias>.service` | Endpoint, sbatch script, status protocol and API startup settings |
| `model_services.<alias>.service.environment` | Fixed source/Conda/Python/model locations exported to that API allocation |
| `model_services.<alias>.request_defaults` | Fixed HTTP request fields such as checkpoint, config and algorithm defaults |
| `workflow_defaults` | `work_dir`, `path_mappings`, DockQ/evaluation options and autonomous defaults |
| `design_profiles.<type>` | Enabled generators, design modes, generator settings and chain roles |
| `target_adapters.torchcraft` | Actual CSV columns and hotspot indexing convention |

`service.environment` accepts string environment values and is passed only to the
`sbatch` launch with `--export=ALL`. Use variable names actually read by the API
and sbatch script. For example, PyRosetta's supplied script supports
`ROSETTA_CONDA_ENV` and `ROSETTA_API_DIR`; a PPIFlow deployment may read `PPIFLOW_ROOT`
and `PPIFLOW_PYTHON`. A hardcoded export in a custom sbatch script must be changed
once to honor the desired variable. Harness does not edit third-party launchers or
install weights. Store credentials through the existing `token_env` mechanism,
not literal values in deployment defaults. Model listing exposes environment names,
not their values.

For example, add fixed request fields to an existing registration:

```json
{
  "kind": "ppiflow",
  "service": {
    "url": "http://slurm-api.invalid:8010",
    "sbatch_script": "/shared/apis/ppiflow/api.sbatch",
    "environment": {
      "PPIFLOW_ROOT": "/shared/models/ppiflow",
      "PPIFLOW_PYTHON": "/shared/envs/ppiflow/bin/python"
    }
  },
  "request_defaults": {
    "model_weights": "/shared/weights/ppiflow.ckpt",
    "config": "/shared/models/ppiflow/config.yaml",
    "num_timesteps": 200
  }
}
```

All seven model kinds support `request_defaults` using their real request schema.
Target-specific paths, hotspots, task names and old target sequence files are
rejected as persistent request defaults. Framework and fixed/designable residue requirements must be supplied by the current task, not persisted in profiles or request defaults. Legacy persisted values are ignored by the workflow for new tasks. If several registrations
share a kind, select an alias once with profile `model_aliases`, for example
`{"torchscore":"score_cluster"}`. Explicit task service blocks still take precedence.

Import a full config fragment containing model registrations and/or persistent
settings with the existing command:

```bash
ddeharness protein-design models register \
  --services-file /actual/path/to/defaults.json \
  --opendde-config /actual/path/to/config.json
```

This is one-time configuration, not a new YAML file for every experiment. Restart
Harness after importing so its plugin tools use the updated instance settings.
`protein_model_list` / CLI `models list` show configured profile names and request
defaults; registration does not claim real inference has been verified.

The built-in `evaluation.dockq_python` default is
`/lustre/grp/cmclab/leiy/conda_envs/ppi_env/bin/python`. No per-task lookup is needed.
An explicit workflow or task configuration can override it; the worker still checks
that the interpreter exists and DockQ imports successfully.

## New tasks in natural language

For example:

> Use target `/shared/inputs/new.pdb`, chain C, VHH, hotspots C42 and C73,
> framework `/shared/frameworks/current`, initial total 10, and run the configured autonomous workflow.

The assistant calls `protein_initial_round_start` with:

```json
{
  "task": {
    "pdb_path": "/shared/inputs/new.pdb",
    "target_chain": "C",
    "design_type": "vhh",
    "hotspots": ["C42", "C73"],
    "initial_samples": 10,
    "framework_dir": "/shared/frameworks/current"
  },
  "user_confirmed": true
}
```

Only use `user_confirmed=true` within the user's authorized computation scope.
Optional fields include `name`, `generators`, `generator_samples`, `framework_dir`,
`framework_cif_path`, `fixed_positions` and validated `generator_overrides`. Total
samples are split evenly across configured generators, with the remainder assigned
in profile order; explicit `generator_samples` must sum to the total. Every enabled
generator needs at least one sample. The prompt can select PPIFlow alone (`generators: ["ppiflow"]`) or TorchCraft alone (`generators: ["torchcraft"]`). Neither is mandatory. Explicit dual-generator requests remain supported. Paired VH/VL requires PPIFlow; TorchCraft supports VHH and mini-binders. The selected generator must exist in the design profile. A TorchCraft-only run does not implicitly configure PPIFlow Partial Flow; that action remains unavailable without a compatible PPIFlow refinement configuration. TorchOptimization remains available when configured and compatible. New fixed
positions require a compatible generator; TorchCraft's initial request has no such
field, so select PPIFlow only for that requirement. Subsequent Partial Flow respects
inherited fixed points and allowed regions.

## New-target handling and recovery

- Check the selected chain and every hotspot against the actual new structure.
  Hotspot inputs use PDB residue IDs, not assumed sequence offsets.
- Extract that target chain into the run's own `inputs/target.pdb`. Preserve residue
  numbers; if the chain conflicts with binder IDs, assign a free target chain and
  record the mapping. Unsupported insertion/negative/ambiguous numbering fails before
  submission because the current Rosetta residue-energy contract cannot encode it.
- Generate a new TorchCraft sequence CSV and convert hotspot indices using the
  configured adapter. Sequence provenance is **observed structure residues**, never
  fabricated missing biological sequence. If the native model needs a longer full
  sequence than the supplied structure represents, provide a compatible complete
  structure/verified sequence input; this adapter does not infer unobserved residues.
  An optional `target_sequence_csv` must match the observed target sequence; stale
  or mismatched files are rejected.
- Clear old target-specific fields and prepared fold templates. Generated requests
  use current PDB, hotspots, chain roles and per-run output directories. Static
  weights/evaluation settings are loaded from defaults. Framework and residue constraints come from the current task.
- Snapshot the resolved configuration to `worker-config.json` / `run-config.json`.
  Editing defaults affects new tasks, not existing runs. Missing or invalid inputs
  produce actionable errors before Slurm submission rather than guessed paths.

Without an explicit name, identical task inputs/defaults/target content get a stable
run name, preventing duplicate submission after reconnects. To intentionally repeat
an identical experiment, provide a new `name`. Target preparation is locked and
existing prepared inputs cannot be overwritten with different contents. Inspect
`inputs/target-provenance.json` for original chain/IDs and the generated input basis.

Precedence is model request defaults, then common workflow defaults, then the
selected design profile, then explicit task values. The final complete request is
validated against the original scientific API schema. Missing services, frameworks,
DockQ interpreters or incompatible budgets remain errors, not automatic substitutions.
