# CLI usage

Prefix commands with `ddeharness`. Every command accepts `--help`.

| Command | Purpose |
| --- | --- |
| `onboard` | Configure the LLM provider, memory and web search |
| `status` | Show client status |
| `doctor` | Inspect provider and memory configuration. Use `--probe` to send an LLM test message |
| `plugins` | List installed plugins |
| `tracing` | Open the tracing dashboard. Use `tracing stop` to stop the viewer |
| `provider` | Manage providers and model settings |
| `skill` | List, inspect, block and unblock skills |
| `tui` | Launch the terminal UI |
| `sessions` | Manage conversation sessions |
| `protein-design models` | List and register model endpoints and manage request defaults |
| `protein-design initial-round --config <profile.yaml> --check` | Validate local inputs and check service readiness without submitting jobs |
| `protein-design initial-round --config <profile.yaml> --detach` | Start a background run |
| `protein-design initial-round-list` | Discover saved background runs |
| `protein-design initial-round-status --run-dir <directory>` | Read saved progress |
| `protein-design initial-round-regions --run-dir <directory>` | Supply the requested regions using the options shown in `--help` |
| `protein-design initial-round-resume --run-dir <directory>` | Resume the saved run |

See [Initial Round](initial-round.md) and [Model Registry](protein-model-tools.md)
for configuration and recovery.
