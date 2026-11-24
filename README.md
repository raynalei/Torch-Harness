# OpenDDE Harness


[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
![Status](https://img.shields.io/badge/status-preview-orange)
[![PyPI](https://img.shields.io/pypi/v/opendde-harness.svg?cacheSeconds=300)](https://pypi.org/project/opendde-harness/)


![tui](docs/assets/tui.png)

Harness for agentic antibody design: prepare targets, optimize CDR sequences, predict structures with registered model services, and inspect results. Supports VHH, scFv and paired VH/VL binders while preserving configured fixed residues.

- Guided setup and reviewed design plans in natural language.
- Generate antibody sequences through LLM reasoning and structural verification.
- Initial Round and Autonomous Design Loop workflows.

> [!NOTE]
> OpenDDE Harness is an early preview, and you may encounter bugs. If something
> goes wrong, please [open an issue](https://github.com/aurekaresearch/OpenDDE-Harness/issues)
> with steps to reproduce it and your `ddeharness doctor --json` output.
> Your feedback helps us fix problems and improve the tool together.


## News

- **2026-09-09: Introducing OpenDDE Harness (preview) for agentic antibody design! Read the [technical report](docs/assets/OpenDDE_harness_tech_report.pdf).**
    - Design VHH, scFv, and paired VH/VL binders with LLM-guided sequence optimization and OpenDDE structure prediction.
    - Get started with the [installation guide](docs/installation.md), [onboarding](docs/onboarding.md), and [design examples](docs/examples/).
    - Inspect candidate sequences, structures, and design progress in the [tracing dashboard](docs/tracing-board.md).

See the [changelog](CHANGELOG.md) for release details.


https://github.com/user-attachments/assets/1530c7e6-9a7e-4069-b861-0e6cf58e6752


## Installation

Use a Linux or macOS client with Python 3.12 or newer. Native Windows is not
currently supported. Use
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) to install the
`opendde-harness` package and its `ddeharness` command in an isolated environment.
No environment activation is needed.

### Install from PyPI

```bash
uv tool install --python 3.12 opendde-harness
```

The release wheel includes the built terminal UI.

To update a PyPI installation, close the TUI and run:

```bash
uv tool upgrade opendde-harness
```

OpenDDE Harness checks PyPI once a day and says when a newer release is out:
in the TUI's status bar and once in its transcript at launch, and in
`ddeharness doctor`. `ddeharness upgrade` runs the update for your kind of
install (`uv tool upgrade` or `pip install --upgrade`). Set
`OPENDDE_HARNESS_NO_UPDATE_CHECK=1` to turn the check off.

### After installation

After installation, verify the client:

```bash
ddeharness --version
ddeharness --help
ddeharness tui --check
```

If the command is not found, run `uv tool update-shell` and open a new terminal.
Updates preserve configuration, results, and model data. See the [installation guide](docs/installation.md) for source installs.

### Uninstall

Close the TUI and remove the client with:

```bash
uv tool uninstall opendde-harness
```

Configuration, results, model data, and compute services are retained.

## Quick Start

### 1. Configure the client and model services

```bash
ddeharness onboard
ddeharness doctor
ddeharness protein-design models list
```

The wizard configures the LLM provider, optional memory, and web search.
See [model registration](docs/protein-model-tools.md) and the
[deployment guide](docs/protein-design.md) for model endpoints and workflow profiles.

### 2. Start a design

Launch the terminal UI:

```bash
ddeharness
```

Then describe your task:

> Design a VHH against human CRLF2. Verify the target and epitope, keep the framework fixed, and design CDRs. Show the plan and start only after I confirm.

See the [terminal UI guide](docs/tui.md) for the screen, commands and keys.
CLI-based design workflows are also supported. See the
[CLI usage guide](docs/cli.md).
See [Initial Round](docs/initial-round.md) and [Autonomous Design Loop](docs/autonomous-feedback.md).

### 3. Inspect results

```bash
ddeharness tracing
```

Open the printed URL and select **Protein design** to inspect sequences, structures, metrics and agent activity.

![Candidate sequences, structures and linked metrics](docs/assets/tracing_board_1.png)

![Design progress, agent activity and candidate lineage](docs/assets/tracing_board_2.png)

Detached runs record their working directory when launched. Use that directory
to inspect or resume a run after reconnecting. See the [CLI guide](docs/cli.md)
and [dashboard guide](docs/tracing-board.md).

## Development & License

For a deterministic PPIFlow + TorchCraft initial round using external Slurm APIs,
see the [initial-round guide](docs/initial-round.md) and
[example configuration](docs/examples/initial_round_sbatch.yaml).
To call PPIFlow, TorchCraft, TorchFold and AF3Score directly from Qwen or another
chat provider, see [registered protein-model tools](docs/protein-model-tools.md).
For the original tools/skills and registration troubleshooting, see the
[tools and skills inventory](docs/tools-and-skills.md).
For PPIFlow-only background automation and shorter Qwen responses, see
[automatic initial rounds](docs/automatic-initial-round.md).

See the [repository rules](AGENTS.md). Licensed under [Apache-2.0](LICENSE); see [third-party notices](LICENSES/README.md). TUI and memory foundations are adapted from Raven, and long-term memory is served by EverOS. Models may have separate terms. Computational results require experimental validation. Model calls and compute may incur costs.

## Citation and Acknowledgements

If you use OpenDDE Harness in your work, please cite this software and the technical report linked above. When using OpenDDE for structure prediction, also cite the [OpenDDE technical report](https://arxiv.org/abs/2607.03787) and follow its citation and acknowledgement guidance. Cite the original methods for the models and tools used in your experiments.

We acknowledge Raven, EverOS, and the upstream projects listed in the [third-party notices](LICENSES/README.md). Their software and model licenses continue to apply.

## Partnership and Collaboration

![Collaboration](docs/assets/hiring.png)
