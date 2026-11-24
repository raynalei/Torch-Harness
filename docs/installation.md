# Installation

Install the client first. Register model services separately using the [deployment guide](protein-design.md).

## Install the client

Use Linux or macOS for the client.

The PyPI package is `opendde-harness`; its command is `ddeharness`. Both methods
use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage an
isolated Python environment. No environment activation is needed.

### Install from PyPI

```bash
uv tool install --python 3.12 opendde-harness
```

The wheel includes the built TUI and the bundled deployment profile. No source checkout or frontend build is required.

To update, close the TUI and run:

```bash
uv tool upgrade opendde-harness
```

OpenDDE Harness checks PyPI once a day and says when a newer release is out:
in the TUI's status bar and once in its transcript at launch, and in
`ddeharness doctor`. `ddeharness upgrade` runs the update for your kind of
install -- `uv tool upgrade opendde-harness` for a uv tool, `pip install
--upgrade opendde-harness` for a plain pip install; a source checkout is not
nudged. Set `OPENDDE_HARNESS_NO_UPDATE_CHECK=1` to turn the check off.

### Install from source

Install Git, uv, and Node.js 22.19 or newer with npm, then run:

```bash
git clone https://github.com/aurekaresearch/OpenDDE-Harness.git
cd OpenDDE-Harness
uv tool install --python 3.12 .
```

The package build installs frontend dependencies and builds the TUI when the
bundle is missing or older than its sources. Source edits do not automatically
update the installed command. To update, close the TUI, preserve local edits,
and run from the checkout:

```bash
git pull --ff-only
uv tool install --python 3.12 --reinstall .
```

To install edits already in the checkout, omit `git pull`.

### Verify the installation

After either installation:

```bash
ddeharness --version
ddeharness --help
ddeharness tui --check
```

If `ddeharness` is not on PATH, run `uv tool update-shell` and open a new terminal.
Source builds require Node.js 22.19 or newer with npm. Use
`ddeharness onboard` to configure the provider, memory and web search.
See [model registration](protein-model-tools.md) for external services.

## Uninstall

After stopping active tasks and closing the TUI, remove the client with:

```bash
uv tool uninstall opendde-harness
```

This leaves configuration, results, compute services, model data and the source checkout intact. Back up anything needed before removing those separately. Do not remove shared Docker resources, GPU drivers or system runtimes as part of client cleanup.

Application settings live in `~/.opendde_harness/config.json`. `OPENDDE_HARNESS_HOME` does not relocate this file or the protein-design task root; it overrides selected runtime directories such as Node.js and tracing. Use `OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT` for task state. Protect the configuration file because it may contain credentials. Default locations include:

```text
~/.opendde_harness/
  config.json                Provider, memory and Protein Design settings
  workspace/                 Agent workspace: TOOLS.md, skills/, agent_memory/,
                             user_memory/, seeded from the bundled templates
  memory/                    Long-term memory store
  protein_design/<task_id>/  Task state and worker logs
  runtime/                   Private Node.js runtime, when one was installed
  logs/                      Client logs, including memory-server.log
```

Original compute outputs live on the compute host. Captured dashboard structures and metrics are stored in tracing data. Model weights and OpenDDE data live under the two roots above.

The verification commands check CLI/TUI startup, not GPU inference. Use the registered model commands and workflow preflight to inspect external services. See [troubleshooting](troubleshooting.md) for installation, image-pull and mount failures.
