import json

import pytest
from typer.testing import CliRunner

from opendde_harness import node_runtime as node_paths
from opendde_harness.cli import onboard_commands, tui_commands
from tests._config import config_dict, keyed

NOT_CONFIGURED = "OpenDDE Harness is not configured. Run `ddeharness onboard`."
#: The least a config can hold and still launch: one provider with a key, and a
#: default model whose prefix names that provider. Built by the one helper that
#: knows the shape, so a change to it reaches this suite without an edit here.
MINIMAL_CONFIG = config_dict(keyed("openai", key="sk-test"), model="openai/gpt-4o-mini")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def write_config(home, payload):
    path = home / ".opendde_harness" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def launch(monkeypatch):
    calls = []
    monkeypatch.setattr(tui_commands, "find_node", lambda: ("node", (22, 19, 0)))
    monkeypatch.setattr(tui_commands, "resolve_dist", lambda name: node_paths.packaged_dist_dir() / name)
    monkeypatch.setattr(tui_commands, "run_subprocess_with_rpc", lambda *args, **kwargs: calls.append(args) or 0)
    monkeypatch.setattr("opendde_harness.cli.update_notice.maybe_refresh_async", lambda: None)
    return calls


def test_non_tty_first_run_exits_without_launching(home, launch, monkeypatch, capsys):
    monkeypatch.setattr(tui_commands, "_stdout_isatty", lambda: False)
    monkeypatch.setattr(
        onboard_commands, "run_wizard", lambda **kwargs: pytest.fail("wizard must not run without a TTY")
    )
    result = CliRunner().invoke(tui_commands.tui_app, [])
    assert result.exit_code == 1
    assert NOT_CONFIGURED in result.output + capsys.readouterr().out
    assert launch == []


def test_tty_first_run_launches_after_the_wizard_configures(home, launch, monkeypatch):
    monkeypatch.setattr(tui_commands, "_stdout_isatty", lambda: True)
    seen = []

    def wizard(**kwargs):
        seen.append(kwargs)
        write_config(home, MINIMAL_CONFIG)

    monkeypatch.setattr(onboard_commands, "run_wizard", wizard)
    result = CliRunner().invoke(tui_commands.tui_app, [])
    assert result.exit_code == 0
    assert seen == [{"show_next_steps": False}]
    assert len(launch) == 1


def test_cancelled_wizard_exits_without_launching(home, launch, monkeypatch, capsys):
    monkeypatch.setattr(tui_commands, "_stdout_isatty", lambda: True)

    def wizard(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(onboard_commands, "run_wizard", wizard)
    result = CliRunner().invoke(tui_commands.tui_app, [])
    assert result.exit_code == 1
    assert NOT_CONFIGURED in result.output + capsys.readouterr().out
    assert launch == []


def test_configured_install_skips_the_wizard(home, launch, monkeypatch):
    write_config(home, MINIMAL_CONFIG)
    monkeypatch.setattr(tui_commands, "_stdout_isatty", lambda: False)
    monkeypatch.setattr(onboard_commands, "run_wizard", lambda **kwargs: pytest.fail("wizard must not run"))
    result = CliRunner().invoke(tui_commands.tui_app, [])
    assert result.exit_code == 0
    assert len(launch) == 1


def test_unusable_default_model_hints_and_launches(home, launch, monkeypatch, capsys):
    # A key for one provider and a default model another one serves: the id's
    # prefix is what routes it, and nothing configured answers to that prefix.
    write_config(home, config_dict(keyed("openai", key="sk-test"), model="anthropic/claude-sonnet-5"))
    monkeypatch.setattr(tui_commands, "_stdout_isatty", lambda: True)
    monkeypatch.setattr(onboard_commands, "run_wizard", lambda **kwargs: pytest.fail("wizard must not run"))
    result = CliRunner().invoke(tui_commands.tui_app, [])
    assert result.exit_code == 0
    assert "No usable provider resolves the default model" in result.output + capsys.readouterr().out
    assert len(launch) == 1


def test_check_smoke_path_needs_no_config(home, monkeypatch):
    calls = []
    monkeypatch.setattr(tui_commands, "find_node", lambda: ("node", (22, 19, 0)))
    monkeypatch.setattr(tui_commands, "resolve_dist", lambda name: node_paths.packaged_dist_dir() / name)
    monkeypatch.setattr(tui_commands, "run_subprocess", lambda *args, **kwargs: calls.append(args) or 0)
    monkeypatch.setattr("opendde_harness.cli.update_notice.maybe_refresh_async", lambda: None)
    monkeypatch.setattr(onboard_commands, "run_wizard", lambda **kwargs: pytest.fail("wizard must not run"))
    result = CliRunner().invoke(tui_commands.tui_app, ["--check"])
    assert result.exit_code == 0
    assert len(calls) == 1


def test_missing_node_is_provisioned_before_launch(home, launch, monkeypatch):
    from opendde_harness.cli import node_runtime

    write_config(home, {**MINIMAL_CONFIG, "language": "zh"})
    monkeypatch.delenv(node_runtime.DISABLE_ENV, raising=False)
    monkeypatch.delenv("OPENDDE_HARNESS_NODE", raising=False)
    monkeypatch.setattr(tui_commands, "_stdout_isatty", lambda: False)
    monkeypatch.setattr(onboard_commands, "run_wizard", lambda **kwargs: pytest.fail("wizard must not run"))
    lookups = [("/old/node", (20, 20, 1)), ("/home/runtime/node", (22, 20, 0))]
    monkeypatch.setattr(tui_commands, "find_node", lambda: lookups.pop(0))
    installs = []
    monkeypatch.setattr(node_runtime, "install_node", lambda **kwargs: installs.append(kwargs) or "/home/runtime/node")
    result = CliRunner().invoke(tui_commands.tui_app, [])
    assert result.exit_code == 0, result.output
    assert len(installs) == 1 and installs[0]["translate"]("en", "zh") == "zh"
    assert launch[0][0] == "/home/runtime/node"


def test_node_provisioning_failure_exits_with_its_message(home, launch, monkeypatch):
    from opendde_harness.cli import node_runtime

    write_config(home, MINIMAL_CONFIG)
    monkeypatch.delenv(node_runtime.DISABLE_ENV, raising=False)
    monkeypatch.delenv("OPENDDE_HARNESS_NODE", raising=False)
    monkeypatch.setattr(tui_commands, "find_node", lambda: (None, None))
    monkeypatch.setattr(
        node_runtime,
        "install_node",
        lambda **kwargs: (_ for _ in ()).throw(
            node_runtime.NodeRuntimeError("download failed for https://nodejs.org/x")
        ),
    )
    result = CliRunner().invoke(tui_commands.tui_app, [])
    assert result.exit_code == 1
    assert "download failed for https://nodejs.org/x" in result.output
    assert launch == []


def test_node_provisioning_can_be_disabled(home, launch, monkeypatch, capsys):
    from opendde_harness.cli import node_runtime

    write_config(home, MINIMAL_CONFIG)
    monkeypatch.setenv(node_runtime.DISABLE_ENV, "1")
    monkeypatch.setattr(tui_commands, "find_node", lambda: (None, None))
    monkeypatch.setattr(node_runtime, "install_node", lambda **kwargs: pytest.fail("provisioning is disabled"))
    result = CliRunner().invoke(tui_commands.tui_app, [])
    assert result.exit_code == 1
    assert "Node.js ≥ 22" in result.output + capsys.readouterr().out
    assert launch == []


def test_terminal_fds_are_redirected_to_the_log_while_the_tui_owns_the_screen(tmp_path):
    import os

    from opendde_harness.cli._log_file import redirect_terminal_fds_to_file

    log = tmp_path / "tui.log"
    before = os.fstat(1).st_ino
    with redirect_terminal_fds_to_file(log):
        os.write(1, b"Runtime code ready: /tmp/x\n")
        os.write(2, b"stray stderr\n")
        assert os.fstat(1).st_ino == os.fstat(2).st_ino == os.stat(log).st_ino
    assert os.fstat(1).st_ino == before
    text = log.read_text()
    assert "Runtime code ready" in text and "stray stderr" in text
