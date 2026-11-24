"""What a slash answers when a command needs something, and how long it waits."""

from __future__ import annotations

import asyncio

from opendde_harness.tui_rpc.methods import slash_routing


def _exec(command: str) -> dict:
    return asyncio.run(slash_routing.slash_exec({"command": command}))


def test_a_command_that_needs_an_argument_says_which_one() -> None:
    """Click answers a red "Usage: Missing argument 'NAME'." over a "no output"
    tail. That is a stack of noise around one fact: it needs a name."""
    answer = _exec("provider get")

    assert answer["output"] == "/provider get needs <name>"
    # One line, no second thing to read, and no escape codes in a transcript.
    assert "warning" not in answer
    assert "\n" not in answer["output"]
    assert "\x1b" not in answer["output"]


def test_a_missing_option_is_named_the_way_it_is_typed() -> None:
    assert _exec("protein-design initial-round")["output"] == "/protein-design initial-round needs --config <config>"


def test_the_hint_in_the_answer_is_the_one_the_popup_shows() -> None:
    """Two ways of telling the user the same thing; they read the same source."""
    from opendde_harness.tui_rpc.methods.commands import commands_catalog

    catalog = asyncio.run(commands_catalog({}))
    hints = {entry["name"]: entry.get("argument_hint") for entry in catalog["commands"]}

    for name in ("provider get", "sessions resume"):
        assert _exec(name)["output"] == f"/{name} needs {hints[name]}"


def test_a_command_that_fails_for_any_other_reason_keeps_its_error() -> None:
    """Only a usage error is reshaped: everything else is the command talking,
    and a summary of it would be this module's invention."""
    answers = []

    async def fake_dispatch(params, **_kwargs):
        answers.append(params)
        return {"stdout": "", "stderr": "boom: the disk is full", "exit_code": 1}

    original = slash_routing.cli_dispatch
    slash_routing.cli_dispatch = fake_dispatch
    try:
        answer = _exec("sessions list")
    finally:
        slash_routing.cli_dispatch = original

    assert answer["warning"] == "boom: the disk is full"


def test_a_command_that_declares_a_timeout_is_given_it(monkeypatch) -> None:
    from opendde_harness.tui_rpc.methods.cli_dispatch import _DISPATCH_TIMEOUTS

    monkeypatch.setitem(_DISPATCH_TIMEOUTS, ("doctor",), 1800.0)
    seen = []

    async def fake_dispatch(params, **_kwargs):
        seen.append(params["timeout_s"])
        return {"stdout": "ok", "stderr": "", "exit_code": 0}

    original = slash_routing.cli_dispatch
    slash_routing.cli_dispatch = fake_dispatch
    try:
        _exec("doctor")
        _exec("sessions list")
    finally:
        slash_routing.cli_dispatch = original

    assert seen == [1800.0, slash_routing._SLASH_TIMEOUT_S]


def test_the_catalog_publishes_what_the_gateway_will_allow(monkeypatch) -> None:
    """So a client can say a command may take a while, and cannot shorten it."""
    from opendde_harness.tui_rpc.methods.cli_dispatch import _DISPATCH_TIMEOUTS, dispatch_timeout
    from opendde_harness.tui_rpc.methods.commands import commands_catalog

    monkeypatch.setitem(_DISPATCH_TIMEOUTS, ("doctor",), 1800.0)
    catalog = asyncio.run(commands_catalog({}))

    for entry in catalog["commands"]:
        declared = dispatch_timeout(entry["name"].split())

        assert entry.get("timeout_s") == declared, entry["name"]

    assert any(entry.get("timeout_s") for entry in catalog["commands"])
