"""The slash catalog offers what the gateway will actually run, and says what it is.

Reported by the owner: several entries in the slash popup were described as
"runs on the gateway" and failed when run. Both defects had one cause. The
catalog served group *heads* -- ``provider``, ``sessions``, ``skill``,
``compute``, ``protein-design`` -- which are not commands: Typer would print
the group's help, and ``cli.dispatch`` answers ``not_dispatch_compatible``. And
it served only names, so the client had nothing to describe them with.

The rule here is the dispatcher's own: an entry belongs in the catalog exactly
when ``cli.dispatch`` would accept it. There is no second list to maintain.
"""

from __future__ import annotations

import asyncio

from opendde_harness.tui_rpc.methods.cli_dispatch import _DISPATCH_BLACKLIST, _is_dispatch_compatible
from opendde_harness.tui_rpc.methods.commands import commands_catalog


def _catalog() -> dict:
    return asyncio.run(commands_catalog({}))


def test_every_command_offered_is_one_dispatch_accepts() -> None:
    for entry in _catalog()["commands"]:
        argv = entry["name"].split()

        assert _is_dispatch_compatible(argv), f"/{entry['name']} is offered but dispatch refuses it"


def test_no_bare_group_head_is_offered() -> None:
    """A head with subcommands under it is the shape that failed."""
    names = {entry["name"] for entry in _catalog()["commands"]}

    for head in ("provider", "sessions", "skill", "protein-design"):
        assert head not in names, f"/{head} is a group, not a command"
        assert any(name.startswith(f"{head} ") for name in names), f"nothing under /{head} is offered"


def test_nothing_the_blacklist_names_is_offered() -> None:
    """The interactive logins, the servers, the wizards and the downloads."""
    names = {entry["name"] for entry in _catalog()["commands"]}

    for prefix in _DISPATCH_BLACKLIST:
        blocked = " ".join(prefix)

        assert not any(name == blocked or name.startswith(f"{blocked} ") for name in names), blocked


def test_what_cannot_run_inside_a_request_is_not_offered() -> None:
    """Named, not derived from the blacklist, so removing an entry fails here
    rather than quietly narrowing the loop above.

    A dispatch is a request: it has a timeout and no terminal of its own. A
    server that runs until stopped, a login that wants a browser, and the two
    wizards that read stdin cannot be one.
    """
    names = {entry["name"] for entry in _catalog()["commands"]}

    assert "compute serve" not in names
    assert "provider login" not in names
    assert "onboard" not in names
    assert "tui" not in names
    # What is left of those groups is what finishes.
    assert not any(name.startswith("compute ") for name in names)
    assert "provider list" in names


def test_every_entry_says_what_it_does() -> None:
    """The client no longer writes a description of its own, so an entry
    without one would reach the popup as a bare name."""
    empty = [entry["name"] for entry in _catalog()["commands"] if not entry["description"].strip()]

    assert empty == []


def test_a_description_is_one_line_of_the_commands_own_help() -> None:
    commands = {entry["name"]: entry for entry in _catalog()["commands"]}

    assert commands["sessions list"]["description"] == "List sessions, sorted by most recently updated."
    # A docstring wraps in source; the description is the sentence, not the
    # first physical line, and it carries no RST markup into the popup.
    assert (
        commands["provider list"]["description"]
        == "Show every provider this config configures, and whether it is usable."
    )
    # A full stop inside "(e.g. cli:<id>)" does not end the sentence.
    assert commands["sessions delete"]["description"].endswith("(e.g. cli:<id>).")

    for entry in commands.values():
        assert "``" not in entry["description"]
        assert "\n" not in entry["description"]
        assert len(entry["description"]) <= 120


def test_an_argument_hint_names_what_the_command_takes() -> None:
    commands = {entry["name"]: entry for entry in _catalog()["commands"]}

    assert commands["sessions resume"]["argument_hint"] == "<id-or-prefix>"
    # Optional in brackets: `tracing` opens the dashboard with no argument.
    assert commands["tracing"]["argument_hint"] == "[action]"
    # An option with no default is not optional: without it the command
    # answers "Missing option", so the hint says how to type it.
    assert commands["protein-design initial-round"]["argument_hint"] == "--config <config>"
    # And a command that takes none says nothing rather than an empty string.
    assert "argument_hint" not in commands["sessions list"]


def test_the_heads_are_still_served_for_the_client_that_reads_them() -> None:
    """``pairs`` predates this and the Ink front-end still gates on it."""
    catalog = _catalog()

    assert catalog["pairs"]
    assert all(pair[0].startswith("/") for pair in catalog["pairs"])
