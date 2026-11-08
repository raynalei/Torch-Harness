"""Provider subcommands — owns the ``provider_app`` Typer instance.

A provider is named by its pi provider id, exactly as ``config.json`` files it:
``anthropic``, ``openai-codex``, ``azure-openai-responses``, or a name this
config declared for a provider of its own. An id has one spelling, so there is
nothing here that normalises a name or resolves an alias.

Lifecycle commands:

- ``provider login <id>`` — pi-ai's own OAuth sign-in, run inside the model
  service (OpenAI Codex today). The grant lands in the service's credential
  store, which is the same file it refreshes the token in.

Config subcommands:

- ``provider list``                 — the providers this config configures
- ``provider get <id>``             — current entry (secrets redacted)
- ``provider set <id> [...]``       — patch fields (--api-key, --base-url, --api, ...)
- ``provider use <id>/<model>``     — make one model the agent's default
- ``provider test <id>``            — ask the model a few tokens through the
                                      model service, the way a turn would
- ``provider reset <id>``           — remove the entry; a stored sign-in goes with it
- ``provider show <id>``            — reflect available ``--flag`` fields

Model subcommands (``provider model ...``) write one row of an entry's
``models`` list -- what the user knows about a model that no catalogue does:
the api it is served on, its context window and output ceiling, a name, a price:

- ``provider model set <id> <model> [--api openai-responses] [--context-window N] ...``

Architecture: write operations go ONLY through
:mod:`opendde_harness.config.update_providers`. Command bodies do not import
``load_config`` / ``save_config`` / provider Pydantic classes.

``commands.py`` imports :data:`provider_app` and registers it on the top-level
``app`` via ``app.add_typer(provider_app, name="provider")``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from opendde_harness import __logo__

# Eagerly, unlike every other import in this file: pi's id tables are pure data
# with no imports of their own, and the ``--api`` flag's help names the wires it
# accepts -- which is decided when the command is defined, not when it runs.
from opendde_harness.providers import pi_ids

console = Console()

provider_app = typer.Typer(help="Manage providers")


# ---------------------------------------------------------------------------
# Signing in
# ---------------------------------------------------------------------------
#
# The flow itself is pi-ai's and runs inside the model service: it is the
# process that holds the credential store and refreshes the token afterwards,
# so it is the one that has to write the grant. What is left on this side is
# the terminal -- printing each step the service reports, and opening a browser,
# which a child process with no display cannot do.


class ConsoleInteraction:
    """How a sign-in reaches the person who asked for it, on a terminal.

    Two things, because two things are all the service cannot do for itself:
    print a line, and hand the user a browser. The prompts pi's flow asks are
    answered inside the service (the login-method menu) or left to the browser
    callback (the paste-the-code fallback), so nothing here reads input.
    """

    def __init__(self, *, open_browser: bool = True) -> None:
        self._open_browser = open_browser and _can_open_browser()

    def notify(self, message: str) -> None:
        console.print(message)

    def open_url(self, url: str) -> bool:
        """Hand the user a browser, or say there is none to hand them.

        The vendors that own these flows print a URL and stop there, which is
        the only reason the sign-in commands do anything beyond running them.
        """
        if not self._open_browser or not url:
            return False

        import webbrowser

        try:
            return bool(webbrowser.open(url))
        except Exception:  # noqa: BLE001 - a missing browser is not an error here
            return False


def _can_open_browser() -> bool:
    """Whether this session has a browser to hand the user off to.

    A headless Linux box has no display to open one on; every other platform is
    assumed to have one.
    """
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _show_login_event(event: dict[str, Any], interaction: ConsoleInteraction) -> None:
    """One step of pi's flow, as the line or two a terminal should show.

    pi notifies four shapes (``AuthEvent``): a device code and where to type it,
    a sign-in URL, an informational line, a progress line. A prompt arrives as
    ``ask`` and is reported rather than answered -- the service settles the
    login-method menu itself and leaves the paste-the-code prompt to its own
    callback -- so the only one worth saying anything about is the wait.
    """
    notify = event.get("notify")
    if isinstance(notify, dict):
        kind = notify.get("type")
        if kind == "device_code":
            uri = str(notify.get("verificationUri") or "")
            code = str(notify.get("userCode") or "")
            interaction.notify(f"Open [cyan]{uri}[/cyan] and enter the code [bold]{code}[/bold]")
            if interaction.open_url(uri):
                interaction.notify("[dim]Opened in your browser. This waits until you are done.[/dim]")
            else:
                interaction.notify("[dim]This waits until you are done.[/dim]")
            return
        if kind == "auth_url":
            url = str(notify.get("url") or "")
            if interaction.open_url(url):
                interaction.notify(f"[dim]Opened {url}[/dim]")
            else:
                interaction.notify(f"Sign in at [cyan]{url}[/cyan]")
            interaction.notify(f"[dim]{notify.get('instructions') or 'Finish there, then come back here.'}[/dim]")
            return
        message = str(notify.get("message") or "")
        if message:
            interaction.notify(f"[dim]{message}[/dim]")
        for link in notify.get("links") or ():
            if isinstance(link, dict) and link.get("url"):
                interaction.notify(f"  [dim]{link.get('label') or 'See'}: {link['url']}[/dim]")
        return
    ask = event.get("ask")
    if isinstance(ask, dict) and ask.get("type") == "manual_code":
        interaction.notify("[dim]Waiting for the browser to come back…[/dim]")


#: What ``--method`` accepts, and the mode the service's ``login`` takes for it.
_LOGIN_MODES = {"browser": "browser", "device": "device_code", "device_code": "device_code"}

#: The default, and why it is not the browser: device code completes on any
#: machine and binds no port, while browser login finishes on the service's own
#: localhost callback -- and when that port is taken, pi falls back to asking
#: for the code to be pasted, which nothing on this side of the pipe can answer.
_DEFAULT_LOGIN_MODE = "device_code"


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(..., help="Provider that signs in (e.g. 'openai-codex')"),
    method: str = typer.Option(
        "",
        "--method",
        help="'device' (default: a code to type, works headless) or 'browser' (a localhost callback)",
    ),
    no_browser: bool = typer.Option(False, "--no-browser", help="Print the sign-in URL instead of opening it"),
):
    """Sign in to a provider that is reached by OAuth."""
    name = provider.strip()
    _refuse_unnameable(name)

    # pi owns the set that signs in, so the answer for anything else is the key
    # path rather than a device flow that no vendor here would honour.
    if name not in pi_ids.OAUTH:
        signs_in = ", ".join(sorted(pi_ids.OAUTH))
        console.print(f"[red]{name} is not signed in to[/red]  The ones that are: {signs_in}")
        console.print(f"  [dim]A key goes in with `ddeharness provider set {name} --api-key <key>`.[/dim]")
        raise typer.Exit(1)

    # Before anything talks to a vendor: an option this command does not accept
    # used to be swallowed by a handler's catch-all, so a command rejected on
    # its own terms still started a device flow and printed a code.
    mode = _LOGIN_MODES.get((method or _DEFAULT_LOGIN_MODE).lower())
    if mode is None:
        console.print(f"[red]Unknown sign-in method: {method}[/red]  Supported: browser, device")
        raise typer.Exit(1)

    label = _display_name(name)
    console.print(f"{__logo__} OAuth Login - {label}\n")
    run_login(name, label, mode=mode, open_browser=not no_browser)


def run_login(provider: str, label: str, *, mode: str = _DEFAULT_LOGIN_MODE, open_browser: bool = True) -> None:
    """Run this provider's sign-in through the model service and report it.

    Success means a credential was stored, not that a model answered: proving a
    sign-in by asking a model for one token conflates holding a credential with
    being entitled to that model, and bills the user to find out.

    Called by the command above and by the onboarding wizard, so both end in one
    flow and one store. The wizard wants a boolean rather than an exit, which is
    why it catches ``typer.Exit`` instead of this raising something of its own.
    """
    from opendde_harness.providers.model_service import ModelServiceError

    interaction = ConsoleInteraction(open_browser=open_browser)
    try:
        result = asyncio.run(_login_through_the_service(provider, mode=mode, interaction=interaction))
    except KeyboardInterrupt:
        console.print("[yellow]Sign-in cancelled.[/yellow]")
        raise typer.Exit(1)
    except ModelServiceError as exc:
        # The code is ours -- "login_failed", "no_node", "no_bundle" -- and says
        # which of the three went wrong. The message is pi's or a vendor's, and
        # a failed token exchange can quote what it was given, so it goes to the
        # debug log rather than to the screen.
        console.print(f"[red]Sign-in failed ({exc.code}).[/red]")
        if exc.code == "login_failed":
            console.print("  [dim]Try again, or run with --method browser.[/dim]")
        logger.debug("provider login: {} failed: {}", provider, exc)
        raise typer.Exit(1)
    except Exception:  # noqa: BLE001 - reported, not raised
        console.print("[red]Sign-in failed.[/red]  [dim]Try again, or run with --method browser.[/dim]")
        logger.debug("provider login: {} failed", provider, exc_info=True)
        raise typer.Exit(1)

    if result.get("type") != "oauth":
        # pi answers with the credential's own type. Anything but a grant means
        # the flow completed as something this command did not ask for.
        console.print(f"[red]Sign-in did not produce a grant for {label}.[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Authenticated with {label}[/green]")


async def _login_through_the_service(provider: str, *, mode: str, interaction: ConsoleInteraction) -> dict[str, Any]:
    """pi's own flow, run in the model service, its steps printed as they arrive.

    The service is configured first, from this config: that is what hands it the
    credential store the grant has to land in, and it is the only way a sign-in
    survives the command that made it.

    The provider goes over as it was typed. It is a pi provider id on both sides
    of the pipe -- the config files it under that id and pi's own auth map is
    keyed by it -- so the translation step this used to make was a chance for the
    two to disagree and nothing else.
    """
    from opendde_harness.config.loader import load_config
    from opendde_harness.providers.pi_service import get_service, shutdown_service

    try:
        service = await get_service(load_config())
        result: dict[str, Any] = {}
        async for step in service.login(provider, mode=mode):
            if step.get("type") == "login_prompt":
                _show_login_event(step, interaction)
            else:
                result = step
        return result
    finally:
        # This command owns the process, so the child goes with it rather than
        # being left for the atexit signal.
        await shutdown_service()


def _help_requested(extra_args: list[str]) -> bool:
    """Detect ``--help`` / ``-h`` inside a free-form ``ctx.args`` list."""
    return any(t in ("--help", "-h") or t.startswith("--help=") for t in extra_args)


def _refuse_unnameable(provider: str) -> None:
    """Stop on a name that cannot be a providers key, before anything is shown.

    Two cases, each worth its own sentence. A product this project removed is
    gone whatever is written for it. A near-miss for a pi id would be read as a
    provider this config declares and then refused for having no address, which
    is a true sentence about the wrong problem.

    The write paths refuse both as well; this is here because the read-only
    surfaces -- ``show``, ``set --help`` -- describe one entry schema for every
    provider, so without it they would answer a question about a provider that
    does not exist.
    """
    removed = pi_ids.removed_message(provider)
    if removed:
        console.print(f"[red]✗[/red] {removed}")
        raise typer.Exit(1)
    meant = pi_ids.suggestion(provider)
    if meant:
        console.print(f"[red]✗[/red] {provider!r} is not a pi provider id -- {meant!r} is the one that reaches it.")
        console.print(f"  [dim]Run the same command with [cyan]{meant}[/cyan].[/dim]")
        raise typer.Exit(1)


def _reason(exc: Exception) -> str:
    """The sentence a refusal was raised with, without the exception's own quotes.

    The write paths raise ``KeyError`` for a name or a field no entry can hold,
    and its message is a whole sentence -- which ``str()`` then wraps in quotes,
    because a ``KeyError``'s argument is normally a key. Printed that way the
    user reads a quoted fragment rather than the sentence.
    """
    return str(exc.args[0]) if exc.args else str(exc)


def _rejection(exc: Exception) -> str:
    """Why a write was refused, as lines that name keys and never values.

    A ``ValidationError``'s readable form renders an ``input_value=`` for every
    error, and the rejected input here is a providers entry holding the user's
    key. So the errors are formatted from their own location and message, and
    anything else -- the ``ValueError`` the write path raises when a write would
    leave the section unloadable, a ``KeyError`` naming a field -- prints its
    sentence.
    """
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        return "\n".join(
            f"  {'.'.join(str(part) for part in error.get('loc') or ()) or 'providers'}: "
            f"{str(error.get('msg') or '').removeprefix('Value error, ')}"
            for error in exc.errors()
        )
    return _reason(exc)


def _display_name(provider: str) -> str:
    """What to call this provider on screen: its own name, else pi's, else the id.

    The entry's ``name`` is the only part of this that is not pi's to answer,
    and a provider being signed in to may have no entry yet -- so a missing or
    unreadable config leaves pi's name rather than failing a report.
    """
    declared = ""
    try:
        from opendde_harness.config.update_providers import get_provider_config

        declared = str(get_provider_config(provider).get("name") or "")
    except Exception:  # noqa: BLE001 - a report never fails on the config it reports about
        declared = ""
    return pi_ids.display_name(provider, declared)


def _credential_store_path() -> str:
    """Where stored sign-ins live, for a message about a damaged one.

    Named rather than described: the file a person has to move aside is one they
    have never had a reason to know the location of, and "your credential file"
    is not a path. One file holds every provider's -- pi's credential store.
    """
    try:
        from opendde_harness.providers.pi_service import credential_store_path

        return str(credential_store_path())
    except Exception:
        # A report never fails on the thing it is reporting about.
        return ""


def _has_entry(provider: str) -> bool:
    """Does the config hold an entry for this provider at all?

    ``list_providers`` is the configured set, so membership in it is the
    question. Asked where the difference matters: an entry schema reads the same
    whether or not one was ever written, and "removed" is a false report about a
    provider that was never there.
    """
    from opendde_harness.config.update_providers import list_providers

    try:
        return any(row["name"] == provider for row in list_providers())
    except Exception:  # noqa: BLE001 - an unreadable config simply holds no entry we can name
        return False


def _print_schema_table(name: str) -> None:
    """Render the provider entry's field-spec table.

    Shared by ``show``, ``set --help`` interception, and the empty-flag
    fallback in ``set``. One schema serves every provider now, so what differs
    between two of them is only which fields their route reads -- which is why
    the name is still asked for.
    """
    from opendde_harness.config.update_providers import provider_field_specs, unread_fields

    _refuse_unnameable(name)
    specs = provider_field_specs()

    # A flag this provider's route never reads is not a flag. Listed, they read
    # as the way to configure a family that is configured by signing in, or as
    # an invitation to override an address pi already carries.
    unread = {path: why for path, why in unread_fields(name).items() if path in specs}
    specs = {path: spec for path, spec in specs.items() if path not in unread}

    table = Table(title=f"Provider: {name}")
    table.add_column("Flag", style="cyan", no_wrap=True)
    table.add_column("Type", overflow="fold")
    table.add_column("Default", no_wrap=True)
    table.add_column("Secret?", no_wrap=True, justify="center")
    table.add_column("Description", overflow="fold")
    for path, spec in specs.items():
        flag = "--" + path.replace("_", "-")
        default = spec["default"]
        default_str = "" if default in (None, "", [], {}) else str(default)
        table.add_row(
            flag,
            spec["type"],
            default_str,
            "✓" if spec["is_secret"] else "",
            spec.get("description", "") or "",
        )
    console.print(table)
    for path, why in unread.items():
        console.print(f"[dim]--{path.replace('_', '-')} is not read for this provider: {why}.[/dim]")


def _parse_provider_flags(extra_args: list[str], provider_name: str) -> dict[str, Any]:
    """Parse arbitrary ``--flag value`` pairs against the provider entry schema.

    Mirrors ``_parse_channel_flags`` (opendde_harness/cli/channel_commands.py:109) — the
    same six forms supported there work here:

    - ``--api-key abc``     -> ``{"api_key": "abc"}``
    - ``--api-key=abc``     -> ``{"api_key": "abc"}``
    - ``--base-url X``      -> ``{"base_url": "X"}``     (kebab -> snake)
    - ``--<flag> true``     -> ``{"<flag>": "true"}``     (string; the schema coerces)
    - ``--no-<flag>``       -> ``{"<flag>": False}``      (bool negative)
    - ``--<flag>`` alone    -> ``{"<flag>": True}``       (bool positive)

    Values come back as written; only the two valueless forms produce a bool
    here, and the schema coerces the rest on validation. The bool forms are
    named generically because no field of a provider entry is a bool today --
    the one that was (Gemini's ``vertex``) described a mechanism that never
    existed and was removed. They stay, matching ``_parse_channel_flags``, so a
    field gaining one needs no parser change.

    The flags ARE the entry's fields: one reflected schema for every provider,
    so ``--api-key``, ``--base-url``, ``--api``, ``--headers``, ``--models``,
    ``--login`` and ``--name`` are here because ``ProviderEntry`` declares them
    and for no other reason.

    Unknown fields raise ``typer.BadParameter`` pointing at ``provider show``.
    """
    from opendde_harness.config.update_providers import provider_field_specs

    specs = provider_field_specs()

    def _normalize(flag: str) -> str:
        return ".".join(seg.replace("-", "_") for seg in flag.split("."))

    out: dict[str, Any] = {}
    i = 0
    while i < len(extra_args):
        tok = extra_args[i]
        if not tok.startswith("--"):
            raise typer.BadParameter(f"Expected --flag, got: {tok}")

        if "=" in tok:
            flag, value = tok[2:].split("=", 1)
            i += 1
        else:
            flag = tok[2:]
            nxt = extra_args[i + 1] if i + 1 < len(extra_args) else None
            if nxt is not None and not nxt.startswith("--"):
                value = nxt
                i += 2
            else:
                value = None
                i += 1

        if flag.startswith("no-") and value is None:
            key = _normalize(flag[3:])
            if key not in specs:
                raise typer.BadParameter(
                    f"Unknown field '--no-{flag[3:]}'. Run 'ddeharness provider show {provider_name}' for available flags."
                )
            out[key] = False
            continue

        key = _normalize(flag)
        if key not in specs:
            raise typer.BadParameter(
                f"Unknown field '--{flag}' for provider '{provider_name}'. "
                f"Run 'ddeharness provider show {provider_name}' for available flags."
            )

        if value is None:
            if specs[key]["type"] == "bool":
                out[key] = True
            else:
                raise typer.BadParameter(f"Missing value for --{flag}")
        else:
            out[key] = value

    return out


#: A few of pi's built-in ids, for somebody who has nothing configured yet. The
#: full set is forty entries long and choosing from it is what the wizard is
#: for, so these are the handful people arrive already holding a key for.
_SAMPLE_BUILTINS = ("anthropic", "openai", "openai-codex", "google", "openrouter", "deepseek")


def _register_config_commands(app: typer.Typer) -> None:
    """Attach config subcommands to ``provider_app``."""
    app.info.no_args_is_help = True

    @app.command("list")
    def provider_list_cmd():
        """Show every provider this config configures, and whether it is usable."""
        from opendde_harness.config.update_providers import list_providers

        rows = list_providers()
        if not rows:
            # Empty is the normal state of a fresh install, not a fault: the
            # listing is the configured set, and pi's built-in ids are offered by
            # the wizard rather than printed here -- forty rows of "not set" is
            # not a status report.
            console.print("[yellow]No providers are configured yet.[/yellow]")
            console.print(
                "  Run [cyan]ddeharness onboard[/cyan] to pick one -- that is where pi's built-in providers "
                "are offered."
            )
            console.print(
                f"  [dim]Already have a key? [cyan]ddeharness provider set <id> --api-key <key>[/cyan] "
                f"-- ids such as {', '.join(_SAMPLE_BUILTINS)}.[/dim]"
            )
            return

        table = Table(title="LLM Providers")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Display", style="dim", overflow="fold")
        table.add_column("Type", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Base URL", overflow="fold")
        damaged: list[str] = []
        for p in rows:
            if p["is_oauth"]:
                type_str = "OAuth"
            elif p["base_url"]:
                # An address is what makes an entry a declaration, whether or not
                # pi ships the id: an Azure resource declared under a built-in id
                # carries its own deployment names and replaces pi's own for it.
                type_str = "Declared"
            elif p["is_builtin"]:
                type_str = "API key"
            else:
                # A key pi does not ship can only have been meant as a
                # declaration; the Status column says what it still needs.
                type_str = "Declared"
            if p["configured"]:
                status = "[green]✓ configured[/green]"
            elif p.get("credential_state") == "invalid":
                # Present and unreadable is not signed out, and the two are
                # fixed differently: one is a sign-in, the other needs the file
                # moved aside first.
                damaged.append(p["name"])
                status = "[red]✗ unreadable[/red]"
            else:
                status = "[dim]not set[/dim]"
            table.add_row(
                p["name"],
                p["display_name"],
                type_str,
                status,
                p.get("base_url") or "",
            )
        console.print(table)
        if damaged:
            path = _credential_store_path()
            for name in damaged:
                console.print(
                    f"[yellow]{name}[/yellow]: the stored credential cannot be read"
                    + (f" ({path})" if path else "")
                    + f" -- move it aside, then run [cyan]ddeharness provider login {name}[/cyan]."
                )
        console.print()
        console.print(
            "[dim]Use the [cyan]Name[/cyan] column with "
            "'provider show/set/get <id>'. "
            "Run 'provider show <id>' to see configurable fields, or "
            "'ddeharness onboard' to add another provider.[/dim]"
        )

    @app.command("get")
    def provider_get_cmd(
        name: str = typer.Argument(..., help="Provider id (e.g. openrouter)"),
        show_secrets: bool = typer.Option(False, "--show-secrets", help="Show secret values in plaintext (dangerous)"),
    ):
        """Print current configuration for a provider. Secrets redacted by default."""
        from opendde_harness.config.update_providers import get_provider_config

        try:
            cfg = get_provider_config(name, redact_secrets=not show_secrets)
        except KeyError as exc:
            console.print(f"[red]✗[/red] {_reason(exc)}")
            raise typer.Exit(1)

        table = Table(title=f"Provider: {name}")
        table.add_column("Flag", style="cyan", no_wrap=True)
        table.add_column("Value", overflow="fold")
        for k, v in cfg.items():
            flag = "--" + k.replace("_", "-")
            if v in ("", None, [], {}):
                display = "[dim](empty)[/dim]"
            else:
                display = str(v)
            table.add_row(flag, display)
        console.print(table)

    @app.command(
        "set",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def provider_set_cmd(
        ctx: typer.Context,
        name: str = typer.Argument(..., help="Provider id"),
    ):
        """Patch provider fields using ``--flag value`` syntax.

        The flags are the entry's own fields -- ``--api-key``, ``--base-url``,
        ``--api``, ``--headers``, ``--models``, ``--login``, ``--name`` -- and
        writing one is how a provider is configured: there is no empty block
        waiting to be filled in. ``--base-url`` and ``--api`` go together, since
        an address needs the protocol it serves and a protocol is only
        meaningful for an address.

        Examples:

            ddeharness provider set openrouter --api-key sk-or-v1-...
            ddeharness provider set my-vllm --base-url http://host:8000/v1 --api openai-completions
            ddeharness provider set my-vllm --models qwen3-32b,qwen3-8b
            ddeharness provider set relay --headers '{"X-Tenant": "acme"}'
        """
        if _help_requested(ctx.args):
            _print_schema_table(name)
            raise typer.Exit(0)

        from opendde_harness.config.update_providers import set_provider_fields

        fields = _parse_provider_flags(ctx.args, name)
        if not fields:
            _print_schema_table(name)
            console.print("  [dim]Tip: re-run with one or more --flag value pairs to update.[/dim]")
            raise typer.Exit(0)

        try:
            prev = set_provider_fields(name, fields)
        except (KeyError, RuntimeError) as exc:
            console.print(f"[red]✗[/red] {_reason(exc)}")
            raise typer.Exit(1)
        except ValueError as exc:
            # ``ValidationError`` is a ``ValueError``, and so is the refusal the
            # write path raises when an entry would leave the whole section
            # unloadable -- the sentence naming the missing baseUrl and api.
            # Both are the same answer to the user and neither reaches the file.
            console.print(f"[red]✗ Refused:[/red]\n{_rejection(exc)}")
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] {name} updated: {', '.join(prev)}")
        console.print(f"  [dim]Run 'ddeharness provider test {name}' to verify the credentials.[/dim]")

    @app.command("test")
    def provider_test_cmd(
        name: str = typer.Argument(..., help="Provider id"),
        timeout: int = typer.Option(60, "--timeout", "-t", help="Timeout seconds"),
    ):
        """Ask this provider for a few tokens, the way a turn would.

        Two questions, both through the model service: whether it resolves a
        credential for the entry at all, then whether one of the models it serves
        answers. It costs a handful of tokens. The free metadata ping it replaces
        was cheaper and answered a different question -- seven vendors publish no
        such route, Azure serves it elsewhere, and none of them says whether the
        model will run.
        """
        from opendde_harness.config.update_providers import test_provider as probe

        console.print(f"[dim]Asking {name} for a few tokens ...[/dim]")
        result = probe(name, timeout_s=timeout)

        if result["ok"]:
            console.print(
                f"[green]✓[/green] {name} OK ([dim]{result['model']} answered in {result['elapsed_ms']}ms[/dim])"
            )
            return

        hints = {
            "not_configured": f"Run: ddeharness provider set {name} --api-key <KEY>",
            "not_served": f"Run: ddeharness provider show {name} to see what the entry still needs",
            "no_model": f"Run: ddeharness provider set {name} --models <model-id>",
            "auth": f"The credential was refused. Run: ddeharness provider set {name} --api-key <NEW-KEY>",
            "oauth": f"The stored sign-in could not be renewed. Run: ddeharness provider login {name}",
            "model_not_found": "The service does not serve that model for this provider",
            "timeout": "No answer in time -- check network / firewall / VPN, or raise --timeout",
            "no_node": "The model service needs Node. Run: ddeharness doctor",
            "no_bundle": "The model service is not built. Run: npm run build in ui-tui/",
            "unknown_provider": "Run: ddeharness provider list to see the ids this config holds",
            "config_unreadable": "Run: ddeharness doctor to see what the config file is refused for",
        }
        console.print(f"[red]✗[/red] {name} failed: {result['status']}")
        hint = hints.get(result["status"], "")
        if hint:
            console.print(f"  [dim]{hint}[/dim]")
        if result.get("error"):
            console.print(f"  [dim]Detail: {result['error']}[/dim]")
        raise typer.Exit(1)

    @app.command("use")
    def provider_use_cmd(
        model: str = typer.Argument(..., help="Model id, e.g. anthropic/claude-sonnet-5"),
        provider: str = typer.Option("", "--provider", "-p", help="Provider serving it, when the id does not say"),
    ):
        """Make this the model the agent runs on.

        Changing it used to mean re-running the whole wizard: the TUI picker and
        onboarding could both switch models and the CLI could not, so a user on a
        headless box had six setup steps to walk to change one field.

        The id is stored qualified -- ``<provider>/<model>`` -- because that
        prefix is the only thing that names the provider serving it. There is no
        separate field to pin one any more, so nothing can disagree with the id
        about what was chosen.
        """
        from opendde_harness.config.loader import load_config
        from opendde_harness.config.update import set_default_model
        from opendde_harness.providers import model_id
        from opendde_harness.providers.auth import credential_status

        named = model_id.provider_of(model)
        asked = provider.strip()
        chosen = asked or named
        if not chosen:
            console.print(f"[red]✗[/red] cannot tell which provider serves {model!r}.")
            console.print(f"  [dim]Write it as <provider>/{model}, or pass --provider.[/dim]")
            raise typer.Exit(1)
        if named and asked and named != asked:
            # Two answers to one question. Picking either would write a default
            # whose prefix sends the model to the other one's credential, so
            # neither is written and the user says which they meant.
            console.print(f"[red]✗[/red] {model!r} names {named}, but --provider says {asked}.")
            console.print("  [dim]Drop one of the two.[/dim]")
            raise typer.Exit(1)
        _refuse_unnameable(chosen)

        previous = set_default_model(model, provider=chosen)
        stored = model_id.join(chosen, model)

        # Read back rather than trust the write: the row that names this model
        # and the credential that has to serve it both live in the file this
        # command has just changed.
        try:
            config = load_config()
        except Exception:  # noqa: BLE001 - the write happened; this is only the report
            config = None

        # The entry's own row is the only thing that names a model: the bundled
        # catalogue that used to supply labels went with the Python routes, and
        # the model service answers for a configured model, not for one being
        # configured. Unlabelled, the id says it.
        row = model_id.row_for(config.providers, stored) if config is not None else None
        declared = (row.name if row else "") or ""
        label = f"{declared} ([dim]{stored}[/dim])" if declared else stored
        console.print(f"[green]✓[/green] default model: {label}")
        if previous and previous != stored:
            console.print(f"  [dim]was {previous}[/dim]")

        # Reported rather than refused: choosing a model before configuring its
        # provider is a normal order to do things in, and the startup gate says
        # the same thing again if it is still missing then.
        if config is not None:
            status = credential_status(chosen, config.providers.get(chosen), include_external=True)
            if not status.ok:
                console.print(f"  [yellow]![/yellow] {status.summary}")

    @app.command("reset")
    def provider_reset_cmd(
        name: str = typer.Argument(..., help="Provider id"),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    ):
        """Remove a provider's entry from the config.

        Removal rather than a rewrite to defaults: an entry exists because
        somebody wrote it, and one emptied of its address and key is not a
        provider with nothing configured -- for a declared provider it is not
        even a valid entry. When the entry named a sign-in, the stored grant goes
        with it, so the provider is signed out as well.
        """
        from opendde_harness.config.update_providers import (
            get_provider_config,
            reset_provider,
            serves_default_model,
        )
        from opendde_harness.providers.auth import CRED_ENDPOINT, CRED_OAUTH, credential_kind

        try:
            current = get_provider_config(name, redact_secrets=False)
        except KeyError as exc:
            console.print(f"[red]✗[/red] {_reason(exc)}")
            raise typer.Exit(1)

        # An entry schema reads the same whether or not one was ever written, so
        # without this the command would report having removed a provider the
        # config never held.
        if not _has_entry(name):
            console.print(f"[yellow]{name} has no entry to remove.[/yellow]")
            console.print("  [dim]`ddeharness provider list` shows the providers this config holds.[/dim]")
            raise typer.Exit(1)

        set_values = [k for k, v in current.items() if v not in (False, "", None, [], {})]
        # Asked of the entry while it is still there: after the removal there is
        # nothing left to say which kind it was, and the way back differs per
        # kind. `provider login` exits 1 for anyone who is not a sign-in family,
        # and a declared provider has an address to give rather than a key.
        kind = credential_kind(name, current)

        # Asked before the confirmation, because it is the part worth confirming:
        # the model id survives the removal and still names this provider, so the
        # next command finds a default nothing can answer.
        serves_default = serves_default_model(name)

        if not yes:
            console.print(f"This will remove the [cyan]{name}[/cyan] entry from your config.")
            if set_values:
                preview = ", ".join(set_values[:5])
                more = f" (+{len(set_values) - 5} more)" if len(set_values) > 5 else ""
                console.print(f"  Currently set: [yellow]{preview}{more}[/yellow]")
            if kind == CRED_OAUTH:
                console.print("  [yellow]The stored sign-in is forgotten too[/yellow] -- you sign in again to use it.")
            if serves_default:
                console.print("  [yellow]This provider serves your current default model[/yellow] -- pick another")
                console.print("  [dim]afterwards with /model in the TUI, or configure it again.[/dim]")
            if not typer.confirm("Continue?", default=False):
                console.print("[yellow]Aborted.[/yellow]")
                raise typer.Exit(0)

        reset_provider(name)
        console.print(f"[green]✓[/green] {name} removed from the providers section")
        if serves_default:
            if kind == CRED_OAUTH:
                back = f"ddeharness provider login {name}"
            elif kind == CRED_ENDPOINT:
                back = f"ddeharness provider set {name} --base-url <URL> --api <API>"
            else:
                back = f"ddeharness provider set {name} --api-key <KEY>"
            console.print("  [yellow]Your default model is served by this provider and no longer works.[/yellow]")
            console.print(f"  [dim]Pick another with /model in the TUI, or set it up again: {back}[/dim]")

    @app.command("show")
    def provider_show_cmd(
        name: str = typer.Argument(..., help="Provider id to describe"),
    ):
        """Show available ``--flag`` fields for a provider (reflection-driven)."""
        _print_schema_table(name)


_register_config_commands(provider_app)


model_app = typer.Typer(
    help=(
        "Describe one model of a provider: the api it is served on (--api), its "
        "context window and output ceiling, a name, a price. Each flag patches "
        "one field of that model's row in providers.<id>.models; the rest keep "
        "their values."
    )
)


@model_app.command("set")
def model_set_cmd(
    provider: str = typer.Argument(..., help="Provider id"),
    model: str = typer.Argument(..., help="Model id, as the endpoint itself serves it"),
    api: str = typer.Option(
        "", "--api", help=f"Wire for this one model, over the provider's own: {', '.join(pi_ids.APIS)}"
    ),
    context_window: int = typer.Option(None, "--context-window", help="Context window in tokens"),
    max_tokens: int = typer.Option(None, "--max-tokens", help="Output ceiling in tokens"),
    qwen_fast: bool = typer.Option(
        False,
        "--qwen-fast",
        help="Disable Qwen hybrid thinking on the wire and default to a 2048-token output ceiling.",
    ),
    thinking_format: str = typer.Option(
        "", "--thinking-format", help="Qwen wire format: qwen (DashScope) or qwen-chat-template (vLLM-compatible)."
    ),
    reasoning: bool = typer.Option(
        None, "--reasoning/--no-reasoning", help="Whether this model thinks before it answers"
    ),
    reasoning_effort: str = typer.Option(
        "", "--reasoning-effort", help="Thinking level for this model: off, minimal, low, medium, high, xhigh, max"
    ),
    temperature: float = typer.Option(None, "--temperature", help="Sampling temperature for this model"),
    name: str = typer.Option("", "--name", help="Display name in the picker"),
    description: str = typer.Option("", "--description", help="One-line description in the picker"),
    catalog_model: str = typer.Option(
        "", "--catalog-model", help="Catalogue model this deployment serves, e.g. openai/gpt-4o"
    ),
    cost: str = typer.Option(
        "", "--cost", help='Per-million rates as JSON: \'{"input": 3, "output": 15}\' (also cacheRead, cacheWrite)'
    ),
):
    """Declare what you know about one model that no catalogue carries.

    Examples:

        ddeharness provider model set my-vllm qwen3-32b --context-window 131072 --max-tokens 32768
        ddeharness provider model set relay gpt-5.6-terra --api openai-responses --reasoning
        ddeharness provider model set azure-openai-responses prod --catalog-model openai/gpt-4o
        ddeharness provider model set openai-codex gpt-5.6-luna --reasoning-effort high
    """
    from opendde_harness.config.update_providers import set_model_row

    fields: dict[str, Any] = {}
    if api:
        fields["api"] = api
    # ``is not None`` rather than truthiness for the three that have a
    # meaningful zero: a temperature of 0 is a real setting, and a rejected 0 for
    # the two limits should be reported as the refusal it is rather than read as
    # "not passed".
    if context_window is not None:
        fields["context_window"] = context_window
    if max_tokens is not None:
        fields["max_tokens"] = max_tokens
    if reasoning is not None:
        fields["reasoning"] = reasoning
    if reasoning_effort:
        fields["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        fields["temperature"] = temperature
    if name:
        fields["name"] = name
    if description:
        fields["description"] = description
    if catalog_model:
        fields["catalog_model"] = catalog_model
    if cost:
        # A price is four numbers, so it arrives as the object pi's row holds
        # rather than as four flags nobody would pass together.
        try:
            fields["cost"] = json.loads(cost)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"--cost is not valid JSON: {exc}")
    if qwen_fast or thinking_format:
        from opendde_harness.config.loader import load_config

        entry = load_config().providers.get(provider)
        if entry is None:
            raise typer.BadParameter("configure the provider before setting its Qwen model profile")
        row = entry.row(model)
        format_name = thinking_format or "qwen"
        if format_name not in {"qwen", "qwen-chat-template"}:
            raise typer.BadParameter("--thinking-format must be qwen or qwen-chat-template")
        wire = api or (row.api if row else None) or entry.api or "openai-completions"
        if wire != "openai-completions":
            raise typer.BadParameter("Qwen thinking controls require an openai-completions endpoint")
        fields["api"] = wire
        fields["compat"] = {**((row.compat or {}) if row else {}), "thinkingFormat": format_name}
        if qwen_fast:
            if reasoning is False or reasoning_effort not in {"", "off"}:
                raise typer.BadParameter(
                    "--qwen-fast cannot be combined with --no-reasoning or non-off effort; reasoning=true records capability while off disables thinking"
                )
            fields.update(
                reasoning=True, reasoning_effort="off", max_tokens=max_tokens if max_tokens is not None else 2048
            )
    if not fields:
        raise typer.BadParameter(
            "pass at least one of --api, --context-window, --max-tokens, --reasoning/--no-reasoning, "
            "--reasoning-effort, --temperature, --name, --description, --catalog-model, --cost"
        )

    try:
        row = set_model_row(provider, model, fields)
    except KeyError as exc:
        console.print(f"[red]✗[/red] {_reason(exc)}")
        raise typer.Exit(1)
    except ValueError as exc:
        console.print(f"[red]✗ Refused:[/red]\n{_rejection(exc)}")
        raise typer.Exit(1)

    # The row as stored, in the spelling the file holds it in, so what is printed
    # is what a later read will find.
    console.print(f"[green]✓[/green] {provider} / {row.get('id', model)}: {json.dumps(row)}")


provider_app.add_typer(model_app, name="model")


__all__ = ["provider_app"]
