"""``cli.dispatch`` RPC handler — runs an EC CLI command in-process.

``argv`` + ``width`` + an optional ``timeout_s`` in, ``{stdout, stderr,
exit_code}`` out: any command Typer registers that is not in
``_DISPATCH_BLACKLIST`` is accepted, anything else is refused with -32015, and a
non-zero exit is reported in ``exit_code`` rather than as an error frame. Click
runs with ``standalone_mode=False`` so ``typer.Exit`` comes back as a return
value, and output is rendered through a Rich ``Console`` at the caller's width
then stripped of ANSI so the TUI's reconciler is not corrupted. Nothing can
interrupt the Typer app once its thread is running, so ``_dispatch_lock`` and
the stdout/console redirections belong to the command rather than to the
awaiting call: a timeout fails the RPC at once but hands both to ``_reap``,
which releases them only when the thread exits. A dispatch therefore waits for
the lock no longer than its own ``timeout_s``.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import io
import threading
from typing import TYPE_CHECKING

import click
from loguru import logger
from pydantic import ValidationError
from rich.console import Console

import opendde_harness.cli.commands as ec_cli
from opendde_harness.tui_rpc._ansi_filter import filter_ansi
from opendde_harness.tui_rpc._confirm_injection import confirm_injection
from opendde_harness.tui_rpc._console_injection import inject_consoles
from opendde_harness.tui_rpc.confirm_broker import _CONFIRM_HARD_LIMIT_S
from opendde_harness.tui_rpc.errors import (
    CliCommandTimeoutError,
    ConfigValidationError,
    NotDispatchCompatibleError,
)
from opendde_harness.tui_rpc.methods._typer_reflect import collect_command_names as _collect_command_names
from opendde_harness.tui_rpc.models import CliDispatchParams

if TYPE_CHECKING:
    from opendde_harness.tui_rpc.confirm_broker import ConfirmBroker
    from opendde_harness.tui_rpc.dispatcher import Dispatcher


# ---------------------------------------------------------------------------
# Blacklist — hard-reject prefixes (single source of truth for both
# cli.dispatch and the dynamic commands.catalog filter).
# ---------------------------------------------------------------------------
# These commands either run indefinitely, hijack stdin, or need a real TTY
# (browser/OAuth/QR). cli.dispatch entry MUST reject them up front with
# -32015 ``not_dispatch_compatible`` so the TUI surfaces a friendly toast
# guiding the user to run the command in their own terminal. The
# harness-command-catalog-dynamic L2 also imports this set from
# ``methods/commands.py`` so the catalog handler filters the same prefixes
# (preventing user-facing slash entries that would only get rejected by
# dispatch later).
#
# Source: CLI-team locked P3 5 commands @
# ``docs/sendbox/toTuiIpcBridge/from-orche-eve15-cli-team-36-commands-locked.md``
# + harness-command-catalog-dynamic extension (``tui`` + ``onboard``) per
# ``docs/openspec/changes/harness-command-catalog-dynamic/design.md §D4``.
_DISPATCH_BLACKLIST: set[tuple[str, ...]] = {
    ("provider", "login"),  # OAuth flow requires browser
    # tui and onboard were unreachable under the old _DISPATCH_WHITELIST.
    # Reflection makes them and the registered upgrade command reachable, so
    # the blacklist hard-rejects all three to preserve safety.
    ("tui",),  # recursive Ink+Node spawn would deadlock + steal stdin
    ("onboard",),  # prompt_toolkit three-step wizard hijacks stdin
    ("upgrade",),  # replacing the active OpenDDE Harness process is terminal-only
}


#: What a command needs instead of the default, where the default is not
#: enough. Declared here, beside the blacklist, because both answer the same
#: question -- what a dispatch can host -- and a client that decided this by
#: name would have to be updated whenever the CLI changes.
_DISPATCH_TIMEOUTS: dict[tuple[str, ...], float] = {}


def dispatch_timeout(argv: "list[str] | tuple[str, ...]") -> float | None:
    """Seconds this command needs, or None to use the caller's default."""
    argv = tuple(argv)
    for prefix, seconds in _DISPATCH_TIMEOUTS.items():
        plen = len(prefix)
        if len(argv) >= plen and argv[:plen] == prefix:
            return seconds
    return None


# Default timeout if caller did not override.
_DEFAULT_TIMEOUT_S = 30.0

# Serializes concurrent dispatches so the monkey-patched module-level
# ``console`` references can't race.
_dispatch_lock = asyncio.Lock()


# asyncio holds only a weak reference to a task, so a reaper lives here until
# it finishes and drops itself.
_reapers: set[asyncio.Task] = set()


def _is_dispatch_compatible(argv: list[str]) -> bool:
    """Return True iff argv is dispatch-compatible.

    Algorithm (harness-command-catalog-dynamic design.md §D7.1):

    1. Empty argv → False.
    2. Blacklist hard reject (prefix match against ``_DISPATCH_BLACKLIST``).
    3. Reflect ``ec_cli.app`` to determine whether argv resolves to a
       registered Typer command:
       - ``argv[0]`` in top-level command names → True.
       - ``argv[0]`` in subgroup names: require ``argv[1]`` in that group's
         subcommand names (incomplete invocations like ``["provider"]``
         return False). Groups whose body is a single
         ``@callback(invoke_without_command=True)`` (no subcommands) accept
         the bare group head — but ``tui`` is the only such group in EC
         and is blacklisted, so this branch is effectively reserved for
         test fakes.

    Reflection runs every call (no module-level cache). The Typer
    registered_commands / registered_groups walks are dict-iteration on
    small lists (<50 entries), microsecond cost; caching would create
    invalidation surface that fights the slash_routing tests that
    monkeypatch ``ec_cli.app``.

    Examples:
        >>> _is_dispatch_compatible(["provider", "list"])
        True
        >>> _is_dispatch_compatible(["provider", "list", "--verbose"])  # prefix args
        True
        >>> _is_dispatch_compatible(["provider"])  # incomplete (group with subs)
        False
        >>> _is_dispatch_compatible(["nonexistent"])
        False
        >>> _is_dispatch_compatible([])
        False
        >>> _is_dispatch_compatible(["tui"])  # blacklist
        False
    """
    if not argv:
        return False
    # Blacklist check first (hard reject)
    for prefix in _DISPATCH_BLACKLIST:
        plen = len(prefix)
        if len(argv) >= plen and tuple(argv[:plen]) == prefix:
            return False
    # Reflection-based positive check (replaces the v0.0.2 hardcoded
    # _DISPATCH_WHITELIST 19-tuple set). The helper module also serves
    # ``methods/commands.py`` so the two reflection sites can't drift.
    head = argv[0]
    app = ec_cli.app
    if head in _collect_command_names(app):
        return True
    sub_app = _find_group(app, head)
    if sub_app is not None:
        sub_names = _collect_command_names(sub_app)
        if not sub_names:
            # Bare-group dispatch (e.g. ``tui`` uses @callback(invoke_without_command=True)).
            # In real EC ``tui`` is blacklisted so this branch fires only for
            # test fakes that register a Typer subgroup with no subcommands.
            return bool(sub_app.info.invoke_without_command)
        if len(argv) < 2:
            return False  # incomplete: group name alone with no subcommand
        return argv[1] in sub_names
    return False


def _find_group(app, name: str):
    """Return the sub-``typer.Typer`` instance for ``name``, or ``None``."""
    for ti in app.registered_groups:
        if ti.name == name:
            return ti.typer_instance
    return None


def _invoke_ec_cli(argv: list[str]) -> int:
    """Synchronous wrapper around the EC Typer app, used by ``asyncio.to_thread``.

    Returns the command exit code (0 on success). Click's ``standalone_mode=False``
    catches ``click.exceptions.Exit`` (Typer's ``typer.Exit``, a ``RuntimeError``
    subclass) and **returns** its ``exit_code`` from ``app()`` instead of raising,
    so we must capture the return value here. ``SystemExit`` (rare, e.g.
    ``sys.exit()`` from helper code) is also handled here for completeness.

    We resolve ``ec_cli.app`` at call time (not import time) so monkey-patches
    in tests take effect.
    """
    try:
        result = ec_cli.app(argv, standalone_mode=False)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    # ``result`` is the exit_code from typer.Exit/click.Exit when raised under
    # standalone_mode=False. Non-int returns (None for normal completion) → 0.
    return int(result) if isinstance(result, int) else 0


def _start_command_thread(argv: list[str]) -> "asyncio.Future[int]":
    """Run ``_invoke_ec_cli(argv)`` on a daemon thread; resolve a future with it.

    ``asyncio.to_thread`` would use the loop's default executor, which
    ``asyncio.run`` joins on the way out, so a command still running at gateway
    shutdown would hold the process open. The future is never cancelled by the
    awaiting dispatch: only the thread can end the command, and the reaper still
    needs its result.
    """
    loop = asyncio.get_running_loop()
    future: "asyncio.Future[int]" = loop.create_future()
    ctx = contextvars.copy_context()

    def _settle(value: int | None, error: BaseException | None) -> None:
        if future.done():
            # Cancelled along with the loop at shutdown — nothing to deliver.
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(int(value or 0))

    def _run() -> None:
        try:
            result = ctx.run(_invoke_ec_cli, argv)
        except BaseException as exc:  # noqa: BLE001 — handed to the awaiter
            value, error = None, exc
        else:
            value, error = result, None
        try:
            loop.call_soon_threadsafe(_settle, value, error)
        except RuntimeError:
            # Loop already closed (gateway shutdown): no one is left to tell.
            pass

    threading.Thread(target=_run, name="ec-cli-dispatch", daemon=True).start()
    return future


def _reap(argv: list[str], command: "asyncio.Future[int]", owned: contextlib.ExitStack) -> None:
    """Hold a walked-away-from command's ownership until its thread exits.

    ``owned`` carries the redirections and injections the command writes through
    and ``_dispatch_lock`` is still held on entry; both are released once the
    command is actually gone, so the next dispatch never shares the
    process-global console patch with one the client already gave up on. A
    gateway shutdown cancels the reaper and leaves them in place, which is
    right: the daemon thread may still be writing.
    """

    async def _reaper() -> None:
        await asyncio.wait({command})
        logger.warning("tui_rpc.cli.dispatch: abandoned command finished argv={!r}", argv)
        owned.close()
        _dispatch_lock.release()

    task = asyncio.ensure_future(_reaper())
    _reapers.add(task)
    task.add_done_callback(_reapers.discard)


async def cli_dispatch(params: dict, *, confirm_broker: "ConfirmBroker | None" = None) -> dict:
    """Run an EC CLI command in-process; return ``CliResult``-shaped dict.

    When ``confirm_broker`` is supplied (TUI production path), ``typer.confirm``
    / ``click.confirm`` are bridged to the broker so destructive confirms are
    answered over RPC instead of reading the EOF dispatch stdin. The
    dispatch timeout then includes a ``_CONFIRM_HARD_LIMIT_S`` grace window so a
    paused-on-confirm command is not killed mid-prompt (path B). Without a
    broker, ``typer.confirm`` keeps its native behavior and the timeout is
    unchanged.

    Raises:
        ConfigValidationError (-32011): params shape / range invalid.
        NotDispatchCompatibleError (-32015): argv not in whitelist.
        CliCommandTimeoutError (-32014): command exceeded ``timeout_s``, or
            never started because another command still holds the lock
            (``data.started`` tells the two apart).
    """
    # ----- Param validation (Pydantic enforces 20 ≤ width ≤ 500) -----------
    try:
        validated = CliDispatchParams.model_validate(params)
    except ValidationError as exc:
        raise ConfigValidationError(
            "cli.dispatch params invalid",
            data={"errors": exc.errors()},
        ) from exc

    argv = list(validated.argv)
    width = validated.width
    timeout_s = validated.timeout_s if validated.timeout_s is not None else _DEFAULT_TIMEOUT_S

    # ----- Whitelist (-32015) ----------------------------------------------
    if not _is_dispatch_compatible(argv):
        # DEBUG level: under slash.exec this fires on every unknown / blacklist
        # slash typed by the user (e.g. /asd, /provider login). It is normal
        # operation, not info-worthy chatter — and at INFO it corrupts the
        # Ink reconciler when stderr inheritance is on.
        logger.debug("tui_rpc.cli.dispatch: rejected non-compatible argv: {!r}", argv)
        raise NotDispatchCompatibleError(
            f"argv {argv!r} not in cli.dispatch whitelist",
            data={"argv": argv, "hint": "use native UI for this command"},
        )

    # ----- Render buffers + Rich Consoles ----------------------------------
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    out_console = Console(
        file=stdout_buf,
        force_terminal=True,
        color_system="truecolor",
        width=width,
    )
    err_console = Console(
        file=stderr_buf,
        force_terminal=True,
        color_system="truecolor",
        width=width,
    )

    exit_code = 0

    # When a confirm broker is present, bridge typer.confirm into it and grant
    # the timeout a confirm grace window (path B). loop is captured here (we are
    # on the event loop) so the worker thread can run_coroutine_threadsafe back.
    if confirm_broker is not None:
        loop = asyncio.get_running_loop()
        confirm_ctx: contextlib.AbstractContextManager = confirm_injection(confirm_broker, loop)
        effective_timeout = timeout_s + _CONFIRM_HARD_LIMIT_S
    else:
        confirm_ctx = contextlib.nullcontext()
        effective_timeout = timeout_s

    # ----- Lock + redirect + inject + invoke -------------------------------
    # The lock and the contexts below belong to the command, not to this await:
    # on timeout or cancellation they are handed to ``_reap`` instead of being
    # unwound under a thread that is still writing through them. ``owned`` is
    # that hand-off unit, entered in the same order the old nested ``with``
    # used. An abandoned command can hold the lock for as long as it keeps
    # running, so the wait for it gets this caller's own timeout: a queued
    # command must not outlive the deadline its caller declared.
    try:
        await asyncio.wait_for(_dispatch_lock.acquire(), timeout_s)
    except TimeoutError:  # asyncio.TimeoutError is this same class on 3.11+
        raise CliCommandTimeoutError(
            "another command is still running; try again when it finishes",
            data={"argv": argv, "timeout_s": timeout_s, "started": False},
        ) from None

    owned = contextlib.ExitStack()
    command: "asyncio.Future[int] | None" = None
    handed_off = False
    try:
        owned.enter_context(contextlib.redirect_stdout(stdout_buf))
        owned.enter_context(contextlib.redirect_stderr(stderr_buf))
        owned.enter_context(inject_consoles(out_console))
        owned.enter_context(confirm_ctx)
        command = _start_command_thread(argv)
        # ``asyncio.wait`` rather than ``wait_for``: a timeout must not cancel
        # the future, because the reaper still needs its result.
        done, _pending = await asyncio.wait({command}, timeout=effective_timeout)
        if not done:
            logger.warning(
                "tui_rpc.cli.dispatch: timeout after {}s argv={!r}; command still running, "
                "holding the dispatch lock and output capture until it exits",
                timeout_s,
                argv,
            )
            _reap(argv, command, owned)
            handed_off = True
            raise CliCommandTimeoutError(
                f"command exceeded {timeout_s}s timeout",
                data={"argv": argv, "timeout_s": timeout_s, "started": True},
            )
        try:
            # _invoke_ec_cli returns the command exit code. Click
            # under standalone_mode=False catches typer.Exit /
            # click.exceptions.Exit and returns the exit_code from
            # ``app()`` (not as raised exception). Critical for B1:
            # without capturing this return value, all typer.Exit(N)
            # paths silently report exit_code=0 to the TUI.
            exit_code = command.result()
        except click.exceptions.UsageError as exc:
            err_console.print(f"[red]Usage:[/] {exc.format_message()}")
            exit_code = 2
        except click.exceptions.Abort:
            # C1: a confirm hit the EOF dispatch stdin (no round-trip
            # available). Abort is a RuntimeError subclass, NOT a
            # ClickException, so without this it falls to the broad
            # catch below as a useless "Internal error: Abort".
            err_console.print("[yellow]This command needs confirmation; re-run with --yes.[/]")
            exit_code = 1
        except click.exceptions.ClickException as exc:
            err_console.print(f"[red]Error:[/] {exc.format_message()}")
            exit_code = 1
    except asyncio.CancelledError:
        # The caller went away (client disconnect / server teardown). Same rule
        # as the timeout: the command owns the contexts until its thread ends.
        if command is not None and not command.done():
            logger.warning(
                "tui_rpc.cli.dispatch: cancelled with the command still running argv={!r}",
                argv,
            )
            _reap(argv, command, owned)
            handed_off = True
        raise
    except (CliCommandTimeoutError, NotDispatchCompatibleError, ConfigValidationError):
        # Re-raise RPC errors unchanged (timeout above is the main path
        # here; the others can't actually be raised inside this block,
        # but defensively preserve them).
        raise
    except Exception as exc:  # noqa: BLE001 — last-resort catch
        logger.exception("tui_rpc.cli.dispatch: unexpected error in argv={!r}", argv)
        err_console.print(f"[red]Internal error:[/] {type(exc).__name__}: {exc}")
        exit_code = 1
    finally:
        if not handed_off:
            owned.close()
            _dispatch_lock.release()

    # ----- Apply ANSI filter and return ------------------------------------
    return {
        "stdout": filter_ansi(stdout_buf.getvalue()),
        "stderr": filter_ansi(stderr_buf.getvalue()),
        "exit_code": exit_code,
    }


def register_cli_methods(dispatcher: "Dispatcher", *, confirm_broker: "ConfirmBroker | None" = None) -> None:
    """Register ``cli.dispatch`` on a dispatcher instance.

    ``confirm_broker`` is pre-bound via a closure (mirrors the turn/emitter
    pattern) so the in-process confirm round-trip activates on the production
    path; when ``None`` (demo runner / tests) dispatch keeps native confirm.
    """

    async def _dispatch(params: dict) -> dict:
        return await cli_dispatch(params, confirm_broker=confirm_broker)

    dispatcher.register("cli.dispatch", _dispatch)


__all__ = [
    "cli_dispatch",
    "register_cli_methods",
    "_is_dispatch_compatible",
    "_DISPATCH_BLACKLIST",
]
