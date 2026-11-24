"""Slash command routing handlers.

Wires four RPC methods that the fork-imported hermes UI invokes but were not
previously registered, causing dogfood failures:

* ``slash.exec`` — hermes routes any *unknown* slash here
  (``ui-tui/src/app/createSlashHandler.ts:82``) expecting
  ``{output?, warning?}``. We shlex-split the command, delegate to
  ``cli.dispatch``, and map the result. **Never raises -32xxx** — failures
  arrive as ``{output: "", warning: "..."}`` so the UI's ``.then()`` branch
  consumes them (the ``.catch()`` branch would fall through to
  ``command.dispatch`` which is also unregistered).
* ``session.status`` — ``cli.dispatch(["status"])`` for the install-wide
  picture, followed by the session's own token totals, cache hit rate and
  estimated cost from the loop's usage tracker.
* ``complete.slash`` / ``complete.path`` — return empty completion lists so
  ``useCompletion.ts`` stops surfacing the "completion unavailable" red-frame
  toast on every keystroke.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from typing import TYPE_CHECKING, Any

from loguru import logger

from opendde_harness.tui_rpc.errors import (
    CliCommandTimeoutError,
    ConfigValidationError,
    NotDispatchCompatibleError,
)
from opendde_harness.tui_rpc.methods.cli_dispatch import (
    _DISPATCH_BLACKLIST,
    cli_dispatch,
    dispatch_timeout,
)

if TYPE_CHECKING:
    from opendde_harness.token_wise.base import UsageSnapshot
    from opendde_harness.token_wise.usage_tracker import CallCounts, SessionUsage
    from opendde_harness.tui_rpc.confirm_broker import ConfirmBroker
    from opendde_harness.tui_rpc.dispatcher import Dispatcher
    from opendde_harness.tui_rpc.methods.session import AgentLoopFactory


_DEFAULT_WIDTH = 100
"""Default Rich console width when slash.exec is invoked from the hermes UI.

The hermes ``createSlashHandler`` does not propagate viewport width to
``slash.exec`` (it only sends ``{command, session_id}``). 100 cols is a sane
middle ground — wide tables can be revisited in v0.0.3 if a width hint is
added to the hermes slash handler.
"""

_SLASH_TIMEOUT_S = 20.0
"""Tighter than cli.dispatch's 30s default — slash commands are interactive
and a 20s ceiling keeps the UI responsive. A command that declares its own
(``_DISPATCH_TIMEOUTS``) gets that instead."""

#: Click's exit code for a command invoked wrongly.
_USAGE_EXIT = 2

#: What Click says when a required argument or option was not given.
_MISSING_INPUT = re.compile(r"missing (argument|option)", re.IGNORECASE)

#: Rich leaves SGR sequences in the captured stderr; a transcript line is text.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text).strip()


def _shape_missing_input(command: str, argv: list[str], stderr: str) -> dict[str, str]:
    """One line naming what to supply, instead of Click's usage block.

    ``/provider get`` answered with a red "Usage: Missing argument 'NAME'."
    over a "no output" tail, which is a stack of noise around one fact: it
    needs a name. The hint is the one the completion popup already shows, so
    the answer and the prompt agree.
    """
    from opendde_harness.tui_rpc.methods._typer_reflect import hint_for

    try:
        import opendde_harness.cli.commands as ec_commands

        hint = hint_for(ec_commands.app, argv)
    except Exception:  # pragma: no cover - reflection failure falls back to Click
        hint = ""

    if hint:
        return {"output": f"/{command} needs {hint}"}

    return {"output": _plain(stderr) or f"/{command} was called without something it needs"}


def _shape_unknown_command_warning(command: str) -> dict[str, str]:
    """Build a friendly message for an unknown / out-of-scope verb.

    We put the message in ``output`` (not ``warning``) on purpose: hermes's
    ``createSlashHandler.ts:88`` falls back to ``/<name>: no output`` when
    ``output`` is empty, so a ``warning``-only response renders as

        warning: unknown command: /asd …
        /asd: no output

    which has an ugly redundant trailing line. Putting the message in
    ``output`` collapses to a clean single line.
    """
    return {
        "output": f"unknown command: /{command} — try `/help` or run `ddeharness --help` in a terminal",
    }


def _shape_blacklist_warning(command: str) -> dict[str, str]:
    """Friendly message for P3 terminal-only commands (output, not warning).

    Same rationale as :func:`_shape_unknown_command_warning`.
    """
    return {
        "output": f"/{command} requires a real terminal — run it directly with `opendde {command}`",
    }


async def slash_exec(params: dict[str, Any], *, confirm_broker: "ConfirmBroker | None" = None) -> dict[str, Any]:
    """Route an unknown hermes slash to ``cli.dispatch``.

    Hermes UI invokes us with ``{command: "provider list", session_id: ...}``
    when the user types a slash that is not in hermes's local registry. We:

    1. shlex-split ``command`` into argv (preserves quoted args).
    2. Delegate to ``cli.dispatch`` with default width / tighter timeout.
    3. Map the result to ``SlashExecResponse`` ``{output, warning?}``:
       - whitelist + exit 0 → ``{output: stdout}``
       - whitelist + exit != 0 → ``{output: stdout, warning: stderr|exit_code}``
       - blacklist → toast-friendly "use a real terminal" warning
       - unknown verb → toast-friendly "unknown command" warning
       - timeout → the timeout message as a warning
       - empty / whitespace command → "empty slash command" warning

    Never raises -32xxx. Hermes's ``createSlashHandler.ts:83-92`` consumes the
    response via ``r?.output`` / ``r?.warning``.
    """
    raw_command = str(params.get("command", "")).strip()
    if not raw_command:
        # Empty / whitespace slash → friendly hint in output (no warning field
        # to avoid the createSlashHandler.ts:88 "/: no output" tail).
        return {"output": "(empty slash command — type /help for a list)"}

    try:
        argv = shlex.split(raw_command)
    except ValueError as exc:
        return {"output": f"could not parse command: {exc}"}

    if not argv:
        return {"output": "(empty slash command — type /help for a list)"}

    timeout_s = dispatch_timeout(argv) or _SLASH_TIMEOUT_S

    try:
        result = await cli_dispatch(
            {
                "argv": argv,
                "width": _DEFAULT_WIDTH,
                "timeout_s": timeout_s,
            },
            confirm_broker=confirm_broker,
        )
    except NotDispatchCompatibleError:
        # Either P3 blacklist (provider login / tui / onboard /
        # agent-REPL) or a verb not in the whitelist. We
        # distinguish by checking the well-known blacklist prefixes.
        if _is_blacklist_argv(argv):
            return _shape_blacklist_warning(raw_command)
        return _shape_unknown_command_warning(raw_command)
    except CliCommandTimeoutError as exc:
        return {"output": "", "warning": str(exc)}
    except ConfigValidationError as exc:
        logger.debug("slash.exec: cli.dispatch param validation failed: {}", exc)
        return {"output": f"invalid slash payload: {exc}"}

    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    exit_code = int(result.get("exit_code", 0) or 0)

    if exit_code == 0:
        return {"output": stdout}

    if exit_code == _USAGE_EXIT and _MISSING_INPUT.search(_plain(stderr)):
        return _shape_missing_input(raw_command, argv, stderr)

    # Non-zero exit — surface stderr (or a generic hint) as the warning so the
    # user can see what went wrong without losing partial stdout.
    warning = stderr.strip() or f"command exited with status {exit_code}"
    return {"output": stdout, "warning": warning}


def _is_blacklist_argv(argv: list[str]) -> bool:
    """Identify P3 blacklist hits so we can choose the right toast message.

    Reads ``_DISPATCH_BLACKLIST`` (cli_dispatch.py) as the single source of
    truth — keeps this module aligned with any future blacklist extension
    automatically. harness-command-catalog-dynamic added ``tui`` + ``onboard``
    to the set and the prior hardcoded variant here would have silently
    misclassified both as "unknown command" rather than the correct
    "requires a real terminal" toast.
    """
    if not argv:
        return False
    for prefix in _DISPATCH_BLACKLIST:
        plen = len(prefix)
        if len(argv) >= plen and tuple(argv[:plen]) == prefix:
            return True
    return False


def _usd(amount: float) -> str:
    """Three decimals, which is what the footer prints.

    The two surfaces report one figure and are read side by side, so they round
    it the same way: four decimals here against the footer's three had the same
    session showing ``$0.4376`` in one place and ``$0.438`` in the other, which
    is the disagreement this whole pass is about. Three is pi's own footer
    format, and the owner asked for that line to read as pi's.
    """
    return f"${amount:,.3f}"


def _cache_hit_rate(usage: "UsageSnapshot", counts: "CallCounts") -> str:
    """The rate over the calls the provider stated a figure for, and how many that was.

    A relay that forwards no cache fields on most calls would otherwise read
    as a near-zero rate over every token sent, which is a claim about the
    relay's reporting, not about the cache.
    """
    if counts.calls == 0:
        return "n/a (no calls yet)"
    if counts.cache_reported == 0:
        return f"not reported (none of the {counts.calls} calls carried a cache figure)"
    prompt = counts.cache_reported_prompt_tokens
    hit = usage.cache_read_tokens / prompt if prompt else 0.0
    counted = (
        "input tokens read from cache"
        if counts.cache_reported == counts.calls
        else f"input tokens read from cache on the {counts.cache_reported} of {counts.calls} calls that reported one"
    )
    return f"{hit:.1%} ({usage.cache_read_tokens:,} of {prompt:,} {counted} · {usage.cache_write_tokens:,} written)"


def session_usage_report(session_id: str, model: str, usage: "SessionUsage | None") -> str:
    """The session's token and cost totals, as the lines ``/status`` appends.

    ``usage`` is the tracker's copy of what the loop and the calls made on its
    turns' behalf (a server-side compaction, say) spent under this session key,
    including the turns that ran before this process: a resumed session's stored
    records are adopted into the tracker when it opens
    (``tui_rpc.methods.session._session_totals``), so this is the session's whole
    spend rather than the current process's share of it.

    Cost is the sum of each call priced at its model's list price when it
    landed -- the rates the model layer reported for the model that answered,
    long-context tiers included, the way pi prices each message. Priced by the
    loop as each call lands, so nothing here re-derives a rate; a model with no
    list price is named as unpriced instead of counted as free.
    """
    lines = [f"Session: {session_id}"]
    if model:
        lines.append(f"Model: {model}")
    if usage is None:
        lines.append("Usage: unavailable until the agent has started")
        return "\n".join(lines)

    totals = usage.totals
    prompt = totals.input_tokens + totals.cache_read_tokens + totals.cache_write_tokens
    total = prompt + totals.output_tokens
    lines.append(
        f"Tokens: {total:,} total · {prompt:,} input · {totals.output_tokens:,} output "
        f"(summed over {usage.counts.calls} calls; the context bar shows the last call's input plus output)"
    )
    lines.append(f"Cache hit rate: {_cache_hit_rate(totals, usage.counts)}")

    cost = sum(u.list_cost_usd for u in usage.by_model.values() if u.list_cost_usd is not None)
    unpriced = [name for name, u in usage.by_model.items() if u.list_cost_usd is None]
    if unpriced:
        cost_line = f"{_usd(cost)} at list price · no list price for {', '.join(unpriced)}"
        if len(unpriced) == len(usage.by_model):
            cost_line = f"unknown (no list price for {', '.join(unpriced)})"
    elif len(usage.by_model) == 1:
        cost_line = f"{_usd(cost)} at list price"
    else:
        cost_line = f"{_usd(cost)} at list price" + (f" across {len(usage.by_model)} models" if usage.by_model else "")
    lines.append(f"Estimated cost: {cost_line}")
    # The one sentence that keeps the two figures apart. They are different
    # questions -- what the session has cost, and how full the window is right
    # now -- and reading the second as the first is what sent the owner here.
    lines.append("(the footer shows this session's total at list price; its context bar shows the last call alone)")
    return "\n".join(lines)


async def session_status(
    params: dict[str, Any], *, agent_loop_factory: "AgentLoopFactory | None" = None
) -> dict[str, Any]:
    """``session.status``: ``ddeharness status`` plus this session's usage.

    The CLI part is the install-wide picture (config, workspace, providers);
    the appended block is what this session has spent, see
    :func:`session_usage_report`. Rendered via ``SessionStatusResponse {output}``.
    """
    from opendde_harness.tui_rpc.methods.session import _safe_invoke_factory

    result = await cli_dispatch(
        {
            "argv": ["status"],
            "width": _DEFAULT_WIDTH,
            "timeout_s": _SLASH_TIMEOUT_S,
        }
    )
    output = str(result.get("stdout", ""))
    session_id = str(params.get("session_id") or "")
    if session_id:
        agent_loop = _safe_invoke_factory(agent_loop_factory)
        tracker = getattr(agent_loop, "usage_tracker", None)
        # This session's model, which is not the default once it switched.
        model = str(agent_loop.session_model(session_id)) if agent_loop is not None else ""
        # Copied here, on the thread that records; the rendering below runs
        # off the loop because the first list-price lookup reads the bundled
        # snapshot from disk.
        usage = tracker.session_usage(session_id) if tracker is not None else None
        report = await asyncio.to_thread(session_usage_report, session_id, model, usage)
        output = f"{output.rstrip()}\n\n{report}\n"
    return {"output": output}


async def complete_slash(params: dict[str, Any]) -> dict[str, Any]:
    """No-op completion provider for slash names.

    Silences ``useCompletion.ts:97-108``'s "completion unavailable" red-frame
    toast. Real completion (slash registry walk) is v0.0.3 polish.
    """
    return {"items": [], "replace_from": 1}


async def complete_path(params: dict[str, Any]) -> dict[str, Any]:
    """No-op completion provider for filesystem paths.

    Same rationale as :func:`complete_slash`. v0.0.3 may add glob-based
    suggestions; v0.0.2 just stops the toast spam.
    """
    return {"items": []}


def register_slash_routing_methods(
    dispatcher: "Dispatcher",
    *,
    confirm_broker: "ConfirmBroker | None" = None,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> None:
    """Register all four slash routing handlers.

    ``confirm_broker`` is pre-bound to ``slash.exec`` so destructive commands
    typed as slashes in the TUI get the confirm round-trip (the production
    path users actually take). ``session.status`` / ``complete.*`` never
    confirm, so they stay broker-less; ``session.status`` gets the loop
    factory instead, for the session's usage totals.
    """

    async def _slash_exec(params: dict[str, Any]) -> dict[str, Any]:
        return await slash_exec(params, confirm_broker=confirm_broker)

    async def _session_status(params: dict[str, Any]) -> dict[str, Any]:
        return await session_status(params, agent_loop_factory=agent_loop_factory)

    dispatcher.register("slash.exec", _slash_exec)
    dispatcher.register("session.status", _session_status)
    dispatcher.register("complete.slash", complete_slash)
    dispatcher.register("complete.path", complete_path)


__all__ = [
    "complete_path",
    "complete_slash",
    "register_slash_routing_methods",
    "session_status",
    "session_usage_report",
    "slash_exec",
]
