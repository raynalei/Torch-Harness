"""``ddeharness doctor`` — health check (static + optional --probe).

The static checks reach no vendor. The one child they start is the model
service, asked what it knows about the default model and closed again -- a
catalogue read, with no credential store written and no key seeded. The store is
read, though, because "is this provider signed in" has no other answer. Memory is probed over HTTP. ``--probe`` sends one chat exchange via
:func:`opendde_harness.cli._helpers.send_probe`.

Exit codes:
  0  — all green (and probe ok if requested)
  1  — static check failed (config missing / schema invalid / unresolved routing)
  2  — static checks ok but ``--probe``, memory readiness failed
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Optional

import typer
from rich.console import Console

from opendde_harness import __logo__, __version__
from opendde_harness.cli._helpers import print_probe_troubleshooting, send_probe
from opendde_harness.plugin.memory.longterm.doctor import MemoryInfo, render_memory

if TYPE_CHECKING:
    from opendde_harness.config.opendde_harness import OpenDDEHarnessConfig
    from opendde_harness.providers.rates import Resolved

console = Console()


@dataclass
class PathsInfo:
    config_path: str
    config_exists: bool
    config_valid: bool = False
    config_invalid_reason: str = ""
    workspace_path: str = ""
    workspace_exists: bool = False


@dataclass
class RoutingInfo:
    model: str
    provider: Optional[str]
    max_tokens: Optional[int]
    context_window_tokens: Optional[int]
    # Which tier answered each number (providers/rates SOURCE_*), so a figure
    # nothing measured reads as unknown rather than as a setting.
    max_tokens_source: str = "unknown"
    context_window_source: str = "unknown"
    # The wire the default model's requests travel on, and what decided it.
    # Always declared and never probed: a provider this config declares names
    # its own ``api``, one of its model rows may override that for a single
    # model, and one of pi's own providers carries pi's -- which lives inside
    # pi-ai and cannot be read without starting the service, so it is named as
    # pi's rather than guessed from the provider's id.
    api: str = ""
    api_source: str = ""
    #: Why the model reaches no provider, when it reaches none. Taken from
    #: ``Config.explain_unrouted`` rather than re-derived here, so doctor and
    #: every other surface give the same reason for the same config.
    unrouted_reason: str = ""


@dataclass
class ProviderInfo:
    """One configured provider, and what actually answers for it.

    ``source`` is the part worth reporting rather than merely "has a key":
    a key in ``config.json``, a key in the vendor's own environment variable and
    a sign-in in the model service's credential store all read as configured,
    and somebody asking why a provider answers -- or answers as the wrong
    account -- needs to know which of the three did.
    """

    name: str
    display_name: str
    #: Does this config declare the provider (an address of its own), rather
    #: than name one of pi's built-ins?
    declared: bool
    configured: bool
    #: ``providers.auth`` KIND_*: which shape of credential this entry is.
    kind: str
    #: ``config`` / ``environment`` / ``store`` / ``declared``, or "" for none.
    source: str = ""
    #: The environment variable the key was read from, when that is the source.
    env_var: str = ""
    base_url: str = ""
    #: Whether ``config.json`` itself holds a key for this entry. Separate from
    #: ``source`` because a declared provider reports as reachable by its
    #: address whether or not it also carries one.
    has_key: bool = False
    #: What is missing, in the words that say how to supply it.
    missing: list[str] = field(default_factory=list)
    model_count: int = 0
    routes_default: bool = False
    #: The bare id the default model asks this provider for, when it is the one
    #: that routes it. Empty otherwise.
    default_served: str = ""

    @property
    def serves_nothing(self) -> bool:
        """A provider this config declares, with no models: it can serve none.

        pi has no catalogue for a declared provider, so the entry's ``models``
        list *is* its catalogue. Empty, the entry is dropped from the model
        service's configuration outright
        (:func:`opendde_harness.providers.pi_auth.configure_payload`) -- so every
        model id naming it fails at the request, and nothing before this line
        ever said why.
        """
        return self.declared and not self.model_count


@dataclass
class FeaturesInfo:
    skill_forge_enabled: bool = False


@dataclass
class ProbeResult:
    ok: bool
    text: Optional[str] = None
    tokens: Optional[int] = None
    elapsed_s: Optional[float] = None
    error: Optional[str] = None


@dataclass
class DoctorReport:
    version: int = 1
    #: The OpenDDE Harness this is, and -- when PyPI has a newer one -- what
    #: it is and the command that installs it here.
    harness_version: str = ""
    update: Optional[dict] = None
    config_loaded: bool = False
    config_error: Optional[str] = None
    paths: Optional[PathsInfo] = None
    routing: Optional[RoutingInfo] = None
    #: Every provider the config holds an entry for. Empty before the config
    #: loads, which is why an unconfigured install has nothing to say here.
    providers: list[ProviderInfo] = field(default_factory=list)
    features: Optional[FeaturesInfo] = None
    memory: Optional[MemoryInfo] = None
    probe: Optional[ProbeResult] = None

    def exit_code(self) -> int:
        if self.paths is None or not self.paths.config_exists:
            return 1
        if not self.paths.config_valid:
            return 1
        if not self.config_loaded:
            return 1
        if self.routing is None or self.routing.provider is None:
            return 1
        if self.probe is not None and not self.probe.ok:
            return 2
        # A role the user configured that the server could not build is a real
        # fault, not a warning: recall silently returns nothing.
        if self.memory is not None and self.memory.broken:
            return 2
        return 0


def _gather_static_checks() -> DoctorReport:
    """Inspect config / routing / features. No vendor is contacted.

    The model layer is asked what it serves for the default model
    (:func:`_catalogue_row`), which reads pi's own built-in rows in a child
    process and nothing else.
    """
    from opendde_harness.config.loader import ConfigSchemaError, get_config_path, load_config

    config_path = get_config_path()
    paths = PathsInfo(
        config_path=str(config_path),
        config_exists=config_path.exists(),
    )
    report = DoctorReport(paths=paths, harness_version=__version__, update=_update_available())

    if not paths.config_exists:
        return report

    # Classify config validity with load_config's eyes: a syntax error, an
    # empty file, and a non-object top level all mean no settings were read.
    # Inspect the file directly -- load_config swallows syntax errors into
    # defaults, and read_raw_or_raise folds the last two cases into {} for
    # its read-modify-write callers, so neither can classify all three.
    try:
        text = config_path.read_text(encoding="utf-8")
        data = json.loads(text) if text.strip() else None
    except (OSError, UnicodeDecodeError, ValueError):
        paths.config_invalid_reason = "invalid JSON"
    else:
        if not text.strip():
            paths.config_invalid_reason = "empty"
        elif not isinstance(data, dict):
            paths.config_invalid_reason = "not a JSON object"
    paths.config_valid = not paths.config_invalid_reason

    try:
        config = load_config()
    except ConfigSchemaError as exc:
        report.config_error = str(exc)
        return report
    except Exception:
        return report
    report.config_loaded = True

    workspace = config.workspace_path
    paths.workspace_path = str(workspace)
    paths.workspace_exists = workspace.exists()

    defaults = config.agents.defaults
    # What a request will actually carry, resolved the same way the loop
    # resolves it -- doctor reporting a configured number that no longer
    # exists would be reporting a setting, not the behaviour.
    ceiling, window = _model_limits(config, defaults.model)
    api, api_source = _api_facts(config)
    provider = config.get_provider_name()
    report.routing = RoutingInfo(
        model=defaults.model,
        provider=provider,
        max_tokens=ceiling.tokens,
        context_window_tokens=window.tokens,
        max_tokens_source=ceiling.source,
        context_window_source=window.source,
        api=api,
        api_source=api_source,
        unrouted_reason="" if provider else config.explain_unrouted(defaults.model),
    )
    report.providers = _gather_providers(config)

    try:
        skill_forge_on = bool(config.skill_forge.enabled)
    except Exception:
        skill_forge_on = False
    report.features = FeaturesInfo(skill_forge_enabled=skill_forge_on)
    return report


def _probe_memory(config: "OpenDDEHarnessConfig") -> MemoryInfo:
    """The memory plugin's own diagnosis. Local HTTP only, never raises.

    Deliberately not part of ``_gather_static_checks``: that stays zero-network.
    This one talks to localhost, which is cheap enough to run unconditionally --
    unlike ``--probe``, it spends no tokens and reaches no third party.
    """
    from opendde_harness.plugin.memory.longterm.doctor import probe_memory

    return probe_memory(config)


def _run_llm_probe(timeout_s: int) -> ProbeResult:
    """Wrap :func:`send_probe` so failures become a structured ProbeResult."""
    try:
        text, tokens, elapsed = send_probe(timeout_s=timeout_s)
        return ProbeResult(ok=True, text=text, tokens=tokens, elapsed_s=elapsed)
    except Exception as exc:
        return ProbeResult(ok=False, error=str(exc) or exc.__class__.__name__)


def _model_limits(config, model: str) -> tuple["Resolved", "Resolved"]:
    """``(output ceiling, context window)`` a request for ``model`` will carry.

    The same two tiers the loop walks -- the user's declaration for this model
    (``AgentLoop.resolve_window``, ``providers.base.send_max_tokens``), then the
    model layer's own row for it -- so doctor reports the behaviour rather than a
    setting. Unknown where neither answers: an invented number here would read
    as a measurement.
    """
    from opendde_harness.providers import model_id
    from opendde_harness.providers.base import declared_tokens
    from opendde_harness.providers.rates import SOURCE_DECLARED, SOURCE_SERVICE, SOURCE_UNKNOWN, Resolved

    # The declaration is the model's own row on its provider's entry, looked up
    # by the qualified id exactly: a row declared for a self-hosted model must
    # never answer for a hosted one that happens to share its name.
    declared = model_id.row_for(config.providers, model)
    ceiling = declared_tokens(declared, "max_tokens")
    window = declared_tokens(declared, "context_window")
    reported = {} if ceiling and window else _catalogue_row(config, model, declared)

    def resolved(written: Optional[int], field: str) -> "Resolved":
        if written:
            return Resolved(written, SOURCE_DECLARED)
        value = reported.get(field)
        if isinstance(value, int) and value > 0:
            return Resolved(value, SOURCE_SERVICE)
        return Resolved(None, SOURCE_UNKNOWN)

    return resolved(ceiling, "maxTokens"), resolved(window, "contextWindow")


def _catalogue_row(config, model: str, declared) -> dict:
    """What the model layer says about ``model``, or ``{}`` when it cannot say.

    Asked over the service's ``catalog`` request, which reads pi's own built-in
    rows and is answered without a ``configure``: this command reports on a
    configuration, it does not install one, so it starts a bare child, asks and
    closes it -- no credential store is written and no key is seeded. Rows for
    the provider this model is actually routed to win; otherwise every row for
    the id is merged, the same way ``configure`` sizes a declared model.

    Anything that goes wrong -- no Node, no built bundle -- is "unknown". A
    diagnostic that cannot read a fact says so; it does not fail.
    """
    import asyncio

    from opendde_harness.providers import model_id
    from opendde_harness.providers.model_service import ModelService
    from opendde_harness.providers.pi_auth import merged_catalog_row

    prefix, served = model_id.split(model)
    wanted = (declared.catalog_model if declared is not None else None) or served or model
    if not wanted:
        return {}

    async def ask() -> list[dict]:
        service = ModelService()
        await service.start()
        try:
            return await service.catalog(wanted)
        finally:
            await service.close()

    try:
        rows = asyncio.run(ask())
    except Exception:  # noqa: BLE001 - a fact doctor cannot read is reported as unknown
        return {}
    # The id's own prefix is the provider, and pi's ids are pi's ids -- there is
    # no name to translate on the way in or out any more.
    routed = config.get_provider_name(model) or prefix
    return merged_catalog_row([row for row in rows if row.get("provider") == routed] or rows)


def _api_facts(config) -> tuple[str, str]:
    """(wire, what decided it) for the default model, or two empties when unrouted.

    The wire is declared, never probed, and there are exactly three answers. A
    model row's own ``api`` wins, because one relay can serve different models on
    different wires. Otherwise a provider this config declares names its ``api``,
    which the schema requires of it alongside the address. And for one of pi's
    own built-ins the wire is pi's: it lives in pi-ai's provider table, this
    process would have to start the model service to read it, and naming pi as
    the holder is the honest report -- the per-vendor default table that used to
    answer here was this project keeping a second copy of pi's own fact, and it
    went stale by construction.
    """
    from opendde_harness.providers import model_id

    model = config.agents.defaults.model or ""
    name = config.get_provider_name(model) or ""
    entry = config.providers.get(name)
    if entry is None:
        return "", ""
    row = model_id.row_for(config.providers, model)
    if row is not None and row.api:
        return row.api, f"the {row.id} row on providers.{name}"
    if entry.api:
        return entry.api, f"providers.{name}.api"
    return "pi's own", f"{name} is one of pi's built-ins; only the model service holds its wire"


def _gather_providers(config) -> list[ProviderInfo]:
    """Every provider the config holds an entry for, and what answers for it.

    The ``providers`` section is the configured set -- an entry is there because
    somebody wrote it -- and :func:`opendde_harness.providers.auth.credential_status`
    is what adds the material the file does not hold. Asked with
    ``include_external`` because doctor reports what is true right now: a sign-in
    in the model service's store and a key in the vendor's own environment
    variable both reach the vendor, and reading the file alone called each of
    them unset.
    """
    from opendde_harness.providers import model_id, pi_ids
    from opendde_harness.providers.auth import KIND_API_KEY, configured_key, credential_status, env_key_name

    default_model = config.agents.defaults.model or ""
    routed = model_id.provider_of(default_model)
    infos: list[ProviderInfo] = []
    for provider, entry in config.providers.items():
        status = credential_status(provider, entry, include_external=True)
        infos.append(
            ProviderInfo(
                name=provider,
                display_name=pi_ids.display_name(provider, entry.name),
                declared=entry.declared,
                configured=status.ok,
                kind=status.kind,
                source=status.source,
                # Named only for the case it explains. A whole environment chain
                # (the AWS one, Google ADC) also reports as "environment" and has
                # no single variable to point at.
                env_var=(
                    env_key_name(provider) if status.kind == KIND_API_KEY and status.source == "environment" else ""
                ),
                base_url=entry.base_url,
                has_key=bool(configured_key(entry)),
                missing=[requirement.hint or requirement.label for requirement in status.missing],
                model_count=len(entry.models),
                routes_default=provider == routed,
                # The id as the endpoint is asked for it, so a remedy naming it
                # does not tell the user to type the provider twice.
                default_served=model_id.display(provider, default_model) if provider == routed else "",
            )
        )
    return infos


def _declare_hint(model: str, flag: str) -> str:
    """The ``provider model set`` command that declares one limit for ``model``.

    A limit is declared per model, on a row of its provider's own entry, so the
    remedy has to name the two halves of the model id separately -- printing the
    qualified id would be telling the user to type the provider twice.
    """
    from opendde_harness.providers import model_id

    provider, served = model_id.split(model)
    return f"declare it: ddeharness provider model set {provider or '<provider>'} {served} {flag} <tokens>"


def _credential_detail(info: ProviderInfo) -> str:
    """What answers for a configured provider, in one phrase.

    The phrase is the ``source``, not a tick: "configured" is true of a key in
    ``config.json``, a key in the vendor's environment variable and a sign-in in
    the model service's store alike, and which one answered is the fact somebody
    reading doctor is here for.
    """
    from opendde_harness.providers.auth import KIND_AMBIENT, KIND_DEVICE_FLOW

    if info.kind == KIND_DEVICE_FLOW:
        # Before the address, because an entry can carry both and the sign-in is
        # what reaches the vendor.
        return "signed in  [dim](grant in the model service's credential store)[/dim]"
    if info.kind == KIND_AMBIENT:
        # A whole environment chain rather than one variable, so there is no key
        # to name a source for.
        return "the environment's own credentials"
    if info.source == "declared":
        # For a provider this config declares the address is what makes it
        # reachable, and a key is optional -- a self-hosted server wants none.
        held = "key in config.json" if info.has_key else "no key"
        return f"{info.base_url}  [dim]({held})[/dim]"
    if info.source == "environment":
        return f"key from [cyan]{info.env_var}[/cyan]" if info.env_var else "key from the environment"
    if info.source == "store":
        return "key in the model service's credential store"
    if info.source == "config":
        return "key in config.json"
    return "configured"


def _render_providers(providers: list[ProviderInfo]) -> None:
    """One line per configured provider, and the two faults only doctor sees.

    An entry that reaches nothing is a fault rather than a blank, because an
    entry exists only where somebody wrote one: nothing is offered empty and
    waiting to be filled in. So every entry gets a line here, unlike ``status``,
    which folds the unusable ones into a count.
    """
    if not providers:
        return
    console.print("\n[bold]Providers[/bold]")
    width = max(len(info.name) for info in providers)
    for info in providers:
        label = f"  {info.name:<{width}}"
        routes = "  [dim]← routes the default model[/dim]" if info.routes_default else ""
        if info.configured:
            console.print(f"{label}  [green]✓[/green] {_credential_detail(info)}{routes}")
        else:
            console.print(f"{label}  [red]✗[/red] needs {'; '.join(info.missing) or 'configuring'}{routes}")
        if info.serves_nothing:
            # pi ships no catalogue for a provider this config declares, so an
            # empty ``models`` list is not "everything it serves" but nothing:
            # the entry never reaches the model service at all, and the failure
            # otherwise surfaces as an unroutable model id at the first request.
            console.print(
                f"  [yellow]⚠ {info.name} declares an address but no models, so it can serve none: "
                "pi has no catalogue for a provider this config declares.[/yellow]"
            )
            console.print(
                f"    [dim]ddeharness provider set {info.name} --models {info.default_served or '<model id>'}[/dim]"
            )


def _update_available() -> Optional[dict]:
    """PyPI's newer release and the command for it, or None. Doctor is the one
    place that waits for the answer: it is a check, and a few seconds is what
    a check costs."""
    from opendde_harness.cli.update_notice import update_notice

    notice = update_notice(__version__, wait=5.0)
    return {"latest": notice[0], "command": notice[1]} if notice else None


def _render_human_output(report: DoctorReport) -> None:
    console.print(f"\n{__logo__} OpenDDE Harness Doctor\n")

    console.print("[bold]Version[/bold]")
    if report.update:
        console.print(
            f"  {report.harness_version}  [yellow]↑ {report.update['latest']} is on PyPI[/yellow]  "
            f"[dim]update with: {report.update['command']}[/dim]"
        )
    else:
        console.print(f"  {report.harness_version}  [green]✓[/green]")
    console.print()

    paths = report.paths
    assert paths is not None  # _gather_static_checks always populates this
    console.print("[bold]Paths[/bold]")
    if not paths.config_exists:
        console.print(f"  Config:    {paths.config_path}  [red]✗  (not found)[/red]")
    elif not paths.config_valid:
        reason = paths.config_invalid_reason or "invalid JSON"
        console.print(f"  Config:    {paths.config_path}  [yellow]⚠  {reason} (running on defaults)[/yellow]")
    else:
        console.print(f"  Config:    {paths.config_path}  [green]✓[/green]")
    if paths.config_exists:
        mark = "[green]✓[/green]" if paths.workspace_exists else "[red]✗[/red]"
        console.print(f"  Workspace: {paths.workspace_path}  {mark}")

    if not paths.config_exists:
        console.print(
            "\n[yellow]⚠ OpenDDE Harness is not configured.[/yellow] Run [cyan]ddeharness onboard[/cyan] to set it up."
        )
        return

    if not report.config_loaded:
        if paths.config_valid:
            console.print(
                "\n[red]✗ Config schema invalid.[/red] Run [cyan]ddeharness onboard --reset[/cyan] to recreate it."
            )
            # The unknown-key line, without pydantic's full listing.
            for line in (report.config_error or "").splitlines():
                if line.startswith("unknown key(s):"):
                    console.print(f"  {line}")
        else:
            reason = paths.config_invalid_reason or "invalid JSON"
            console.print(f"\n[yellow]⚠ Config file is {reason}; the checks above ran on built-in defaults.[/yellow]")
            console.print(f"Fix [cyan]{paths.config_path}[/cyan] or run [cyan]ddeharness onboard --reset[/cyan].")
        return

    routing = report.routing
    if routing is not None:
        console.print("\n[bold]Routing[/bold]")
        console.print(f"  Model:        {routing.model}")
        if routing.provider:
            console.print(f"  Routes to:    {routing.provider}")
        else:
            console.print("  Routes to:    [red]<unresolved>[/red]")
        if routing.max_tokens:
            console.print(f"  Max tokens:   {routing.max_tokens}  [dim]({routing.max_tokens_source})[/dim]")
        else:
            console.print(
                "  Max tokens:   [yellow]unknown[/yellow]  "
                "[dim](nothing declares one, and the model service refuses a request with no ceiling)[/dim]"
            )
            # Only worth saying once the model reaches a provider: with no entry
            # to hold the row, the fix is the entry and the tail line says so.
            if routing.provider:
                console.print(f"                [dim]{_declare_hint(routing.model, '--max-tokens')}[/dim]")
        if routing.context_window_tokens:
            console.print(
                f"  Context win:  {routing.context_window_tokens}  [dim]({routing.context_window_source})[/dim]"
            )
        else:
            console.print(
                "  Context win:  [yellow]unknown[/yellow]  [dim](the model layer lists no window; history is not trimmed)[/dim]"
            )
            if routing.provider:
                console.print(f"                [dim]{_declare_hint(routing.model, '--context-window')}[/dim]")
        if routing.api:
            console.print(f"  Wire:         {routing.api}  [dim]({routing.api_source})[/dim]")

    _render_providers(report.providers)

    features = report.features
    if features is not None:
        console.print("\n[bold]Features[/bold]")
        sf_label = "enabled" if features.skill_forge_enabled else "[dim]disabled[/dim]"
        console.print(f"  Skill forge: {sf_label}")

    memory = report.memory
    if memory is not None and memory.disabled_reason:
        render_memory(console, memory)
    elif memory is not None and memory.backend:
        console.print("\n[bold]Memory[/bold]")
        console.print(f"  Backend:    {memory.backend}")
        render_memory(console, memory)

    if report.probe is not None:
        console.print("\n[bold]LLM Probe[/bold]")
        if routing:
            console.print(f"  → {routing.model}")
        if report.probe.ok:
            console.print(f'  [green]✓ Response:[/green] "{report.probe.text}"')
            extras: list[str] = []
            if report.probe.tokens:
                extras.append(f"{report.probe.tokens} tokens")
            if report.probe.elapsed_s is not None:
                extras.append(f"{report.probe.elapsed_s:.1f}s")
            if extras:
                console.print(f"  [green]✓ {', '.join(extras)}[/green]")
        else:
            console.print(f"  [red]✗ Failed:[/red] {report.probe.error}")
            print_probe_troubleshooting(routing.provider if routing else None)

    console.print()
    code = report.exit_code()
    if code == 0:
        if report.probe is None:
            console.print("[green]✓ Configuration looks healthy.[/green]")
            console.print("Run [cyan]doctor --probe[/cyan] to send a test message and verify the LLM responds.")
        else:
            console.print("[green]✓ All checks passed.[/green]")
    elif not paths.config_valid:
        reason = paths.config_invalid_reason or "invalid JSON"
        console.print(f"[yellow]⚠ Config file is {reason}; the checks above ran on built-in defaults.[/yellow]")
        console.print(f"Fix [cyan]{paths.config_path}[/cyan] (JSON allows no comments or trailing commas).")
    elif routing and routing.provider is None:
        console.print(f"[red]✗ Model [bold]{routing.model}[/bold] could not be routed: {routing.unrouted_reason}[/red]")
        console.print(
            "Run [cyan]ddeharness provider list[/cyan] / [cyan]ddeharness provider set[/cyan] to fix routing."
        )


def register(app: typer.Typer) -> None:
    @app.command()
    def doctor(
        probe: bool = typer.Option(False, "--probe", help="Send a test message to verify the LLM responds."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON (CI-friendly)."),
        timeout: int = typer.Option(
            15,
            "--timeout",
            help="LLM probe timeout in seconds.",
            min=1,
        ),
    ) -> None:
        """Check provider and memory configuration."""
        report = _gather_static_checks()

        if report.config_loaded:
            from opendde_harness.config.opendde_harness import load_opendde_harness_config

            report.memory = _probe_memory(load_opendde_harness_config())

        if probe and report.routing is not None and report.routing.provider is not None:
            report.probe = _run_llm_probe(timeout_s=timeout)

        if json_output:
            console.print_json(json.dumps(asdict(report)))
        else:
            _render_human_output(report)

        raise typer.Exit(report.exit_code())


__all__ = ["register"]
