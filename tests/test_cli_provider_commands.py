"""``ddeharness provider ...``: what the commands write, and what they refuse.

The entry is pi's ``models.json`` shape, so the flags are its own fields --
``--api-key``, ``--base-url``, ``--api``, ``--headers``, ``--models``,
``--login``, ``--name`` -- and ``provider model set`` writes one row of an
entry's ``models`` list. Nothing here normalises a provider name: a key is a pi
provider id, which has one spelling.
"""

import json

import pytest
from typer.testing import CliRunner

from opendde_harness.cli.provider_commands import provider_app
from opendde_harness.config import loader
from opendde_harness.config.schema import Config
from tests._config import config_dict, declared


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    """A config holding both kinds: one of pi's own that signs in, one declared."""
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            config_dict(
                {
                    "openai-codex": {"login": "oauth"},
                    **declared("my-relay", base_url="http://relay/v1", apiKey="sk", models=["lab-1"]),
                },
                model="my-relay/lab-1",
            )
        )
    )
    monkeypatch.setattr(loader, "_current_config_path", path)
    return path


def _entry(config_path, provider: str) -> dict:
    return json.loads(config_path.read_text())["providers"][provider]


def test_provider_set_writes_the_api(config_path):
    result = CliRunner().invoke(provider_app, ["set", "my-relay", "--api", "openai-responses"])

    assert result.exit_code == 0, result.output
    assert _entry(config_path, "my-relay")["api"] == "openai-responses"

    rejected = CliRunner().invoke(provider_app, ["set", "my-relay", "--api", "grpc"])
    assert rejected.exit_code == 1
    # The refusal names the value and the seven it could have been, and quotes
    # no part of the entry -- an entry holds the user's key.
    assert "'grpc' is not one of" in rejected.output
    assert "openai-completions" in rejected.output
    assert _entry(config_path, "my-relay")["api"] == "openai-responses", "the refused value is not written"


def test_an_address_is_written_with_the_protocol_it_serves(config_path):
    """The two go together, and the file holds them in pi's own spelling.

    The writer stored snake_case into a file the schema serializes camelCase,
    which showed only in the file: a fresh config is camelCase throughout and
    every entry the management surface touched flipped, leaving one file in two
    conventions with no rule a reader could infer.
    """
    result = CliRunner().invoke(
        provider_app,
        ["set", "my-vllm", "--base-url", "http://127.0.0.1:8000/v1", "--api", "openai-completions"],
    )

    assert result.exit_code == 0, result.output
    entry = _entry(config_path, "my-vllm")
    assert entry == {"baseUrl": "http://127.0.0.1:8000/v1", "api": "openai-completions"}
    assert "base_url" not in entry


def test_an_address_without_its_protocol_is_refused_rather_than_written(config_path):
    before = config_path.read_bytes()
    result = CliRunner().invoke(provider_app, ["set", "my-vllm", "--base-url", "http://127.0.0.1:8000/v1"])

    assert result.exit_code == 1
    assert "baseUrl is set but api is not" in result.output
    assert config_path.read_bytes() == before


def test_provider_set_writes_the_models_bare(config_path):
    """pi's ``models`` holds the ids the endpoint serves: the prefix is the key
    the entry is filed under, and storing it twice is how two spellings of one
    model used to land in one list."""
    result = CliRunner().invoke(provider_app, ["set", "my-relay", "--models", "lab-1,my-relay/lab-2"])

    assert result.exit_code == 0, result.output
    assert _entry(config_path, "my-relay")["models"] == ["lab-1", "lab-2"]


def test_provider_model_set_writes_one_models_row(config_path):
    result = CliRunner().invoke(
        provider_app,
        ["model", "set", "my-relay", "gpt-5.6-terra", "--api", "openai-responses", "--context-window", "262144"],
    )
    assert result.exit_code == 0, result.output

    again = CliRunner().invoke(provider_app, ["model", "set", "my-relay", "my-relay/gpt-5.6-terra", "--name", "Terra"])
    assert again.exit_code == 0, again.output

    entry = Config.model_validate(json.loads(config_path.read_text())).providers.get("my-relay")
    row = entry.row("gpt-5.6-terra")
    # One row, patched twice, whichever spelling the second call used.
    assert entry.model_ids == ["lab-1", "gpt-5.6-terra"]
    assert (row.api, row.context_window, row.name) == ("openai-responses", 262144, "Terra")


def test_provider_model_set_needs_a_field(config_path):
    result = CliRunner().invoke(provider_app, ["model", "set", "my-relay", "gpt-5.6-terra"])

    assert result.exit_code != 0
    assert "at least one of" in result.output


def test_a_row_edit_never_rewrites_an_invalid_entry_with_defaults(config_path):
    from pydantic import ValidationError

    from opendde_harness.config.update_providers import set_model_row

    raw = json.loads(config_path.read_text())
    raw["providers"]["my-relay"] = {"apiKey": "keep-me", "baseUrl": "http://relay/v1", "api": "telepathy"}
    config_path.write_text(json.dumps(raw))

    with pytest.raises(ValidationError):
        set_model_row("my-relay", "gpt-x", {"context_window": 1000})

    assert _entry(config_path, "my-relay")["apiKey"] == "keep-me"


def test_a_write_that_would_make_the_section_unloadable_is_refused(config_path):
    """Every write validates the whole ``providers`` map, not just the entry.

    A provider pi does not ship needs an address and a protocol, and that is a
    rule about the key an entry is filed under -- which only the section can see.
    Validating the entry alone let a write succeed and the next ``load_config``
    refuse the file, which is the one failure a single write path exists to make
    impossible.
    """
    from opendde_harness.config.update_providers import set_model_row, set_provider_fields

    before = config_path.read_bytes()

    with pytest.raises(ValueError, match="needs baseUrl and api"):
        set_model_row("brand-new", "m1", {"context_window": 131072})
    with pytest.raises(ValueError, match="needs baseUrl and api"):
        set_provider_fields("brand-new", {"api_key": "synthetic-test-key"})

    assert config_path.read_bytes() == before
    assert loader.load_config() is not None, "the file the next read gets is still a loadable one"


def test_provider_model_set_writes_and_validates_the_reasoning_effort(config_path):
    ok = CliRunner().invoke(provider_app, ["model", "set", "my-relay", "deep-thinker", "--reasoning-effort", "high"])
    assert ok.exit_code == 0, ok.output

    config = Config.model_validate(json.loads(config_path.read_text()))
    assert config.providers.get("my-relay").row("deep-thinker").reasoning_effort == "high"

    bad = CliRunner().invoke(provider_app, ["model", "set", "my-relay", "deep-thinker", "--reasoning-effort", "ultra"])
    assert bad.exit_code != 0


# ---------------------------------------------------------------------------
# Sign-in commands
# ---------------------------------------------------------------------------


class FakeService:
    """The model service as the CLI sees it: scripted replies, no child process.

    The real one has its own suite next door (``test_model_service``) and needs
    Node and a built bundle. What is under test here is the command: which
    provider id and mode it asks for, what it prints, and what it does with the
    answer.
    """

    def __init__(self, *, login_steps=(), providers=(), models=(), events=(), failure=None):
        self._login_steps = list(login_steps)
        self._providers = list(providers)
        self._models = list(models)
        self._events = list(events)
        self._failure = failure
        self.logins: list[tuple[str, str]] = []
        self.streams: list[tuple[str, str, dict, dict | None]] = []

    async def login(self, provider, *, mode="device_code"):
        self.logins.append((provider, mode))
        for step in self._login_steps:
            yield step
        if self._failure is not None:
            raise self._failure

    async def auth(self):
        return {"providers": self._providers, "stored": []}

    async def models(self):
        return self._models

    async def stream(self, provider, model, context, options=None, *, replay=None):
        self.streams.append((provider, model, context, options))
        for event in self._events:
            yield event


@pytest.fixture
def service(monkeypatch):
    """Hand every command that asks for a model service this one instead."""
    from opendde_harness.providers import pi_service

    holder: dict[str, FakeService] = {}

    async def get_service(_config):
        return holder["service"]

    async def shutdown_service():
        holder["shut_down"] = True

    monkeypatch.setattr(pi_service, "get_service", get_service)
    monkeypatch.setattr(pi_service, "shutdown_service", shutdown_service)

    def install(**kwargs) -> FakeService:
        holder["service"] = FakeService(**kwargs)
        return holder["service"]

    return install


def device_code_step(code="ABCD-1234", uri="https://auth.openai.com/device"):
    return {"type": "login_prompt", "notify": {"type": "device_code", "userCode": code, "verificationUri": uri}}


def test_provider_login_runs_pis_flow_and_stores_what_it_returns(config_path, service):
    """The sign-in is pi's, run inside the model service, printed here.

    What this command still owns is the terminal: the code and the URL reach the
    user from here, because a child process with no display and no tty cannot
    show them.
    """
    fake = service(
        login_steps=[
            {"type": "login_prompt", "notify": {"type": "progress", "message": "Requesting a device code"}},
            device_code_step(),
            {"provider": "openai-codex", "type": "oauth"},
        ]
    )

    result = CliRunner().invoke(provider_app, ["login", "openai-codex", "--no-browser"])

    assert result.exit_code == 0, result.output
    # The id goes over as it was typed: it is a pi provider id on both sides.
    assert fake.logins == [("openai-codex", "device_code")]
    assert "ABCD-1234" in result.output
    assert "auth.openai.com/device" in result.output
    assert "Authenticated with OpenAI Codex" in result.output


def test_the_sign_in_method_is_chosen_explicitly_and_never_by_a_failure(config_path, service):
    """The device code is the default: it needs no port bound and no display.

    Browser login finishes on a callback inside the service, and when that port
    is taken pi falls back to asking for the code to be pasted -- a question
    nothing on this side of the pipe can answer.
    """
    fake = service(login_steps=[{"provider": "openai-codex", "type": "oauth"}])

    assert CliRunner().invoke(provider_app, ["login", "openai-codex"]).exit_code == 0
    assert CliRunner().invoke(provider_app, ["login", "openai-codex", "--method", "browser"]).exit_code == 0
    assert fake.logins == [("openai-codex", "device_code"), ("openai-codex", "browser")]

    rejected = CliRunner().invoke(provider_app, ["login", "openai-codex", "--method", "magic"])
    assert rejected.exit_code == 1
    assert "Supported: browser, device" in rejected.output
    # Refused on its own terms, before anything talked to a vendor.
    assert len(fake.logins) == 2


def test_a_provider_reached_by_a_key_is_told_so_rather_than_signed_in(config_path, service):
    """pi owns the set that signs in, so anything else is answered with the key
    path rather than a device flow no vendor here would honour."""
    fake = service()
    result = CliRunner().invoke(provider_app, ["login", "deepseek"])
    said = " ".join(result.output.split())

    assert result.exit_code == 1
    assert "openai-codex" in said, "the ones that do sign in are named"
    assert "provider set deepseek --api-key" in said
    assert fake.logins == []


def test_a_sign_in_failure_never_prints_what_the_failure_was_carrying(config_path, service):
    """A failed token exchange can quote what it was given.

    The code says which of the three things went wrong; the message goes to the
    debug log. A sign-in failure is not worth printing a credential for.
    """
    from opendde_harness.providers.model_service import ModelServiceError

    service(
        login_steps=[device_code_step()],
        failure=ModelServiceError("login_failed", "refused: eyJhbGciOi.synthetic-secret"),
    )

    result = CliRunner().invoke(provider_app, ["login", "openai-codex", "--no-browser"])

    assert result.exit_code == 1
    assert "synthetic-secret" not in result.output
    assert "Sign-in failed (login_failed)" in result.output
    assert "--method browser" in result.output


def test_a_flow_that_ends_without_a_grant_is_not_reported_as_a_sign_in(config_path, service):
    service(login_steps=[{"provider": "openai-codex", "type": "api_key"}])

    result = CliRunner().invoke(provider_app, ["login", "openai-codex"])

    assert result.exit_code == 1
    assert "did not produce a grant" in result.output


def test_provider_reset_removes_the_entry_and_signs_the_provider_out(config_path, monkeypatch, tmp_path):
    """Removal rather than a rewrite to defaults: an entry exists because
    somebody wrote it, and one emptied of its address and key is not a provider
    with nothing configured -- for a declared provider it is not even a valid
    entry.

    The sign-out deletes pi's entry, and does it inside the service: against the
    file it would race the refresh the service runs under its own lock, and it is
    the service that put the credential there.
    """
    import json as _json

    from opendde_harness.providers import pi_service

    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    store = tmp_path / "pi-credentials.json"
    store.write_text(_json.dumps({"openai-codex": {"type": "oauth", "access": "a", "refresh": "r", "expires": 1}}))
    seen: dict[str, object] = {}

    class StubService:
        async def start(self):
            seen["started"] = True

        async def configure(self, payload):
            seen["credentials"] = payload["credentials"]
            return {}

        async def logout(self, provider):
            seen["logout"] = provider
            entries = _json.loads(store.read_text())
            forgotten = entries.pop(provider, None) is not None
            store.write_text(_json.dumps(entries))
            return {"forgotten": forgotten, "provider": provider}

        async def close(self):
            seen["closed"] = True

    # The process's own service, started by the command's loop and ended with
    # it: the one writer of the store, rather than a second child of its own.
    monkeypatch.setattr(pi_service, "ModelService", StubService)

    result = CliRunner().invoke(provider_app, ["reset", "openai-codex", "--yes"])

    assert result.exit_code == 0, result.output
    assert "openai-codex" not in json.loads(config_path.read_text())["providers"]
    assert "my-relay" in json.loads(config_path.read_text())["providers"], "only the one named is removed"
    assert seen["logout"] == "openai-codex"
    assert seen["credentials"] == str(store), "the service is pointed at this project's own store"
    assert seen["closed"] is True, "ended with the command, on the loop that started it"
    assert _json.loads(store.read_text()) == {}


def test_provider_reset_says_so_when_there_is_no_entry_to_remove(config_path):
    """An entry schema reads the same whether or not one was ever written, so
    without this the command would report having removed a provider the config
    never held."""
    result = CliRunner().invoke(provider_app, ["reset", "deepseek", "--yes"])

    assert result.exit_code == 1
    assert "no entry to remove" in result.output


@pytest.mark.parametrize("name", ["github_copilot", "github-copilot", "copilot"])
def test_removed_provider_cannot_be_readded_by_naming_it(config_path, name):
    """A removed provider is answered with its removal, whatever is written for
    it and however it is spelled -- pi still ships the id, so dropping our own
    support is not by itself a refusal."""
    before = config_path.read_bytes()
    result = CliRunner().invoke(provider_app, ["set", name, "--api-key", "synthetic-test-key"])

    assert result.exit_code != 0
    # Rejoined: Typer draws the refusal in a box, so the sentence arrives
    # wrapped across lines with a border between them.
    said = " ".join(result.output.replace("│", " ").split())
    assert "GitHub Copilot support was removed" in said
    assert config_path.read_bytes() == before


def test_provider_show_hides_the_flags_a_sign_in_never_reads(config_path):
    """One schema serves every provider, so reflection alone offered nonsense.

    ``provider show openai-codex`` listed --api-key, --base-url and --api for a
    family that refuses the first, ignores the second by design so a configured
    address cannot redirect an account's token, and refuses the third. The worst
    of them looked like a way to configure a credential that lives in a
    different file entirely.
    """
    result = CliRunner().invoke(provider_app, ["show", "openai-codex"])

    assert result.exit_code == 0, result.output
    for flag in ("--api-key", "--base-url", "--api"):
        assert f"│ {flag}" not in result.output, flag
    assert "is not read for this provider" in result.output
    assert "ddeharness provider login openai-codex" in " ".join(result.output.split())

    # A provider this config declares reads all of them: it is reached by the
    # address and the protocol its own entry states.
    declared_provider = CliRunner().invoke(provider_app, ["show", "my-relay"])
    assert declared_provider.exit_code == 0, declared_provider.output
    for flag in ("--api-key", "--base-url", "--api"):
        assert f"│ {flag}" in declared_provider.output, flag
    assert "is not read for this provider" not in declared_provider.output


def test_provider_show_refuses_a_near_miss_rather_than_describing_it(config_path):
    """The read-only surfaces describe one entry schema for every provider, so
    without this they would answer a question about a provider that does not
    exist -- and a near-miss reads as a provider this config declares."""
    result = CliRunner().invoke(provider_app, ["show", "gemini"])
    said = " ".join(result.output.split())

    assert result.exit_code == 1
    assert "'google' is the one that reaches it" in said


def test_provider_list_shows_the_entries_this_config_holds_and_no_others(config_path):
    """The listing is the configured set. pi's built-in ids are offered by the
    wizard rather than printed here -- forty rows of "not set" is not a status
    report."""
    result = CliRunner().invoke(provider_app, ["list"])

    assert result.exit_code == 0, result.output
    assert "my-relay" in result.output
    assert "openai-codex" in result.output
    assert "anthropic" not in result.output, "a provider pi ships is not a provider this config holds"
    assert "Declared" in result.output, "an address is what makes an entry a declaration"


def test_provider_list_on_a_fresh_config_offers_the_wizard(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_dict({})))
    monkeypatch.setattr(loader, "_current_config_path", path)

    result = CliRunner().invoke(provider_app, ["list"])

    assert result.exit_code == 0, result.output
    assert "No providers are configured yet" in result.output
    assert "ddeharness onboard" in result.output


@pytest.mark.parametrize("width", [60, 80, 120])
def test_provider_list_says_unreadable_rather_than_not_set(config_path, monkeypatch, tmp_path, width):
    """ "Signed out" and "the file is there and unreadable" are different answers.

    Reporting the second as the first told someone whose store was damaged that
    they had simply never signed in, and the two are not fixed the same way.
    """
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path / "chatgpt"))
    monkeypatch.setattr("opendde_harness.cli.provider_commands.console.width", width)

    from opendde_harness.providers.pi_service import credential_store_path

    path = credential_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("this is not a credential store")

    result = CliRunner().invoke(provider_app, ["list"])

    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "unreadable" in text
    assert "move it aside" in text
    # Unwrapped: the console folds a path longer than the terminal is wide.
    assert str(path) in "".join(result.output.split()), "the file to move aside is named"
    assert "ddeharness provider login openai-codex" in text


def _lab(tmp_path, monkeypatch, **overrides):
    """A config with one declared provider, pointed at nothing real."""
    entry = {
        "apiKey": "sk-lab-not-real",
        "baseUrl": "https://llm.corp.internal/v1",
        "api": "openai-completions",
        "models": ["lab-1"],
    }
    entry.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_dict({"my-relay": entry}, model="my-relay/lab-1")))
    monkeypatch.setattr(loader, "_current_config_path", path)
    return path


def test_the_credential_check_asks_the_model_the_way_a_turn_would(tmp_path, monkeypatch, service):
    """The free ``GET /v1/models`` ping is gone, and with it what it could not say.

    Seven vendors publish no such route, Azure serves it somewhere else, and a
    200 from it says nothing about whether the model will run. This asks the
    model layer instead, on the model the config actually names, and stops at
    the first event.
    """
    path = _lab(tmp_path, monkeypatch)
    from opendde_harness.config import update_providers as ops

    fake = service(
        providers=[{"id": "my-relay", "configured": True, "type": "api_key", "source": "configured"}],
        models=[{"provider": "my-relay", "id": "lab-1"}, {"provider": "openai", "id": "gpt-5.5"}],
        events=[{"type": "start"}, {"type": "text_delta", "delta": "OK"}],
    )
    result = ops.test_provider("my-relay", config_path=path)

    assert result["ok"] is True, result
    assert result["status"] == "valid"
    assert result["model"] == "lab-1"
    assert result["model_ids"] == ["lab-1"], "only this provider's rows"
    provider, model, _context, options = fake.streams[0]
    assert (provider, model) == ("my-relay", "lab-1")
    assert options["maxTokens"] <= 16, "a reachability question, not a paid one"


def test_the_credential_check_reports_the_services_own_error_code(tmp_path, monkeypatch, service):
    """pi classifies the failure; this repeats its word rather than matching text."""
    path = _lab(tmp_path, monkeypatch)
    from opendde_harness.config import update_providers as ops

    service(
        providers=[{"id": "my-relay", "configured": True, "type": "api_key", "source": "configured"}],
        models=[{"provider": "my-relay", "id": "lab-1"}],
        events=[
            {"type": "start"},
            {"type": "error", "code": "auth", "error": {"errorMessage": "401 from the relay"}},
        ],
    )
    result = ops.test_provider("my-relay", config_path=path)

    assert result["ok"] is False
    assert result["status"] == "auth"
    assert "401 from the relay" in result["error"]


def test_the_credential_check_says_not_configured_before_it_asks_a_model(tmp_path, monkeypatch, service):
    """A missing key costs nothing to find out, so nothing is sent."""
    path = _lab(tmp_path, monkeypatch, apiKey="")
    from opendde_harness.config import update_providers as ops

    fake = service(providers=[{"id": "my-relay", "configured": False, "error": "no key"}])
    result = ops.test_provider("my-relay", config_path=path)

    assert result == {
        "ok": False,
        "status": "not_configured",
        "elapsed_ms": result["elapsed_ms"],
        "model": "",
        "models_count": None,
        "model_ids": None,
        "error": "no key",
    }
    assert fake.streams == []


def test_an_entry_the_model_layer_does_not_carry_says_so(tmp_path, monkeypatch, service):
    """A declared provider reaches pi by being declared, and pi has no catalogue
    of its own for one -- so an entry naming no models can be declared from
    nothing. That is not a bad key, and saying "invalid key" sent the user to a
    field that was already right."""
    path = _lab(tmp_path, monkeypatch, models=[])
    from opendde_harness.config import update_providers as ops

    fake = service(providers=[{"id": "openai", "configured": True, "type": "api_key"}])
    result = ops.test_provider("my-relay", config_path=path)

    assert result["status"] == "not_served"
    assert "provider show my-relay" in result["error"]
    assert fake.streams == []


def test_the_command_prints_the_one_thing_that_fixes_each_failure(tmp_path, monkeypatch, service):
    _lab(tmp_path, monkeypatch)
    service(
        providers=[{"id": "my-relay", "configured": True, "type": "api_key"}],
        models=[{"provider": "my-relay", "id": "lab-1"}],
        events=[{"type": "error", "code": "auth", "error": {"errorMessage": "refused"}}],
    )
    result = CliRunner().invoke(provider_app, ["test", "my-relay"])

    assert result.exit_code == 1
    assert "my-relay failed: auth" in result.output
    assert "provider set my-relay --api-key" in result.output


def test_the_onboarding_probe_asks_for_no_thinking(tmp_path, monkeypatch):
    """Its 200-token cap cannot pay for a thinking budget, and it does not need to.

    A budget-thinking Anthropic model needs at least 1024 thinking tokens, so
    the probe's cap and the global medium default were refused before any
    request was built: the wizard reported a correctly configured provider as
    broken and offered to re-enter the key. The probe states ``off`` rather
    than inheriting a level, and a declared medium must not reach the request.
    """
    from opendde_harness.providers import model_id
    from opendde_harness.providers.pi_provider import build_pi_provider
    from tests._config import config as build_config

    model = "anthropic/claude-opus-4-5"
    config = build_config(
        {"anthropic": {"apiKey": "sk-ant-not-real", "models": [{"id": "claude-opus-4-5", "maxTokens": 8000}]}},
        model=model,
    )
    provider = build_pi_provider(config, model, "anthropic")
    assert config.agents.defaults.reasoning_effort == "medium", "the level the request must not inherit"
    assert model_id.row_for(config.providers, model) is not None, "the model's own row is what declares its ceiling"

    options = provider._options(model, max_tokens=200, reasoning_effort="off", tool_choice=None)

    assert options["maxTokens"] == 200
    assert "reasoning" not in options, "off is spelled by leaving the level out"


def test_the_endpoint_subcommand_is_gone_along_with_multi_endpoint_failover():
    """One address per provider, so there is nothing to manage a list of.

    Asserted on the command surface rather than on an import, because the
    subcommand group is what a user would find: a `provider endpoint add` that
    still parsed would write a key into a field nothing reads.
    """
    result = CliRunner().invoke(provider_app, ["endpoint", "list", "openrouter"])

    assert result.exit_code != 0
    assert "endpoint" not in CliRunner().invoke(provider_app, ["--help"]).output


def test_qwen_fast_profile_disables_thinking_without_erasing_provider(config_path):
    before = _entry(config_path, "my-relay")
    result = CliRunner().invoke(provider_app, ["model", "set", "my-relay", "lab-1", "--qwen-fast"])
    assert result.exit_code == 0, result.output
    entry = _entry(config_path, "my-relay")
    assert entry["apiKey"] == before["apiKey"] and entry["baseUrl"] == before["baseUrl"]
    row = Config.model_validate(json.loads(config_path.read_text())).providers.get("my-relay").row("lab-1")
    assert row.reasoning is True and row.reasoning_effort == "off"
    assert row.max_tokens == 2048 and row.compat["thinkingFormat"] == "qwen"


def test_qwen_fast_preserves_compat_and_supports_vllm_format(config_path):
    from opendde_harness.config.update_providers import set_model_row

    set_model_row("my-relay", "lab-1", {"compat": {"supportsStore": False}, "max_tokens": 8000})
    result = CliRunner().invoke(
        provider_app,
        [
            "model",
            "set",
            "my-relay",
            "lab-1",
            "--qwen-fast",
            "--thinking-format",
            "qwen-chat-template",
            "--max-tokens",
            "1024",
        ],
    )
    assert result.exit_code == 0, result.output
    row = Config.model_validate(json.loads(config_path.read_text())).providers.get("my-relay").row("lab-1")
    assert row.compat == {"supportsStore": False, "thinkingFormat": "qwen-chat-template"}
    assert row.max_tokens == 1024


def test_qwen_fast_rejects_conflicting_reasoning_request(config_path):
    before = config_path.read_bytes()
    result = CliRunner().invoke(
        provider_app, ["model", "set", "my-relay", "lab-1", "--qwen-fast", "--reasoning-effort", "high"]
    )
    assert result.exit_code != 0 and config_path.read_bytes() == before
