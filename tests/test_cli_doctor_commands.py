import json
import re

import pytest
import typer
from typer.testing import CliRunner

from opendde_harness.cli import doctor_commands, onboard_commands
from opendde_harness.cli.doctor_commands import DoctorReport, MemoryInfo, PathsInfo, RoutingInfo
from tests._config import config as build_config
from tests._config import declared, keyed


@pytest.fixture
def app(monkeypatch, tmp_path):
    # Never the machine's own config: doctor reads it wherever it is not stubbed.
    from opendde_harness.config import loader

    monkeypatch.setattr(loader, "_current_config_path", tmp_path / "config.json")
    healthy = DoctorReport(
        config_loaded=True,
        paths=PathsInfo(config_path="/test/config.json", config_exists=True, config_valid=True),
        # A qualified model id, because that is the only kind there is now: the
        # prefix names the provider, and the remedy lines print the two halves
        # separately.
        routing=RoutingInfo(model="openai/gpt-4o", provider="openai", max_tokens=1, context_window_tokens=None),
    )
    monkeypatch.setattr(doctor_commands, "_gather_static_checks", lambda: healthy)
    monkeypatch.setattr(doctor_commands, "_probe_memory", lambda _: MemoryInfo())
    app = typer.Typer()
    doctor_commands.register(app)
    return app


def _config(protein_design=None):
    return {"plugins": {"config": {"protein-design": protein_design}}} if protein_design is not None else {}


def test_bare_doctor_without_protein_design_config_exits_zero_and_skips_compute(app, monkeypatch):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: _config())
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert "Configuration looks healthy" in result.output
    as_json = CliRunner().invoke(app, ["--json"])
    assert as_json.exit_code == 0, as_json.output


def test_doctor_has_no_preparation_flags(app):
    for flag in ("--fix", "--assets-only", "--root", "--mode", "--checkpoint", "--code-only"):
        assert CliRunner().invoke(app, [flag]).exit_code == 2


def test_unknown_context_window_is_said_not_estimated(app, monkeypatch):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: _config())
    result = CliRunner().invoke(app, [])
    text = re.sub(r"\s+", " ", result.output)
    assert "Context win: unknown" in text
    # And the remedy names the row that would declare it: a limit is a field on
    # the model's own row of its provider's entry, so the command takes the two
    # halves of the model id separately.
    assert "ddeharness provider model set openai gpt-4o --context-window" in text
    assert "Max tokens: 1 (unknown)" in text


def test_resolved_context_window_names_its_source(app, monkeypatch):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: _config())
    report = doctor_commands._gather_static_checks()
    report.routing = RoutingInfo(
        model="deepseek/deepseek-v4-flash",
        provider="deepseek",
        max_tokens=384000,
        context_window_tokens=1000000,
        max_tokens_source="models.dev/provider",
        context_window_source="models.dev/provider",
    )
    result = CliRunner().invoke(app, [])
    text = re.sub(r"\s+", " ", result.output)
    assert "Context win: 1000000 (models.dev/provider)" in text
    assert "Max tokens: 384000 (models.dev/provider)" in text
    as_json = json.loads(CliRunner().invoke(app, ["--json"]).output)
    assert as_json["routing"]["context_window_source"] == "models.dev/provider"


def test_memory_present_but_switched_off_is_reported(app, monkeypatch):
    """A managed memory root with no credentials answers zero hits forever, silently."""
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: _config())
    monkeypatch.setattr(
        doctor_commands,
        "_probe_memory",
        lambda _: MemoryInfo(root="/test/memory", disabled_reason="no credentials for llm"),
    )

    result = CliRunner().invoke(app, [])

    assert "Disabled" in result.output and "no credentials for llm" in result.output
    assert "recall returns nothing" in result.output
    # Not a fault: the user chose not to finish setting it up.
    assert result.exit_code == 0


def test_api_facts_report_the_models_own_row_over_the_entry():
    """One relay can serve two models on two protocols, so a row's ``api`` wins.

    Three answers and no fourth: the model's own row, else the entry's ``api``
    (which the schema requires of every provider this config declares), else
    pi's -- which lives inside pi-ai and is named as pi's rather than guessed.
    """
    row_wins = build_config(
        declared(
            "custom",
            base_url="http://relay/v1",
            api="openai-completions",
            models=[{"id": "gpt-x", "api": "openai-responses"}],
            apiKey="k",
        ),
        model="custom/gpt-x",
    )
    assert doctor_commands._api_facts(row_wins) == ("openai-responses", "the gpt-x row on providers.custom")

    entry_answers = build_config(
        declared("custom", base_url="http://relay/v1", api="openai-completions", models=["gpt-x"], apiKey="k"),
        model="custom/gpt-x",
    )
    assert doctor_commands._api_facts(entry_answers) == ("openai-completions", "providers.custom.api")

    # One of pi's own: the wire is pi's, and saying so is the honest report.
    builtin = build_config(keyed("anthropic"), model="anthropic/claude-sonnet-5")
    api, source = doctor_commands._api_facts(builtin)
    assert api == "pi's own" and "pi's built-ins" in source


def test_every_configured_provider_gets_a_row_and_an_empty_declaration_is_a_fault(tmp_path, monkeypatch):
    """The report says what answers for each entry, not merely that one exists.

    An entry is there because somebody wrote it, so a declaration with no models
    is a fault and not a blank: pi ships no catalogue for a provider this config
    declares, so the entry never reaches the model service and every model id
    naming it fails at the request.
    """
    from opendde_harness.config import loader
    from tests._config import write_config

    path = write_config(
        tmp_path / "config.json",
        {**keyed("anthropic", key="sk-ant-0000000000000000"), **declared("my-vllm", models=[])},
        model="anthropic/claude-sonnet-5",
    )
    monkeypatch.setattr(loader, "_current_config_path", path)
    # No child process: the catalogue is the model service's answer, and doctor
    # reports "unknown" rather than starting one here.
    monkeypatch.setattr(doctor_commands, "_catalogue_row", lambda *_a: {})
    monkeypatch.setattr(doctor_commands, "_probe_memory", lambda _: MemoryInfo())
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: _config())
    app = typer.Typer()
    doctor_commands.register(app)

    rows = {row["name"]: row for row in json.loads(CliRunner().invoke(app, ["--json"]).output)["providers"]}

    assert rows["anthropic"]["configured"] is True and rows["anthropic"]["declared"] is False
    assert rows["anthropic"]["routes_default"] is True and rows["anthropic"]["default_served"] == "claude-sonnet-5"
    assert rows["my-vllm"]["declared"] is True and rows["my-vllm"]["model_count"] == 0
    assert rows["my-vllm"]["base_url"] == "http://127.0.0.1:8000/v1"

    text = re.sub(r"\s+", " ", CliRunner().invoke(app, []).output)
    assert "my-vllm declares an address but no models, so it can serve none" in text


def test_a_model_nothing_sizes_is_reported_as_unknown_on_both_lines(monkeypatch, tmp_path):
    """No declared row and nothing the model layer can report: the window and
    the ceiling are both unknown, and each line says so instead of printing a
    figure that would read as a measurement."""
    from opendde_harness.config import loader

    monkeypatch.setattr(loader, "_current_config_path", tmp_path / "config.json")
    report = DoctorReport(
        config_loaded=True,
        paths=PathsInfo(config_path="/test/config.json", config_exists=True, config_valid=True),
        routing=RoutingInfo(model="nano-gpt/m", provider="nano-gpt", max_tokens=None, context_window_tokens=None),
    )
    monkeypatch.setattr(doctor_commands, "_gather_static_checks", lambda: report)
    monkeypatch.setattr(doctor_commands, "_probe_memory", lambda _: MemoryInfo())
    app = typer.Typer()
    doctor_commands.register(app)

    text = re.sub(r"\s+", " ", CliRunner().invoke(app, []).stdout)

    assert "Max tokens: unknown" in text and "refuses a request with no ceiling" in text
    assert "Context win: unknown" in text and "history is not trimmed" in text
    # Both remedies name the row to declare the limit on, per model.
    assert "ddeharness provider model set nano-gpt m --max-tokens" in text
    assert "ddeharness provider model set nano-gpt m --context-window" in text


def test_the_reported_limits_are_the_ones_a_request_will_carry(monkeypatch):
    """Doctor walks the loop's own ladders: what the model's row declares first,
    then the model layer's row for it. Neither reads a table, and neither
    contacts a vendor."""
    rows = {"contextWindow": 128_000, "maxTokens": 16_384}
    monkeypatch.setattr(doctor_commands, "_catalogue_row", lambda *_a: rows)
    # A row that declares nothing but the id: there is no limit written, so the
    # model layer's answer is the one a request will carry.
    config = build_config(
        declared("custom", base_url="http://127.0.0.1:8000/v1", models=["qwen3-32b"], apiKey="k"),
        model="custom/qwen3-32b",
    )

    ceiling, window = doctor_commands._model_limits(config, "custom/qwen3-32b")

    assert (window.tokens, window.source) == (128_000, "model-service")
    assert (ceiling.tokens, ceiling.source) == (16_384, "model-service")

    # The same model with its limits written on its own row, which is where a
    # declaration lives now -- keyed by the row's ``id``, not by a spelling.
    written = build_config(
        declared(
            "custom",
            base_url="http://127.0.0.1:8000/v1",
            models=[{"id": "qwen3-32b", "contextWindow": 40_960, "maxTokens": 8_192}],
            apiKey="k",
        ),
        model="custom/qwen3-32b",
    )
    ceiling, window = doctor_commands._model_limits(written, "custom/qwen3-32b")

    assert (window.tokens, window.source) == (40_960, "declared")
    assert (ceiling.tokens, ceiling.source) == (8_192, "declared")


def test_limits_the_model_layer_cannot_report_read_as_unknown(monkeypatch):
    """No Node, no built bundle, a child that will not start: a diagnostic that
    cannot read a fact says so rather than failing or inventing one."""
    monkeypatch.setattr(doctor_commands, "_catalogue_row", lambda *_a: {})
    config = build_config(keyed("openai"), model="openai/gpt-4o")

    ceiling, window = doctor_commands._model_limits(config, "openai/gpt-4o")

    assert (window.tokens, window.source) == (None, "unknown")
    assert (ceiling.tokens, ceiling.source) == (None, "unknown")


def test_doctor_names_the_version_and_a_newer_release_on_pypi(monkeypatch, capsys):
    """The first thing doctor says is which OpenDDE Harness this is, and when
    PyPI has a newer one, what it is and the command that installs it here."""
    report = DoctorReport(
        paths=PathsInfo(config_path="/nowhere/config.json", config_exists=False),
        harness_version="0.0.3",
        update={"latest": "0.0.9", "command": "uv tool upgrade opendde-harness"},
    )
    doctor_commands._render_human_output(report)
    out = capsys.readouterr().out
    assert "Version" in out
    assert "0.0.3" in out and "↑ 0.0.9 is on PyPI" in out
    assert "update with: uv tool upgrade opendde-harness" in out

    report.update = None
    doctor_commands._render_human_output(report)
    assert "is on PyPI" not in capsys.readouterr().out


def test_doctors_json_report_carries_the_update(monkeypatch, tmp_path):
    from opendde_harness.cli import update_notice
    from opendde_harness.config import loader

    monkeypatch.setattr(loader, "_current_config_path", tmp_path / "config.json")
    monkeypatch.setattr(
        update_notice, "update_notice", lambda version, wait=0.0: ("0.0.9", "pip install --upgrade opendde-harness")
    )
    report = doctor_commands._gather_static_checks()
    assert report.harness_version
    assert report.update == {"latest": "0.0.9", "command": "pip install --upgrade opendde-harness"}
