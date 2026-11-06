"""Register all documented services in the actual config without losing Qwen settings."""

import json
from pathlib import Path

from typer.testing import CliRunner

from opendde_harness.cli.protein_design_commands import protein_design_app


def invoke(path, *args):
    return CliRunner().invoke(protein_design_app, ["models", *args, "--opendde-config", str(path)])


def test_registration_merges_five_services_preserving_provider_and_workspace(tmp_path):
    path = tmp_path / "config.json"
    original = {
        "providers": {"qwen": {"apiKey": "synthetic-key", "baseUrl": "https://qwen.test"}},
        "agents": {"defaults": {"workspace": "/workspace/test"}},
        "plugins": {"config": {"protein-design": {"compute_url": "http://compute.test"}, "other": {"setting": True}}},
    }
    path.write_text(json.dumps(original))
    result = invoke(path, "register", "--local-defaults")
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert len(summary["registered_aliases"]) == 5
    data = json.loads(path.read_text())
    assert data["providers"] == original["providers"]
    assert data["agents"] == original["agents"]
    assert data["plugins"]["config"]["other"] == {"setting": True}
    assert data["plugins"]["config"]["protein-design"]["compute_url"] == "http://compute.test"
    models = data["plugins"]["config"]["protein-design"]["model_services"]
    assert models["torchfold"]["service"]["url"].endswith(":8050")
    assert "synthetic-key" not in result.output
    backup = Path(summary["backup_path"])
    assert json.loads(backup.read_text()) == original
    assert backup.stat().st_mode & 0o077 == 0
    assert path.stat().st_mode & 0o077 == 0


def test_repeated_default_registration_keeps_custom_alias_and_endpoint(tmp_path):
    path = tmp_path / "config.json"
    custom = {"kind": "torchcraft", "service": {"url": "http://custom.test:9999"}}
    path.write_text(json.dumps({"plugins": {"config": {"protein-design": {"model_services": {"mycraft": custom}}}}}))
    assert invoke(path, "register", "--local-defaults").exit_code == 0
    before = path.read_bytes()
    repeated = invoke(path, "register", "--local-defaults")
    assert repeated.exit_code == 0
    assert json.loads(repeated.output)["changed_aliases"] == []
    assert path.read_bytes() == before
    models = json.loads(before)["plugins"]["config"]["protein-design"]["model_services"]
    assert len(models) == 5 and "torchcraft" not in models and models["mycraft"] == custom


def test_register_requires_explicit_endpoint_source(tmp_path):
    path = tmp_path / "config.json"
    assert invoke(path, "register").exit_code == 1
    assert not path.exists()


def test_import_documented_fragment_and_list_active_file(tmp_path):
    path = tmp_path / "instance/config.json"
    source = Path(__file__).parents[1] / "docs/examples/model_services.json"
    result = invoke(path, "register", "--services-file", str(source))
    assert result.exit_code == 0, result.output
    listing = invoke(path, "list")
    assert listing.exit_code == 0, listing.output
    payload = json.loads(listing.output)
    assert payload["active_config_path"] == str(path.resolve())
    assert len(payload["models"]) == 5 and payload["unconfigured_model_kinds"] == []
    assert not payload["models"][0]["inference_verified"]


def test_empty_registry_identifies_config_and_missing_models(tmp_path):
    path = tmp_path / "missing.json"
    result = invoke(path, "list")
    assert result.exit_code == 0
    value = json.loads(result.output)
    assert value["models"] == []
    assert set(value["unconfigured_model_kinds"]) == {"af3score", "torchfold", "torchcraft", "ppiflow", "rosetta"}
    assert value["active_config_path"] == str(path)
    assert not path.exists()


def test_registration_does_not_silently_enable_disabled_plugin(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"plugins": {"disabled": ["protein-design"]}}))
    result = invoke(path, "register", "--local-defaults")
    assert result.exit_code == 0 and not json.loads(result.output)["plugin_enabled"]
    assert json.loads(path.read_text())["plugins"]["disabled"] == ["protein-design"]


def test_bad_existing_json_is_never_replaced(tmp_path):
    for content in ("not json", "[]", "null", '{"plugins":true}', '{"plugins":{"disabled":true}}'):
        path = tmp_path / "config.json"
        path.write_text(content)
        assert invoke(path, "register", "--local-defaults").exit_code == 1
        assert path.read_text() == content


def test_conflicting_import_requires_explicit_replacement(tmp_path):
    path = tmp_path / "config.json"
    assert invoke(path, "register", "--local-defaults").exit_code == 0
    incoming = tmp_path / "models.json"
    incoming.write_text(json.dumps({"af3score": {"kind": "af3score", "service": {"url": "https://replacement.test"}}}))
    before = path.read_bytes()
    assert invoke(path, "register", "--services-file", str(incoming)).exit_code == 1
    assert path.read_bytes() == before
    result = invoke(path, "register", "--services-file", str(incoming), "--replace-existing")
    assert result.exit_code == 0
    assert (
        json.loads(path.read_text())["plugins"]["config"]["protein-design"]["model_services"]["af3score"]["service"][
            "url"
        ]
        == "https://replacement.test"
    )
