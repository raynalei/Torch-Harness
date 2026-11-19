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
    assert len(summary["registered_aliases"]) == 7
    data = json.loads(path.read_text())
    assert data["providers"] == original["providers"]
    assert data["agents"] == original["agents"]
    assert data["plugins"]["config"]["other"] == {"setting": True}
    assert data["plugins"]["config"]["protein-design"]["compute_url"] == "http://compute.test"
    models = data["plugins"]["config"]["protein-design"]["model_services"]
    assert models["torchfold"]["service"]["url"].endswith(":8050")
    assert models["torchscore"]["kind"] == "torchscore"
    assert models["torchscore"]["service"]["url"].endswith(":8070")
    assert models["torchopt"]["service"]["url"].endswith(":8060")
    assert models["torchopt"]["service"]["status_field"] == "state"
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
    assert len(models) == 7 and "torchcraft" not in models and models["mycraft"] == custom


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
    assert len(payload["models"]) == 7 and payload["unconfigured_model_kinds"] == []
    assert not payload["models"][0]["inference_verified"]


def test_empty_registry_identifies_config_and_missing_models(tmp_path):
    path = tmp_path / "missing.json"
    result = invoke(path, "list")
    assert result.exit_code == 0
    value = json.loads(result.output)
    assert value["models"] == []
    assert set(value["unconfigured_model_kinds"]) == {
        "af3score",
        "torchscore",
        "torchopt",
        "torchfold",
        "torchcraft",
        "ppiflow",
        "rosetta",
    }
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


def test_import_persistent_defaults_preserves_provider_and_existing_models(tmp_path):
    path = tmp_path / "config.json"
    original = {
        "providers": {"example": {"api_key": "preserve-me"}},
        "plugins": {
            "config": {
                "protein-design": {
                    "model_services": {"score": {"kind": "torchscore", "service": {"url": "https://score.test"}}}
                }
            }
        },
    }
    path.write_text(json.dumps(original))
    source = tmp_path / "settings.json"
    source.write_text(
        json.dumps(
            {
                "model_services": {
                    "flow": {
                        "kind": "ppiflow",
                        "service": {"url": "https://flow.test", "environment": {"PPIFLOW_ROOT": "/shared/source"}},
                        "request_defaults": {"model_weights": "/shared/weights.ckpt", "num_timesteps": 100},
                    }
                },
                "workflow_defaults": {
                    "work_dir": "/shared/runs",
                    "path_mappings": [{"local": "/shared", "remote": "/shared"}],
                    "evaluation": {"dockq_python": "/shared/dockq/bin/python"},
                },
            }
        )
    )
    result = invoke(path, "register", "--services-file", str(source))
    assert result.exit_code == 0, result.output
    saved = json.loads(path.read_text())
    assert saved["providers"] == original["providers"]
    section = saved["plugins"]["config"]["protein-design"]
    assert set(section["model_services"]) == {"score", "flow"}
    assert section["workflow_defaults"]["work_dir"] == "/shared/runs"
    assert json.loads(result.output)["backup_path"]
    assert json.loads(invoke(path, "list").output)["workflow_defaults_configured"] is True


def test_import_workflow_once_discards_previous_target_fields(tmp_path):
    path = tmp_path / "config.json"
    profile = tmp_path / "old-task.json"
    profile.write_text(
        json.dumps(
            {
                "name": "old",
                "work_dir": "/shared/runs",
                "path_mappings": [{"local": "/shared", "remote": "/shared"}],
                "evaluation": {"dockq_python": "/shared/dockq/python"},
                "generators": {
                    "ppiflow": {
                        "request": {
                            "mode": "nanobody",
                            "name": "old",
                            "antigen_pdb": "/shared/old.pdb",
                            "specified_hotspots": "A1",
                            "framework_dir": "/shared/framework",
                            "fixed_positions": "B2",
                            "cdr_position": "B3-5",
                            "model_weights": "/shared/weights.ckpt",
                        },
                        "chains": {"target": ["A"], "binder": ["B"], "refold_map": {"A": "A", "B": "B"}},
                        "fold_template": "/shared/old-fold.json",
                    }
                },
            }
        )
    )
    result = invoke(path, "defaults", "--from-workflow", str(profile), "--design-type", "vhh")
    assert result.exit_code == 0, result.output
    section = json.loads(path.read_text())["plugins"]["config"]["protein-design"]
    generator = section["design_profiles"]["vhh"]["generators"]["ppiflow"]
    assert not {"framework_dir", "fixed_positions", "cdr_position"} & generator["request"].keys()
    assert "antigen_pdb" not in generator["request"]
    assert "specified_hotspots" not in generator["request"]
    assert generator["request"]["model_weights"] == "/shared/weights.ckpt"
    assert generator["fold_template"] is None
    assert section["workflow_defaults"]["evaluation"]["scoring_backend"] == "torchscore"
    assert "autonomous" in section["workflow_defaults"]


def test_replacing_static_defaults_requires_explicit_flag(tmp_path):
    path = tmp_path / "config.json"
    original = {"plugins": {"config": {"protein-design": {"workflow_defaults": {"work_dir": "/shared/old"}}}}}
    path.write_text(json.dumps(original))
    source = tmp_path / "settings.json"
    source.write_text(json.dumps({"workflow_defaults": {"work_dir": "/shared/new"}}))
    assert invoke(path, "register", "--services-file", str(source)).exit_code == 1
    assert json.loads(path.read_text()) == original
    assert invoke(path, "register", "--services-file", str(source), "--replace-existing").exit_code == 0


def test_cluster_deployment_yaml_requires_native_contract_and_imports_when_completed(tmp_path):
    from pathlib import Path

    source = tmp_path / "deployment.yaml"
    content = (Path(__file__).resolve().parents[1] / "docs/examples/cluster_deployment.yaml").read_text()
    source.write_text(content)
    path = tmp_path / "config.json"
    result = invoke(path, "register", "--services-file", str(source))
    assert result.exit_code == 1
    assert not path.exists()
    content = content.replace("__REQUIRED_TORCHCRAFT_HOTSPOT_NUMBERING__", "sequence_0_based")
    content = content.replace("__REQUIRED_TORCHCRAFT_CSV_ID_COLUMN__", "id")
    content = content.replace("__REQUIRED_TORCHCRAFT_CSV_SEQUENCE_COLUMN__", "sequence")
    source.write_text(content)
    result = invoke(path, "register", "--services-file", str(source))
    assert result.exit_code == 0, result.output
    settings = json.loads(path.read_text())["plugins"]["config"]["protein-design"]
    assert len(settings["model_services"]) == 7
    assert (
        settings["workflow_defaults"]["evaluation"]["dockq_python"]
        == "/lustre/grp/cmclab/leiy/conda_envs/ppi_env/bin/python"
    )
    assert all(not model["request_defaults"] for model in settings["model_services"].values())
