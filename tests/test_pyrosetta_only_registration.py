"""Regression: legacy Rosetta HTTP is not part of executable routing."""
from pathlib import Path

import pytest
from opendde_harness.plugin.protein_design.core.model_registry import ModelRegistration, ProteinModelRegistry
from opendde_harness.plugin.protein_design.core.model_requests import ROSETTA_OPERATIONS


def test_no_http_rosetta_registration():
    with pytest.raises(ValueError, match="HTTP API has been retired"):
        ModelRegistration.model_validate({"kind": "rosetta", "service": {"url": "http://localhost:8050"}})


def test_direct_pyrosetta_registration():
    record = ModelRegistration.model_validate({
        "kind": "rosetta",
        "service": {"backend": "slurm", "python": "/usr/bin/python3"}
    })
    assert record.service.backend == "slurm"
    assert ROSETTA_OPERATIONS == ("interface_analyze",)


def test_rosetta_cannot_resolve_http_route():
    with pytest.raises(ValueError, match="no HTTP route"):
        ProteinModelRegistry.routes("rosetta", "interface_analyze")


def test_rossetta_standalone_tool_not_exported():
    manifest = (Path(__file__).resolve().parents[1] /
                "opendde_harness/plugin/protein_design/opendde-harness-plugin.toml").read_text()
    assert 'name = "protein_model_rosetta"' not in manifest
    assert 'name = "protein_design_start"' in manifest


def test_cluster_template_uses_direct_slurm():
    import yaml

    configuration = yaml.safe_load((Path(__file__).resolve().parents[1] /
                                    "docs/examples/cluster_deployment.yaml").read_text())
    assert configuration["model_services"]["rosetta"]["service"]["backend"] == "slurm"
