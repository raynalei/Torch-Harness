"""Regression for automatically registered scientific web search."""
import json
from opendde_harness.plugin.protein_design.agents.skills import ProteinDesignSkillCatalog
from opendde_harness.plugin.protein_design.scientific_agent.backends.web_search_queue import search_queue
from opendde_harness.plugin.protein_design.tools.web_search import ScientificWebSearchTool


def test_skill_catalog_no_legacy_lookups():
    names = set(ProteinDesignSkillCatalog.builtin()._documents)
    assert "scientific-web-search" in names
    assert not names.intersection({"pdb-lookup", "uniprot-lookup", "pubmed-lookup"})


def test_submit_complete_and_reuse_request_id(tmp_path):
    item = search_queue(source="sabdab", query="erbB-2", limit=2, root=tmp_path)
    assert item["status"] == "pending"
    reqid = item["request_id"]
    queued = json.loads((tmp_path / "requests" / (reqid + ".json")).read_text())
    assert queued["query"] == "erbB-2"
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / (reqid + ".json")).write_text(
        json.dumps({"status": "completed", "request_id": reqid, "data": {"hits": [{"id": "test"}]}})
    )
    done = search_queue(request_id=reqid, root=tmp_path)
    assert done["data"]["hits"][0]["id"] == "test"
    assert len(list((tmp_path / "requests").glob("*.json"))) == 1


async def test_tool_writes_queue(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_WEB_ROOT", str(tmp_path))
    tool = ScientificWebSearchTool()
    output = await tool.execute(source="pubmed", query="IL7R antibody", wait_seconds=0)
    result = json.loads(output.model_text)
    assert result["status"] == "pending"
    assert (tmp_path / "requests" / (result["request_id"] + ".json")).exists()
    invalid = await tool.execute(source="fake", query="x")
    assert json.loads(invalid.model_text)["status"] == "error"
