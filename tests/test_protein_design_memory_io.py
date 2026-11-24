"""Design memory uses compact queries and durable, deferred appends."""

import json

import httpx
import pytest

from opendde_harness.plugin.memory.longterm.backend import _HttpMemoryAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("app_id", ["protein-design", "opendde_harness"])
@pytest.mark.parametrize("is_final", [False, True])
async def test_design_append_defers_extraction_until_flush(app_id, is_final):
    requests = []

    def respond(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"data": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        adapter = _HttpMemoryAdapter("http://memory", client=client)
        await adapter.memorize("run", [{"content": "case"}], app_id=app_id, project_id="target", is_final=is_final)
    assert requests[0][0] == "/api/v2/memory/add"
    assert requests[0][1].get("defer_extraction", False) is (app_id == "protein-design")
    assert len(requests) == (2 if is_final else 1)
    if is_final:
        assert requests[1] == (
            "/api/v2/memory/flush",
            {"session_id": "run", "app_id": app_id, "project_id": "target"},
        )
