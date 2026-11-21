"""Automatically registered Qwen tool for network-free scientific database search."""
from __future__ import annotations

import asyncio
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from opendde_harness.agent.tools.base import Tool, ToolResult
from opendde_harness.plugin.context import PluginContext
from ..scientific_agent.backends.web_search_queue import search_queue


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["rcsb", "uniprot", "pubmed", "sabdab"] | None = None
    query: str | None = Field(default=None, max_length=300)
    request_id: str | None = None
    limit: int = Field(default=10, ge=1, le=20)
    wait_seconds: int = Field(default=5, ge=0, le=30)

    @model_validator(mode="after")
    def check_mode(self):
        if self.request_id:
            if self.source is not None or self.query is not None:
                raise ValueError("polling requires request_id alone; omit source and query")
        elif not self.source or not self.query or not self.query.strip():
            raise ValueError("source and query are required")
        return self


class ScientificWebSearchTool(Tool):
    name = "scientific_web_search"
    description = (
        "Search real scientific databases through an offline /lustre queue (RCSB structures, "
        "UniProt sequences, PubMed literature, SAbDab2 antibody-antigen entries). "
        "Submit with source and query; call again with request_id ONLY if status is pending. "
        "Search uses an external login/service-node worker; this tool never needs HTTP or "
        "internet connectivity on a compute node. Report only returned sources and results."
    )

    @property
    def parameters(self):
        return SearchInput.model_json_schema()

    async def execute(self, **kwargs):
        try:
            value = SearchInput.model_validate(kwargs)
            result = await asyncio.to_thread(search_queue, **value.model_dump())
            return ToolResult(json.dumps(result, ensure_ascii=False, allow_nan=False),
                              "Scientific Web Search", retryable=False)
        except Exception as exc:
            return ToolResult(json.dumps({"status": "error", "error": str(exc)}),
                              "Scientific Web Search failed", retryable=False)


def make_web_search_tool(ctx: PluginContext):
    return ScientificWebSearchTool()
