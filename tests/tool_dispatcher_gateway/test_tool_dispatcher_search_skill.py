from __future__ import annotations

import json

import pytest

from core.memento_s.tool_dispatcher import ToolDispatcher
from core.memento_s.tool_schemas import AGENT_TOOL_SCHEMAS


def test_agent_tool_schemas_only_search_and_execute():
    tool_names = [item["function"]["name"] for item in AGENT_TOOL_SCHEMAS]
    assert tool_names == ["search_skill", "execute_skill"]


@pytest.mark.asyncio
async def test_search_skill_missing_query_real_gateway(real_dispatcher: ToolDispatcher):
    raw = await real_dispatcher.execute("search_skill", {})
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["error_code"] == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_search_skill_success_real_gateway(real_dispatcher: ToolDispatcher):
    raw = await real_dispatcher.execute(
        "search_skill",
        {"query": "web search", "k": 5},
    )
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["status"] == "success"
    assert isinstance(payload["output"], list)
    assert payload["metrics"]["k"] == 5
    assert payload["metrics"]["cloud_count"] == len(payload["output"])
    assert payload["diagnostics"]["local_in_context"] >= 0

    for item in payload["output"]:
        assert "name" in item
        assert "description" in item
        assert "source" in item
