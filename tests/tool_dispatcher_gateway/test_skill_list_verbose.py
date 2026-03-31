from __future__ import annotations

import json

import pytest

from core.memento_s.tool_dispatcher import ToolDispatcher


@pytest.mark.asyncio
async def test_skill_list_verbose(real_dispatcher: ToolDispatcher):
    raw = await real_dispatcher.execute("skill_list", {"verbose": True})
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["status"] == "success"
    assert isinstance(payload["output"], list)
    assert len(payload["output"]) > 0

    first = payload["output"][0]
    assert "name" in first
    assert "execution_mode" in first
    assert "governance" in first
