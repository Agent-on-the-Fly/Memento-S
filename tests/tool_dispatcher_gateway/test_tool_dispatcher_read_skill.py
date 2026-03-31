from __future__ import annotations

import json

import pytest

from core.memento_s.tool_dispatcher import ToolDispatcher


@pytest.mark.asyncio
async def test_read_skill_missing_name_real_gateway(real_dispatcher: ToolDispatcher):
    raw = await real_dispatcher.execute("read_skill", {})
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["error_code"] == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_read_skill_not_found_real_gateway(real_dispatcher: ToolDispatcher):
    raw = await real_dispatcher.execute(
        "read_skill",
        {"skill_name": "completely_nonexistent_skill_xyz"},
    )
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["error_code"] == "SKILL_NOT_FOUND"


@pytest.mark.asyncio
async def test_read_skill_success_real_gateway(real_dispatcher: ToolDispatcher):
    listed = json.loads(await real_dispatcher.execute("skill_list", {"verbose": False}))
    skill_name = listed["output"][0]["name"]

    raw = await real_dispatcher.execute("read_skill", {"skill_name": skill_name})
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["status"] == "success"
    assert payload["skill_name"] == skill_name
    assert isinstance(payload["output"], str)
    assert len(payload["output"]) > 0
