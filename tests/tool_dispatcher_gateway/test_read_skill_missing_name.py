from __future__ import annotations

import json

import pytest

from core.memento_s.tool_dispatcher import ToolDispatcher


@pytest.mark.asyncio
async def test_read_skill_missing_name(real_dispatcher: ToolDispatcher):
    raw = await real_dispatcher.execute("read_skill", {})
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["error_code"] == "INVALID_INPUT"
