from __future__ import annotations

import json

import pytest

from core.memento_s.tool_dispatcher import ToolDispatcher


@pytest.mark.asyncio
async def test_read_skill_not_found(real_dispatcher: ToolDispatcher):
    raw = await real_dispatcher.execute(
        "read_skill", {"skill_name": "completely_nonexistent_skill_xyz"}
    )
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["error_code"] == "SKILL_NOT_FOUND"
