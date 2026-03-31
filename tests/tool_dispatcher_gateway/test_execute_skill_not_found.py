from __future__ import annotations

import json

import pytest

from core.memento_s.tool_dispatcher import ToolDispatcher


@pytest.mark.asyncio
async def test_execute_skill_not_found(real_dispatcher: ToolDispatcher):
    raw = await real_dispatcher.execute(
        "execute_skill",
        {"skill_name": "completely_nonexistent_skill_xyz", "request": "hello"},
    )
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["status"] in ("blocked", "failed")
    assert payload["error_code"] in ("SKILL_NOT_FOUND", "INVALID_INPUT", "INTERNAL_ERROR")
