from __future__ import annotations

import json

import pytest

from core.memento_s.tool_dispatcher import ToolDispatcher


@pytest.mark.asyncio
async def test_skill_install_missing_name_real_gateway(real_dispatcher: ToolDispatcher):
    raw = await real_dispatcher.execute("skill_install", {})
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["error_code"] == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_unknown_tool_raises_real_gateway(real_dispatcher: ToolDispatcher):
    with pytest.raises(ValueError, match="Unknown tool"):
        await real_dispatcher.execute("totally_unknown_tool", {})


@pytest.mark.asyncio
async def test_policy_denies_dangerous_bash_real_gateway(real_dispatcher: ToolDispatcher):
    with pytest.raises(PermissionError):
        await real_dispatcher.execute(
            "bash_tool",
            {
                "command": "rm -rf /",
                "description": "dangerous command",
            },
        )
