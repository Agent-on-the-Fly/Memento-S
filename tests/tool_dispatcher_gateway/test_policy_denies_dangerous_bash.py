from __future__ import annotations

import pytest

from core.memento_s.tool_dispatcher import ToolDispatcher


@pytest.mark.asyncio
async def test_policy_denies_dangerous_bash(real_dispatcher: ToolDispatcher):
    with pytest.raises(PermissionError):
        await real_dispatcher.execute(
            "bash_tool",
            {"command": "rm -rf /", "description": "dangerous command should be blocked"},
        )
