from __future__ import annotations

import pytest

from core.memento_s.tool_dispatcher import ToolDispatcher


@pytest.mark.asyncio
async def test_unknown_tool_raises(real_dispatcher: ToolDispatcher):
    with pytest.raises(ValueError):
        await real_dispatcher.execute("totally_unknown_tool", {})
