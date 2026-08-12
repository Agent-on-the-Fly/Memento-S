# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from core.memento_s.skill_dispatch import SkillDispatcher


@pytest.mark.asyncio
async def test_unknown_tool_raises(real_dispatcher: SkillDispatcher):
    with pytest.raises(ValueError):
        await real_dispatcher.execute("totally_unknown_tool", {})
