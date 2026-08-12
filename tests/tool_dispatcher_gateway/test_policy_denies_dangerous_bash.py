# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from core.memento_s.skill_dispatch import SkillDispatcher


@pytest.mark.asyncio
async def test_policy_denies_dangerous_bash(real_dispatcher: SkillDispatcher):
    with pytest.raises(PermissionError):
        await real_dispatcher.execute(
            "bash_tool",
            {"command": "rm -rf /", "description": "dangerous command should be blocked"},
        )
