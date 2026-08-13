# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest

from core.memento_s.skill_dispatch import SkillDispatcher


@pytest.mark.asyncio
async def test_skill_install_missing_name_real_gateway(real_dispatcher: SkillDispatcher):
    raw = await real_dispatcher.execute("skill_install", {})
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["error_code"] == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_unknown_tool_raises_real_gateway(real_dispatcher: SkillDispatcher):
    payload = json.loads(await real_dispatcher.execute("totally_unknown_tool", {}))
    assert payload["error_code"] == "UNKNOWN_TOOL"


@pytest.mark.asyncio
async def test_policy_denies_dangerous_bash_real_gateway(real_dispatcher: SkillDispatcher):
    with pytest.raises(PermissionError):
        await real_dispatcher.execute(
            "bash_tool",
            {
                "command": "rm -rf /",
                "description": "dangerous command",
            },
        )
