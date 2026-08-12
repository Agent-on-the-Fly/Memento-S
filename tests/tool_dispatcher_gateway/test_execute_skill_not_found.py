# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest

from core.memento_s.skill_dispatch import SkillDispatcher


@pytest.mark.asyncio
async def test_execute_skill_not_found(real_dispatcher: SkillDispatcher):
    raw = await real_dispatcher.execute(
        "execute_skill",
        {"skill_name": "completely_nonexistent_skill_xyz", "request": "hello"},
    )
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["status"] in ("blocked", "failed")
    assert payload["error_code"] in ("SKILL_NOT_FOUND", "INVALID_INPUT", "INTERNAL_ERROR")
