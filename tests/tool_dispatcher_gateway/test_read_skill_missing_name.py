# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest

from core.memento_s.skill_dispatch import SkillDispatcher


@pytest.mark.asyncio
async def test_read_skill_missing_name(real_dispatcher: SkillDispatcher):
    raw = await real_dispatcher.execute("read_skill", {})
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["error_code"] == "INVALID_INPUT"
