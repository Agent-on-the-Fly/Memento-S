# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest

from core.memento_s.skill_dispatch import SkillDispatcher


@pytest.mark.asyncio
async def test_skill_list_basic(real_dispatcher: SkillDispatcher):
    raw = await real_dispatcher.execute("skill_list", {"verbose": False})
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["status"] == "success"
    assert isinstance(payload["output"], list)
    assert len(payload["output"]) > 0
    assert {"name", "description"}.issubset(payload["output"][0].keys())
