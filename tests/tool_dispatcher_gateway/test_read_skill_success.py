from __future__ import annotations

import json

import pytest

from core.memento_s.tool_dispatcher import ToolDispatcher


@pytest.mark.asyncio
async def test_read_skill_success(real_dispatcher: ToolDispatcher):
    manifests = json.loads(await real_dispatcher.execute("skill_list", {"verbose": False}))[
        "output"
    ]
    skill_name = manifests[0]["name"]

    raw = await real_dispatcher.execute("read_skill", {"skill_name": skill_name})
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["status"] == "success"
    assert payload["skill_name"] == skill_name
    assert isinstance(payload["output"], str)
    assert len(payload["output"]) > 0
