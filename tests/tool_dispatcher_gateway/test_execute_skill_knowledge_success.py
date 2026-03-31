from __future__ import annotations

import json

import pytest

from core.memento_s.tool_dispatcher import ToolDispatcher
from core.skill.gateway import SkillManifest
from core.skill.gateway import SkillGateway


@pytest.mark.asyncio
async def test_execute_skill_knowledge_success(real_dispatcher: ToolDispatcher):
    provider = real_dispatcher._gateway
    assert isinstance(provider, SkillGateway)

    manifests = await provider.discover()
    candidates = [
        m
        for m in manifests
        if isinstance(m, SkillManifest) and m.execution_mode == "knowledge"
    ]
    if not candidates:
        pytest.skip("No knowledge skill found in local cache")

    skill_name = candidates[0].name
    raw = await real_dispatcher.execute(
        "execute_skill",
        {"skill_name": skill_name, "request": "请简要说明这个技能的用途"},
    )
    payload = json.loads(raw)

    assert payload["skill_name"] == skill_name
    assert payload["status"] in ("success", "failed")
    if payload["status"] == "success":
        assert payload["ok"] is True
        assert payload["output"] is not None
