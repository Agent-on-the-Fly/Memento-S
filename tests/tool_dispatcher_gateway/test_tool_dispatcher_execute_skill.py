from __future__ import annotations

import json

import pytest

from core.memento_s.tool_dispatcher import ToolDispatcher
from core.skill.gateway import SkillManifest
from core.skill.gateway import SkillGateway


@pytest.mark.asyncio
async def test_execute_skill_missing_name_real_gateway(real_dispatcher: ToolDispatcher):
    raw = await real_dispatcher.execute("execute_skill", {"request": "hello"})
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["error_code"] == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_execute_skill_not_found_real_gateway(real_dispatcher: ToolDispatcher):
    # Unknown skill without prior search_skill → SEARCH_REQUIRED
    raw = await real_dispatcher.execute(
        "execute_skill",
        {"skill_name": "completely_nonexistent_skill_xyz", "request": "hello"},
    )
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["status"] in ("blocked", "failed")
    assert payload["error_code"] in (
        "SKILL_NOT_FOUND",
        "INVALID_INPUT",
        "INTERNAL_ERROR",
        "SEARCH_REQUIRED",
    )


@pytest.mark.asyncio
async def test_execute_skill_knowledge_success_real_gateway(
    real_dispatcher: ToolDispatcher,
):
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
