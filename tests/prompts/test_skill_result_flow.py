# SPDX-License-Identifier: Apache-2.0
"""集成测试 — ContextManager.persist_tool_result + ArtifactStore: tool message 格式 + seq 注入。"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from core.context.context_manager import ContextManager
from core.context.config import ContextManagerConfig
from core.context.session_context import SessionContext
from infra.compact.storage import ArtifactStore, FileStorageBackend


class TestSkillResultFlow:
    """验证 skill 执行结果 → tool message → state.messages 的完整链路。"""

    @pytest.mark.asyncio
    async def test_persist_tool_result_small(self, tmp_path: Path) -> None:
        """小结果（< threshold）不落盘，直接返回。"""
        store = ArtifactStore(session_dir=tmp_path)
        result = await store.process_tool_result(
            tool_call_id="c1",
            tool_name="read",
            content="short content",
            remaining_budget_tokens=5000,
        )
        assert result["role"] == "tool"
        assert result["_persisted"] is False
        assert result["content"] == "short content"

    @pytest.mark.asyncio
    async def test_persist_tool_result_large_persisted(self, tmp_path: Path) -> None:
        """大结果（> threshold）落盘，内容被 preview 替换。"""
        store = ArtifactStore(session_dir=tmp_path)
        large = "x" * 20000
        result = await store.process_tool_result(
            tool_call_id="c2",
            tool_name="execute_skill",
            content=large,
            remaining_budget_tokens=100,
        )
        assert result["_persisted"] is True
        assert result["content"] != large
        assert "[Output persisted" in result["content"]
        assert (tmp_path / "artifacts" / "c2.txt").exists()

    @pytest.mark.asyncio
    async def test_read_artifact_roundtrip(self, tmp_path: Path) -> None:
        """落盘后可完整读取回来。"""
        store = ArtifactStore(session_dir=tmp_path)
        original = "important data " * 500
        await store.process_tool_result(
            "c3", "write", original, remaining_budget_tokens=50
        )
        read_back = await store.read_artifact("c3")
        assert read_back == original

    @pytest.mark.asyncio
    async def test_backend_has_after_persist(self, tmp_path: Path) -> None:
        """FileStorageBackend.has() 在 persist 后返回 True。"""
        backend = FileStorageBackend(session_dir=tmp_path)
        await backend.persist("data", {"tool_call_id": "fid1"})
        assert await backend.has("fid1") is True

    @pytest.mark.asyncio
    async def test_context_manager_persist(self, mock_g_config_paths, mock_session_ctx) -> None:
        """ContextManager.persist_tool_result() 委托 ArtifactStore 处理。"""
        cfg = ContextManagerConfig()
        ctx = ContextManager(
            ctx=mock_session_ctx,
            config=cfg,
            skill_gateway=None,
        )
        ctx._total_tokens = 100

        result = await ctx.persist_tool_result(
            "call_abc", "read", "data content"
        )
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_abc"
        assert result["name"] == "read"

    @pytest.mark.asyncio
    async def test_persist_tool_result_contains_role_and_tool_call_id(self, tmp_path: Path) -> None:
        """process_tool_result 返回的 dict 包含必需字段。"""
        store = ArtifactStore(session_dir=tmp_path)
        result = await store.process_tool_result(
            "c5", "execute_skill",
            json.dumps({"ok": True, "summary": "done"}),
            remaining_budget_tokens=500,
        )
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "c5"
        assert result["name"] == "execute_skill"
        assert "content" in result
        assert "_persisted" in result
