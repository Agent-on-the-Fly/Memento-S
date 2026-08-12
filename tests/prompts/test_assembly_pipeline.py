# SPDX-License-Identifier: Apache-2.0
"""集成测试 — core/context/context_manager.py: ContextManager 组装与 append 逻辑。"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from core.context.context_manager import ContextManager
from core.context.config import ContextManagerConfig
from core.context.session_context import SessionContext


class TestSystemPromptAssembly:
    """验证 ContextManager.assemble_system_prompt() 的各 priority 段落顺序。"""

    @pytest.fixture
    def ctx(self, mock_g_config_paths, mock_session_ctx) -> ContextManager:
        cfg = ContextManagerConfig()
        ctx = ContextManager(
            ctx=mock_session_ctx,
            config=cfg,
            skill_gateway=None,
        )
        return ctx

    @pytest.mark.asyncio
    async def test_assemble_returns_string(self, ctx: ContextManager) -> None:
        """assemble_system_prompt() 返回非空字符串。"""
        result = await ctx.assemble_system_prompt(
            current_message="hello",
            mode="agentic",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_assemble_with_different_modes(self, ctx: ContextManager) -> None:
        """direct 和 agentic 模式输出应不同。"""
        p_direct = await ctx.assemble_system_prompt(
            current_message="hello",
            mode="direct",
        )
        p_agentic = await ctx.assemble_system_prompt(
            current_message="hello",
            mode="agentic",
        )
        assert p_direct != p_agentic

    @pytest.mark.asyncio
    async def test_assemble_with_mock_sessions(self, ctx: ContextManager) -> None:
        """Mock Session Memory 和 L2 Memory 注入。"""
        sm = MagicMock()
        sm.is_empty = MagicMock(return_value=False)
        sm.to_prompt_section = MagicMock(return_value="## Session Memory\nactive task")
        ctx._session_memory = sm

        l2 = MagicMock()
        l2.load_memory_prompt = MagicMock(return_value="## Long-term Memory\nproj info")
        ctx._memory = l2

        prompt = await ctx.assemble_system_prompt(
            current_message="hello",
            mode="agentic",
        )
        assert "active task" in prompt or "Session Memory" in prompt

    @pytest.mark.asyncio
    async def test_session_memory_skipped_when_empty(self, ctx: ContextManager) -> None:
        """SessionMemory 为空时不应出现在 system prompt 中。"""
        sm = MagicMock()
        sm.is_empty = MagicMock(return_value=True)
        sm.to_prompt_section = MagicMock(return_value="")
        ctx._session_memory = sm
        ctx._memory = None

        prompt = await ctx.assemble_system_prompt(
            current_message="hello",
            mode="agentic",
        )
        assert "## Session Memory" not in prompt

    @pytest.mark.asyncio
    async def test_l2_memory_injected(self, ctx: ContextManager) -> None:
        """L2 AutoMemory 内容应被注入到 system prompt。"""
        l2 = MagicMock()
        l2.load_memory_prompt = MagicMock(return_value="## Long-term Memory\nproject X")
        ctx._memory = l2
        sm = MagicMock()
        sm.is_empty = MagicMock(return_value=True)
        sm.to_prompt_section = MagicMock(return_value="")
        ctx._session_memory = sm

        prompt = await ctx.assemble_system_prompt(
            current_message="hello",
            mode="agentic",
        )
        assert "project X" in prompt or "Long-term Memory" in prompt


class TestAppend:
    """验证 ContextManager.append() 的纯追加 + _seq 注入逻辑。"""

    @pytest.mark.asyncio
    async def test_append_increments_seq(self, mock_g_config_paths, mock_session_ctx) -> None:
        """append() 为新消息注入递增的 _seq。"""
        cfg = ContextManagerConfig()
        ctx = ContextManager(
            ctx=mock_session_ctx,
            config=cfg,
            skill_gateway=None,
        )
        ctx._msg_seq = 0
        ctx._total_tokens = 0

        msgs = [{"role": "system", "content": "sys"}]
        new_msgs = [
            {"role": "assistant", "content": "hi", "tool_calls": []},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]

        result = await ctx.append(msgs, new_msgs)
        assert len(result) == 3
        assert result[1].get("_seq") == 1
        assert result[2].get("_seq") == 2

    @pytest.mark.asyncio
    async def test_append_does_not_mutate_original(self, mock_g_config_paths, mock_session_ctx) -> None:
        """append() 返回新列表，不修改原始 messages。"""
        cfg = ContextManagerConfig()
        ctx = ContextManager(
            ctx=mock_session_ctx,
            config=cfg,
            skill_gateway=None,
        )
        ctx._msg_seq = 0
        ctx._total_tokens = 0

        original = [{"role": "system", "content": "sys"}]
        new_msgs = [{"role": "assistant", "content": "hi", "tool_calls": []}]

        result = await ctx.append(original, new_msgs)
        assert original == [{"role": "system", "content": "sys"}]
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_append_seq_continues(self, mock_g_config_paths, mock_session_ctx) -> None:
        """连续多次 append() 的 _seq 连续递增。"""
        cfg = ContextManagerConfig()
        ctx = ContextManager(
            ctx=mock_session_ctx,
            config=cfg,
            skill_gateway=None,
        )
        ctx._msg_seq = 0
        ctx._total_tokens = 0

        r1 = await ctx.append([], [{"role": "assistant", "content": "a", "tool_calls": []}])
        r2 = await ctx.append(r1, [{"role": "assistant", "content": "b", "tool_calls": []}])

        assert r1[0].get("_seq") == 1
        assert r2[1].get("_seq") == 2
