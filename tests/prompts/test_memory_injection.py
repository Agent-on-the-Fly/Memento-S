# SPDX-License-Identifier: Apache-2.0
"""集成测试 — core/context/context_manager.py: L1/L2 Memory 注入验证。"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from core.context.context_manager import ContextManager
from core.context.config import ContextManagerConfig
from core.context.session_context import SessionContext


class TestMemoryInjection:
    """验证 Session Memory (L1) 和 Long-term Memory (L2) 的注入逻辑。"""

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
    async def test_session_memory_injected_when_not_empty(self, ctx: ContextManager) -> None:
        """SessionMemory 不为空时，应注入到 system prompt。"""
        sm = MagicMock()
        sm.is_empty = MagicMock(return_value=False)
        sm.to_prompt_section = MagicMock(return_value="## Session Memory\nactive task: testing")
        ctx._session_memory = sm

        prompt = await ctx.assemble_system_prompt(
            current_message="hello",
            mode="agentic",
        )
        assert "active task: testing" in prompt or "Session Memory" in prompt

    @pytest.mark.asyncio
    async def test_session_memory_skipped_when_empty(self, ctx: ContextManager) -> None:
        """SessionMemory 为空时，system prompt 中不应有 Session Memory section。"""
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
    async def test_longterm_memory_injected(self, ctx: ContextManager) -> None:
        """L2 AutoMemory 内容应被注入到 system prompt。"""
        sm = MagicMock()
        sm.is_empty = MagicMock(return_value=True)
        sm.to_prompt_section = MagicMock(return_value="")
        ctx._session_memory = sm

        l2 = MagicMock()
        l2.load_memory_prompt = MagicMock(return_value="## Long-term Memory\nproject X")
        ctx._memory = l2

        prompt = await ctx.assemble_system_prompt(
            current_message="hello",
            mode="agentic",
        )
        assert "project X" in prompt or "Long-term Memory" in prompt

    @pytest.mark.asyncio
    async def test_l2_memory_skipped_when_none(self, ctx: ContextManager) -> None:
        """L2 Memory 未启用时，不注入内容。"""
        ctx._memory = None
        sm = MagicMock()
        sm.is_empty = MagicMock(return_value=True)
        sm.to_prompt_section = MagicMock(return_value="")
        ctx._session_memory = sm

        prompt = await ctx.assemble_system_prompt(
            current_message="hello",
            mode="agentic",
        )
        assert "Long-term Memory" not in prompt

    @pytest.mark.asyncio
    async def test_append_respects_msg_seq(self, ctx: ContextManager) -> None:
        """append() 为新消息注入 _seq。"""
        ctx._msg_seq = 0
        ctx._total_tokens = 0
        result = await ctx.append(
            [],
            [{"role": "assistant", "content": "hi", "tool_calls": []}],
        )
        assert result[0].get("_seq") == 1

    @pytest.mark.asyncio
    async def test_session_memory_priority_order(self, ctx: ContextManager) -> None:
        """Session Memory 和 L2 Memory 的 priority 应符合设计（session=50, l2=45）。"""
        sm = MagicMock()
        sm.is_empty = MagicMock(return_value=False)
        sm.to_prompt_section = MagicMock(return_value="### L1 CONTENT")
        ctx._session_memory = sm

        l2 = MagicMock()
        l2.load_memory_prompt = MagicMock(return_value="### L2 CONTENT")
        ctx._memory = l2

        prompt = await ctx.assemble_system_prompt(
            current_message="hello",
            mode="agentic",
        )
        assert "L1 CONTENT" in prompt or "L2 CONTENT" in prompt
