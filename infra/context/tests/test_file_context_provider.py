# SPDX-License-Identifier: Apache-2.0
"""Tests for infra/context/impl/file_context.py — FileContextProvider.

这些测试通过实际 ContextManager、SessionMemory 和 ContextMemory（使用 tmp 目录）
进行完整的 FileContextProvider 集成测试。LLM 调用被 mock。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from infra.context.factory import ContextFactoryConfig, create_context
from infra.context.impl.file_context import FileContextProvider


def _make_ctx_cfg(session_id: str, tmp_path: Path) -> ContextFactoryConfig:
    """Helper: 创建带显式路径的 ContextFactoryConfig。"""
    return ContextFactoryConfig(
        session_id=session_id,
        session_dir=tmp_path / "session",
        data_dir=tmp_path,
    )


class TestFileContextProvider:
    """Test FileContextProvider wrapping real ContextManager."""

    @pytest.fixture
    def provider(self, tmp_path: Path) -> FileContextProvider:
        cfg = _make_ctx_cfg("test-file-ctx", tmp_path)
        provider = create_context(cfg)
        assert isinstance(provider, FileContextProvider)
        return provider

    @pytest.fixture
    def provider_with_memory(self, tmp_path: Path) -> FileContextProvider:
        cfg = _make_ctx_cfg("test-file-ctx-mem", tmp_path)
        provider = create_context(cfg)
        assert isinstance(provider, FileContextProvider)
        return provider

    # ── load_and_assemble ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_load_and_assemble_basic(self, provider: FileContextProvider):
        msgs = await provider.load_and_assemble(
            current_message="Hello world",
            history=[],
        )
        assert isinstance(msgs, list)
        assert len(msgs) >= 2
        assert msgs[0]["role"] == "system"
        last = msgs[-1]
        assert last["role"] == "user"
        assert "Hello world" in last["content"]

    @pytest.mark.asyncio
    async def test_load_and_assemble_with_history(
        self, provider: FileContextProvider, mock_history_loader
    ):
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous response"},
        ]
        msgs = await provider.load_and_assemble(
            current_message="Follow up",
            history=history,
        )
        assert isinstance(msgs, list)
        assert len(msgs) >= 3

    @pytest.mark.asyncio
    async def test_load_and_assemble_with_matched_skills(
        self, provider: FileContextProvider
    ):
        msgs = await provider.load_and_assemble(
            current_message="Test query",
            history=[],
            matched_skills_context="## Relevant Skills\n- skill-A: does X",
        )
        content = msgs[0]["content"]
        assert "Relevant Skills" in content or "skill-A" in content

    @pytest.mark.asyncio
    async def test_load_and_assemble_with_mode(
        self, tmp_path: Path
    ):
        cfg = _make_ctx_cfg("test-mode", tmp_path)
        provider = create_context(cfg)
        msgs = await provider.load_and_assemble(
            current_message="Direct mode test",
            history=[],
            mode="direct",
        )
        assert isinstance(msgs, list)
        assert msgs[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_load_and_assemble_with_intent_shifted(
        self, tmp_path: Path
    ):
        cfg = _make_ctx_cfg("test-intent", tmp_path)
        provider = create_context(cfg)
        msgs = await provider.load_and_assemble(
            current_message="Intent shift test",
            history=[],
            intent_shifted=True,
        )
        assert isinstance(msgs, list)

    @pytest.mark.asyncio
    async def test_load_and_assemble_with_effective_context_window(
        self, tmp_path: Path
    ):
        cfg = _make_ctx_cfg("test-window", tmp_path)
        provider = create_context(cfg)
        msgs = await provider.load_and_assemble(
            current_message="Context window test",
            history=[],
            effective_context_window=100000,
        )
        assert isinstance(msgs, list)

    # ── append ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_append_messages(self, provider: FileContextProvider):
        initial = [{"role": "system", "content": "Test"}]
        new = [{"role": "user", "content": "New message"}]
        result = await provider.append(initial, new)
        assert len(result) == len(initial) + len(new)
        assert result[-1]["content"] == "New message"

    @pytest.mark.asyncio
    async def test_append_seq_injected(self, provider: FileContextProvider):
        initial = [{"role": "system", "content": "You are a helpful assistant."}]
        new = [{"role": "user", "content": "Test"}]
        result = await provider.append(initial, new)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Test"

    # ── prepare_for_api ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_prepare_for_api_basic(self, provider: FileContextProvider):
        msgs = await provider.load_and_assemble(
            current_message="Test for pipeline",
            history=[],
        )
        provider.init_budget(8192)
        result = await provider.prepare_for_api(msgs)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_prepare_for_api_with_state_messages(self, provider: FileContextProvider):
        msgs = await provider.load_and_assemble(
            current_message="Test",
            history=[],
        )
        provider.init_budget(8192)
        state_msgs = [{"role": "user", "content": "State info"}]
        result = await provider.prepare_for_api(msgs, state_messages=state_msgs)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_prepare_for_api_force_compact(self, provider: FileContextProvider):
        msgs = await provider.load_and_assemble(
            current_message="Force compact test",
            history=[],
        )
        provider.init_budget(4096)
        result = await provider.prepare_for_api(msgs, force_compact=True)
        assert isinstance(result, list)

    # ── assemble_system_prompt ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_assemble_system_prompt_basic(self, provider: FileContextProvider):
        prompt = await provider.assemble_system_prompt()
        assert isinstance(prompt, str)

    @pytest.mark.asyncio
    async def test_assemble_system_prompt_with_context(
        self, tmp_path: Path
    ):
        cfg = _make_ctx_cfg("test-prompt-ctx", tmp_path)
        provider = create_context(cfg)
        prompt = await provider.assemble_system_prompt(
            matched_skills_context="## Skills\n- coding: write code",
        )
        assert isinstance(prompt, str)

    @pytest.mark.asyncio
    async def test_assemble_system_prompt_direct_mode(
        self, tmp_path: Path
    ):
        cfg = _make_ctx_cfg("test-prompt-direct", tmp_path)
        provider = create_context(cfg)
        prompt = await provider.assemble_system_prompt(mode="direct")
        assert isinstance(prompt, str)

    # ── load_history ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_load_history_no_loader(self, provider: FileContextProvider):
        history = await provider.load_history()
        assert history == []

    @pytest.mark.asyncio
    async def test_load_history_with_loader(
        self, tmp_path: Path, mock_history_loader
    ):
        cfg = _make_ctx_cfg("test-history", tmp_path)
        provider = create_context(cfg)
        provider._history_loader = mock_history_loader
        history = await provider.load_history()
        assert isinstance(history, list)

    # ── build_history_summary ─────────────────────────────────────────

    def test_build_history_summary_empty(self, provider: FileContextProvider):
        summary = provider.build_history_summary(None)
        assert isinstance(summary, str)

    def test_build_history_summary_with_history(self, provider: FileContextProvider):
        history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
        ]
        summary = provider.build_history_summary(history)
        assert isinstance(summary, str)

    def test_build_history_summary_max_tokens(self, provider: FileContextProvider):
        history = [
            {"role": "user", "content": "A" * 500},
            {"role": "assistant", "content": "B" * 500},
        ]
        summary = provider.build_history_summary(history, max_tokens=100)
        assert isinstance(summary, str)

    # ── init_budget / sync_tokens / total_tokens ─────────────────────

    def test_init_budget(self, provider: FileContextProvider):
        provider.init_budget(8192)
        stats = provider.get_stats()
        assert "total_tokens" in stats

    def test_sync_tokens(self, provider: FileContextProvider):
        msgs = [
            {"role": "system", "content": "Test"},
            {"role": "user", "content": "Hello"},
        ]
        provider.sync_tokens(msgs)
        stats = provider.get_stats()
        assert "total_tokens" in stats

    def test_total_tokens_property(self, provider: FileContextProvider):
        msgs = [{"role": "system", "content": "Test message"}]
        provider.sync_tokens(msgs)
        assert provider.total_tokens >= 0

    # ── persist_tool_result ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_persist_tool_result_basic(self, provider: FileContextProvider):
        result = await provider.persist_tool_result(
            tool_call_id="call_123",
            tool_name="bash",
            result="Hello from tool",
        )
        assert isinstance(result, dict)
        assert "tool_call_id" in result

    @pytest.mark.asyncio
    async def test_persist_tool_result_no_persist_flag(self, provider: FileContextProvider):
        result = await provider.persist_tool_result(
            tool_call_id="call_456",
            tool_name="bash",
            result='{"_no_persist": true, "content": "skip this"}',
        )
        assert result.get("_persisted") is False

    # ── session_memory / context_memory ────────────────────────────────

    def test_session_memory_property(self, provider: FileContextProvider):
        sm = provider.session_memory
        assert sm is not None
        assert hasattr(sm, "get_content")

    def test_context_memory_property(self, provider: FileContextProvider):
        """L2 Long Term Memory 由 FileContextProvider 创建并注入 prompt。"""
        cm = provider.context_memory
        assert cm is not None
        assert hasattr(cm, "load_memory_prompt")

    # ── get_stats ─────────────────────────────────────────────────

    def test_get_stats_basic(self, provider: FileContextProvider):
        stats = provider.get_stats()
        assert isinstance(stats, dict)
        assert stats["provider"] == "file"
        assert "total_tokens" in stats
        assert "compact" in stats

    # ── force_compact_now ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_force_compact_now_basic(self, provider: FileContextProvider):
        old, new, preview = await provider.force_compact_now()
        assert isinstance(old, int)
        assert isinstance(new, int)
        assert isinstance(preview, str)
        assert old >= 0
        assert new >= 0
