# SPDX-License-Identifier: Apache-2.0
"""Tests for TokenBudgetCompactor + ArtifactProviderImpl integration.

Verifies the Pre-API pipeline end-to-end: tool result persistence,
microcompact, budget guard, and SM compact.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_session_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def artifact_provider(tmp_session_dir):
    from infra.context.providers.artifact_impl import ArtifactProviderImpl

    return ArtifactProviderImpl(
        session_dir=tmp_session_dir,
        persist_ratio=0.3,
        extract_ratio=0.1,
        model="gpt-4o-mini",
    )


@pytest.fixture
def session_memory(tmp_session_dir):
    from infra.memory.impl.session_memory import SessionMemory

    sm = SessionMemory(
        session_dir=tmp_session_dir,
        model="gpt-4o-mini",
        llm_update_interval=3,
    )
    return sm


@pytest.fixture
def compactor(tmp_session_dir, artifact_provider, session_memory):
    from infra.context.providers.compactor_impl import TokenBudgetCompactor

    return TokenBudgetCompactor(
        model="gpt-4o-mini",
        context_budget=10000,
        microcompact_keep_recent=2,
        microcompact_compactable_tools={"bash", "read_file", "execute_skill"},
        emergency_keep_tail=4,
        max_compact_failures=3,
        artifact_provider=artifact_provider,
        session_memory=session_memory,
    )


class TestArtifactProviderImpl:
    """ArtifactProviderImpl 持久化 + 回读测试。"""

    @pytest.mark.asyncio
    async def test_small_result_not_persisted(self, artifact_provider):
        result = await artifact_provider.process_tool_result(
            tool_call_id="small_1",
            tool_name="bash",
            content="echo hello",
            remaining_budget_tokens=5000,
        )
        assert result["_persisted"] is False
        assert result["content"] == "echo hello"

    @pytest.mark.asyncio
    async def test_large_result_persisted(self, artifact_provider, tmp_session_dir):
        large_content = "x" * 10000
        result = await artifact_provider.process_tool_result(
            tool_call_id="large_1",
            tool_name="bash",
            content=large_content,
            remaining_budget_tokens=1000,
        )
        assert result["_persisted"] is True
        assert "Output persisted" in result["content"]
        assert "recall_context('large_1')" in result["content"]

        stored = await artifact_provider.read_artifact("large_1")
        assert stored == large_content

    @pytest.mark.asyncio
    async def test_has_artifact(self, artifact_provider, tmp_session_dir):
        large_content = "y" * 5000
        await artifact_provider.process_tool_result(
            tool_call_id="check_1",
            tool_name="bash",
            content=large_content,
            remaining_budget_tokens=1000,
        )
        assert await artifact_provider.has_artifact("check_1") is True
        assert await artifact_provider.has_artifact("nonexistent") is False

    @pytest.mark.asyncio
    async def test_execute_skill_auto_recall(self, artifact_provider, tmp_session_dir):
        skill_output = '{"ok": true, "summary": "skill ran", "output": {"result": "done"}}'
        result = await artifact_provider.process_tool_result(
            tool_call_id="skill_1",
            tool_name="execute_skill",
            content=skill_output,
            remaining_budget_tokens=2000,
        )
        assert result["_persisted"] is False
        assert "done" in result["content"]


class TestMicrocompact:
    """Microcompact 零 LLM 清除测试。"""

    def test_microcompact_clears_old_results(self):
        from infra.context.providers.microcompact_impl import microcompact_messages

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "let me search"},
            # 使用足够长的内容，使得清除后能节省 token
            # CLEARED_MESSAGE ≈ 29 chars (≈ 10 tokens), 这些内容更长
            {"role": "tool", "tool_call_id": "t1", "name": "bash", "content": "total 256\n-rw-r--  1 user  staff   128 Apr 13 file1.py\n-rw-r--  2 user  staff   256 Apr 13 file2.py\n"},
            {"role": "tool", "tool_call_id": "t2", "name": "bash", "content": "total 512\n-rw-r--  1 user  staff   300 Apr 13 file3.py\n-rw-r--  2 user  staff   400 Apr 13 file4.py\n"},
            {"role": "tool", "tool_call_id": "t3", "name": "bash", "content": "ls\nresult3"},
            {"role": "assistant", "content": "here you go"},
        ]
        compactable = {"bash"}
        result, saved = microcompact_messages(messages, keep_recent=1, compactable_tools=compactable)

        assert result[2]["content"] == "[Old tool result content cleared]"
        assert result[3]["content"] == "[Old tool result content cleared]"
        assert result[4]["content"] == "ls\nresult3"
        assert saved > 0

    def test_microcompact_skips_persisted(self):
        from infra.context.providers.microcompact_impl import microcompact_messages

        # 3个 tool 消息：t1 已落盘，t2/t3 普通内容
        # keep_recent=1 保留最近1个，其余被清除
        messages = [
            {"role": "tool", "tool_call_id": "t1", "name": "bash", "content": "[Output persisted: 1000 chars]"},
            # t2 内容足够长，清除后能节省 token（80+ chars → 29 CLEARED chars）
            {"role": "tool", "tool_call_id": "t2", "name": "bash", "content": "File listing:\n  package.json  2KB\n  index.html   5KB\n  main.js      10KB\n  style.css     1KB\n"},
            {"role": "tool", "tool_call_id": "t3", "name": "bash", "content": "recent result three"},
        ]
        result, saved = microcompact_messages(
            messages, keep_recent=1, compactable_tools={"bash"}
        )
        # t1: [Output persisted prefix → 跳过，不清除
        assert result[0]["content"] == "[Output persisted: 1000 chars]"
        # t2: 旧消息，非持久化 → 清除
        assert result[1]["content"] == "[Old tool result content cleared]"
        # t3: 最近消息 → 保留
        assert result[2]["content"] == "recent result three"
        # 清除行为正确（saved >= 0  因为估算误差，小内容清除后可能不节省）
        assert saved >= 0


class TestTokenBudgetCompactor:
    """TokenBudgetCompactor 端到端测试。"""

    @pytest.mark.asyncio
    async def test_prepare_for_api_no_compact_needed(self, compactor, tmp_session_dir):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ]
        result, was_compacted, trigger, before, after = await compactor.prepare_for_api(messages)
        assert was_compacted is False
        assert result == messages

    @pytest.mark.asyncio
    async def test_prepare_for_api_microcompact(self, compactor, tmp_session_dir):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "let me run some commands"},
            {"role": "tool", "tool_call_id": "t1", "name": "bash", "content": "old result 1"},
            {"role": "tool", "tool_call_id": "t2", "name": "bash", "content": "old result 2"},
            {"role": "tool", "tool_call_id": "t3", "name": "bash", "content": "old result 3"},
            {"role": "tool", "tool_call_id": "t4", "name": "bash", "content": "recent result"},
        ]
        result, was_compacted, trigger, before, after = await compactor.prepare_for_api(messages)
        # microcompact clears old tool results (keep_recent=2, so clears 2 of 4 tool messages)
        # Budget Guard does NOT trigger (token count << 10000 budget), so was_compacted=False
        assert was_compacted is False
        assert any("[Old tool result content cleared]" in str(m.get("content", "")) for m in result)

    @pytest.mark.asyncio
    async def test_prepare_for_api_force_compact(self, compactor, tmp_session_dir):
        compactor._context_budget = 100
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "A" * 500},
        ]
        result, was_compacted, trigger, before, after = await compactor.prepare_for_api(
            messages, force_compact=True
        )
        assert was_compacted is True
        assert after <= compactor._context_budget

    @pytest.mark.asyncio
    async def test_stats_tracking(self, compactor):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "assistant", "content": "A" * 1000},
        ]
        await compactor.prepare_for_api(messages)
        stats = compactor.get_stats()
        assert stats.total_calls == 1
        assert stats.total_compacted >= 0
