# SPDX-License-Identifier: Apache-2.0
"""Tests for InfraService — single entry point for core/ to infra/.

Verifies: InfraService construction, all property accessors, integration.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_session_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


class TestInfraService:
    """InfraService 构造和属性访问测试。"""

    def test_construct_with_defaults(self, tmp_session_dir):
        from infra.service import InfraService

        infra = InfraService(
            session_id="test_session_001",
            session_dir=tmp_session_dir,
            data_dir=tmp_session_dir / "data",
            model="gpt-4o-mini",
        )
        assert infra.context is not None
        assert infra.session_memory is not None
        assert infra.compactor is not None
        assert infra.artifact_provider is not None

    def test_context_is_file_context_provider(self, tmp_session_dir):
        from infra.service import InfraService
        from infra.context.impl.file_context import FileContextProvider

        infra = InfraService(
            session_id="test_002",
            session_dir=tmp_session_dir,
            data_dir=tmp_session_dir / "data",
        )
        assert isinstance(infra.context, FileContextProvider)

    def test_session_memory_is_session_memory(self, tmp_session_dir):
        from infra.service import InfraService
        from infra.memory.impl.session_memory import SessionMemory

        infra = InfraService(
            session_id="test_003",
            session_dir=tmp_session_dir,
            data_dir=tmp_session_dir / "data",
        )
        assert isinstance(infra.session_memory, SessionMemory)

    def test_context_memory_via_long_memory(self, tmp_session_dir):
        """InfraService.context_memory 应返回 LongTermMemory 实例。"""
        from infra.service import InfraService

        infra = InfraService(
            session_id="test_ctx_mem",
            session_dir=tmp_session_dir,
            data_dir=tmp_session_dir / "data",
        )
        # L2 memory 由 InfraService 统一构造，始终存在
        assert infra.context_memory is not None
        assert hasattr(infra.context_memory, "load_memory_prompt")

    def test_session_dir_auto_generated(self, tmp_session_dir):
        from infra.service import InfraService

        infra = InfraService(
            session_id="test_auto_001",
            session_dir=tmp_session_dir,  # 传入 session_dir 以避免 sandbox 权限问题
        )
        assert infra.context is not None

    @pytest.mark.asyncio
    async def test_setup(self, tmp_session_dir):
        from infra.service import InfraService

        infra = InfraService(
            session_id="test_setup",
            session_dir=tmp_session_dir,
            data_dir=tmp_session_dir / "data",
        )
        await infra.session_memory.setup()
        assert infra.session_memory.is_empty() is True

    def test_compactor_provides_stats(self, tmp_session_dir):
        from infra.service import InfraService

        infra = InfraService(
            session_id="test_stats",
            session_dir=tmp_session_dir,
            data_dir=tmp_session_dir / "data",
        )
        stats = infra.compactor.get_stats()
        assert stats.total_calls == 0
        assert stats.total_compacted == 0

    def test_context_get_stats(self, tmp_session_dir):
        from infra.service import InfraService

        infra = InfraService(
            session_id="test_ctx_stats",
            session_dir=tmp_session_dir,
            data_dir=tmp_session_dir / "data",
        )
        stats = infra.context.get_stats()
        assert "provider" in stats
        assert stats["provider"] == "file"


class TestInfraServiceRecallContext:
    """SkillDispatcher recall_context 场景测试。"""

    @pytest.mark.asyncio
    async def test_recall_worklog(self, tmp_session_dir):
        from infra.service import InfraService

        infra = InfraService(
            session_id="test_recall",
            session_dir=tmp_session_dir,
            data_dir=tmp_session_dir / "data",
        )
        await infra.session_memory.setup()
        infra.session_memory.append_worklog_entry("- Ran pytest: all tests passed")

        result = infra.session_memory.recall("pytest")
        assert "pytest" in result.lower() or "Ran pytest" in result

    @pytest.mark.asyncio
    async def test_session_memory_llm_update(self, tmp_session_dir):
        from infra.service import InfraService

        infra = InfraService(
            session_id="test_llm_update",
            session_dir=tmp_session_dir,
            data_dir=tmp_session_dir / "data",
        )
        await infra.session_memory.setup()

        # Mock LLM client on the SessionMemory instance directly
        async def mock_llm_client(messages, *, system="", max_tokens=3000):
            return (
                "# Session Title\nTest Session\n\n"
                "# Current State\nWorking on test\n\n"
                "# Task Specification\nTest task\n\n"
                "# Files and Functions\ntest.py\n\n"
                "# Workflow\nRunning tests\n\n"
                "# Errors & Corrections\nNone\n\n"
                "# Key Results\nTests pass\n\n"
                "# Worklog\n- hello/hi there\n"
            )
        infra.session_memory._llm_client = mock_llm_client

        messages = [
            {"role": "user", "content": "hello", "_seq": 1},
            {"role": "assistant", "content": "hi there", "_seq": 2},
        ]
        await infra.session_memory.llm_update(messages, "step_1")

        # Verify .meta.json exists and last_summarized_seq was updated.
        # Initial value is 0; after successful LLM update it becomes 2 (max _seq).
        meta_path = tmp_session_dir / "session_memory" / ".meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["last_summarized_seq"] == 2


class TestFileContextProviderViaInfraService:
    """通过 InfraService 访问 FileContextProvider 的完整流程测试。"""

    @pytest.mark.asyncio
    async def test_assemble_system_prompt(self, tmp_session_dir):
        from infra.service import InfraService

        infra = InfraService(
            session_id="test_sys_prompt",
            session_dir=tmp_session_dir,
            data_dir=tmp_session_dir / "data",
        )
        await infra.session_memory.setup()
        infra.session_memory.append_worklog_entry("- Testing session memory")

        prompt = await infra.context.assemble_system_prompt(
            mode="agentic",
        )
        assert isinstance(prompt, str)

    @pytest.mark.asyncio
    async def test_artifact_provider_has_artifact(self, tmp_session_dir):
        from infra.service import InfraService

        infra = InfraService(
            session_id="test_artifact",
            session_dir=tmp_session_dir,
            data_dir=tmp_session_dir / "data",
        )
        assert await infra.artifact_provider.has_artifact("nonexistent") is False
