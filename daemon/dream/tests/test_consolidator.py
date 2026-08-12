# SPDX-License-Identifier: Apache-2.0
"""Tests for daemon/dream/consolidator.py — DreamConsolidator.

Verifies: lock acquisition, staging gate, deep_run delegation.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_memory_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def ltm(tmp_memory_dir):
    from infra.memory.impl.long_term_memory import LongTermMemory

    return LongTermMemory(memory_dir=tmp_memory_dir, model="")


@pytest.fixture
def engine(tmp_memory_dir, ltm):
    from infra.memory.consolidation import MemoryConsolidationEngine
    from middleware.config.schemas.config_models import MemoryConsolidationConfig

    config = MemoryConsolidationConfig(
        enabled=True,
        min_staging_sessions=2,  # 匹配测试中 2 个 session
        min_staging_bytes=100,
        max_tokens_per_call=2000,
        poll_interval_seconds=60.0,
    )
    return MemoryConsolidationEngine(
        memory=ltm,
        config=config,
    )


@pytest.fixture
def consolidator(ltm, engine):
    from daemon.dream.consolidator import DreamConsolidator
    from middleware.config.schemas.config_models import DreamConfig

    config = DreamConfig(
        enabled=True,
        min_hours=1,
        poll_interval_seconds=600.0,
        scan_interval_seconds=600.0,
    )
    return DreamConsolidator(
        memory=ltm,
        engine=engine,
        config=config,
    )


class TestLockAcquisition:
    """Test _acquire_lock / _release_lock."""

    def test_acquires_lock(self, consolidator):
        assert consolidator._acquire_lock() is True

    def test_blocks_second_acquire_before_timeout(self, consolidator):
        consolidator._acquire_lock()
        assert consolidator._acquire_lock() is False


class TestStagingGate:
    """Test _staging_gate_passed."""

    def test_fails_empty_staging(self, consolidator):
        assert consolidator._staging_gate_passed() is False

    def test_fails_insufficient_sessions(self, ltm, engine):
        from daemon.dream.consolidator import DreamConsolidator
        from middleware.config.schemas.config_models import DreamConfig

        ltm._staging_path.write_text(
            "## Session 2025-01-01\ncontent\n",
            encoding="utf-8",
        )
        # DreamConsolidator now delegates to engine.check_should_consolidate(),
        # which uses MemoryConsolidationConfig.min_staging_sessions (default 5).
        # With only 1 session marker the gate should fail.
        config = DreamConfig(min_hours=1, poll_interval_seconds=600.0, scan_interval_seconds=600.0)
        c = DreamConsolidator(memory=ltm, engine=engine, config=config)
        assert c._staging_gate_passed() is False

    def test_passes_with_enough_sessions(self, ltm, engine):
        from daemon.dream.consolidator import DreamConsolidator
        from middleware.config.schemas.config_models import DreamConfig

        ltm._staging_path.write_text(
            "## Session 2025-01-01\nc1\n---\n## Session 2025-01-02\nc2\n",
            encoding="utf-8",
        )
        config = DreamConfig(min_hours=1, poll_interval_seconds=600.0, scan_interval_seconds=600.0)
        c = DreamConsolidator(memory=ltm, engine=engine, config=config)
        assert c._staging_gate_passed() is True


class TestMaybeTrigger:
    """Test maybe_trigger gate ordering and engine delegation."""

    @pytest.mark.asyncio
    async def test_skips_without_staging(self, consolidator):
        await consolidator.maybe_trigger()

    @pytest.mark.asyncio
    async def test_delegates_to_engine_deep_run(
        self, consolidator, ltm, monkeypatch
    ):
        ltm._staging_path.write_text(
            "## Session 2025-01-01\nc1\n---\n## Session 2025-01-02\nc2\n",
            encoding="utf-8",
        )

        llm_response = (
            '{"updated_topics":[],"new_topics":[{'
            '"slug":"test-topic","title":"Test","content":"Test content"}],'
            '"deleted_topics":[],"index_content":"# Memory Index\\n"}'
        )

        async def fake_chat(*args, **kwargs):
            return llm_response

        monkeypatch.setattr(
            "middleware.llm.llm_client.chat_completions_async", fake_chat
        )

        await consolidator.maybe_trigger()

        assert ltm.read_topic("test-topic") == "# Test\n\nTest content"
        assert ltm.get_staging_content().strip() == ""
