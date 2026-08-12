# SPDX-License-Identifier: Apache-2.0
"""Tests for SessionMemory lifecycle.

Verifies: setup, worklog append, LLM update, SM compact, recall.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_session_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def session_memory(tmp_session_dir):
    from infra.memory.impl.session_memory import SessionMemory

    sm = SessionMemory(
        session_dir=tmp_session_dir,
        model="gpt-4o-mini",
        llm_update_interval=3,
    )
    return sm


class TestSessionMemory:
    """SessionMemory 生命周期测试。"""

    @pytest.mark.asyncio
    async def test_setup_creates_directory(self, session_memory, tmp_session_dir):
        await session_memory.setup()
        assert (tmp_session_dir / "session_memory").exists()
        assert (tmp_session_dir / "session_memory" / "summary.md").exists()

    @pytest.mark.asyncio
    async def test_is_empty_before_content(self, session_memory):
        await session_memory.setup()
        assert session_memory.is_empty() is True

    def test_append_worklog_entry(self, session_memory, tmp_session_dir):
        session_memory._dir = tmp_session_dir / "session_memory"
        session_memory._dir.mkdir(parents=True, exist_ok=True)
        session_memory._template = "# Session Title\n_Title_\n\n# Current State\n_State_\n\n# Worklog\n_Worklog_\n"
        session_memory._path = tmp_session_dir / "session_memory" / "summary.md"
        session_memory._path.write_text(session_memory._template, encoding="utf-8")

        session_memory.append_worklog_entry("- 2026-04-13: Did something")
        content = session_memory.get_content()
        assert "Did something" in content

    def test_recall_query_match(self, session_memory, tmp_session_dir):
        session_memory._dir = tmp_session_dir / "session_memory"
        session_memory._dir.mkdir(parents=True, exist_ok=True)
        session_memory._template = "# Session Title\nTest Session\n\n# Worklog\n_Worklog_\n"
        session_memory._path = tmp_session_dir / "session_memory" / "summary.md"
        session_memory._path.write_text(
            "# Session Title\nTest Session\n\n# Worklog\n- did pytest\n",
            encoding="utf-8",
        )

        result = session_memory.recall("pytest")
        assert "pytest" in result.lower()

    def test_recall_no_match(self, session_memory, tmp_session_dir):
        session_memory._dir = tmp_session_dir / "session_memory"
        session_memory._dir.mkdir(parents=True, exist_ok=True)
        session_memory._template = "# Session Title\nTest\n\n# Worklog\n_Worklog_\n"
        session_memory._path = tmp_session_dir / "session_memory" / "summary.md"
        session_memory._path.write_text("# Session Title\nTest\n\n# Worklog\n- test\n", encoding="utf-8")

        result = session_memory.recall("nonexistent_term_xyz")
        assert "No matches found" in result

    def test_truncate_for_compact(self, session_memory, tmp_session_dir):
        session_memory._dir = tmp_session_dir / "session_memory"
        session_memory._dir.mkdir(parents=True, exist_ok=True)
        # 需要 > max_chars (≈8000) 才触发截断
        sections = []
        for i in range(300):
            sections.append(f"# Section {i}\n{'Content line ' + str(i) + ' ' * 50}\n")
        large_content = "# Session Title\nTest\n\n" + "\n".join(sections)
        session_memory._template = "# Session Title\nTemplate\n"
        session_memory._path = tmp_session_dir / "session_memory" / "summary.md"
        session_memory._path.write_text(large_content, encoding="utf-8")
        session_memory._cached_content = None  # 清除缓存，确保读取文件内容

        truncated, was_truncated = session_memory.truncate_for_compact()
        assert was_truncated is True
        assert "[... session memory truncated" in truncated
        assert len(truncated) < len(large_content)

    def test_to_prompt_section(self, session_memory, tmp_session_dir):
        session_memory._dir = tmp_session_dir / "session_memory"
        session_memory._dir.mkdir(parents=True, exist_ok=True)
        session_memory._template = "# Session Title\n_Title_\n"
        session_memory._path = tmp_session_dir / "session_memory" / "summary.md"
        session_memory._path.write_text("# Session Title\nTest\n\n# Worklog\n- test\n", encoding="utf-8")

        section = session_memory.to_prompt_section()
        assert "## Session Memory" in section
        assert "Test" in section

    def test_should_llm_update(self, session_memory):
        session_memory._react_since_llm_update = 2
        session_memory._llm_update_interval = 5
        assert session_memory.should_llm_update() is False

        session_memory._react_since_llm_update = 5
        assert session_memory.should_llm_update() is True

    def test_last_summarized_seq(self, session_memory):
        assert session_memory.last_summarized_seq == 0

    @pytest.mark.asyncio
    async def test_llm_update_calls_client(
        self, session_memory, tmp_session_dir
    ):
        session_memory._dir = tmp_session_dir / "session_memory"
        session_memory._dir.mkdir(parents=True, exist_ok=True)
        session_memory._template = (
            "# Session Title\n_Title_\n\n"
            "# Current State\n_State_\n\n"
            "# Task Specification\n_Task_\n\n"
            "# Files and Functions\n_Files_\n\n"
            "# Workflow\n_Workflow_\n\n"
            "# Errors & Corrections\n_Errors_\n\n"
            "# Key Results\n_Results_\n\n"
            "# Worklog\n_Worklog_\n"
        )
        session_memory._path = tmp_session_dir / "session_memory" / "summary.md"
        session_memory._path.write_text(session_memory._template, encoding="utf-8")

        # Mock LLM client to return valid summary content
        async def mock_llm_client(messages, *, system="", max_tokens=3000):
            return (
                "# Session Title\nTest Session\n\n"
                "# Current State\nWorking on test\n\n"
                "# Task Specification\nTest task\n\n"
                "# Files and Functions\ntest.py\n\n"
                "# Workflow\nRunning tests\n\n"
                "# Errors & Corrections\nNone\n\n"
                "# Key Results\nTests pass\n\n"
                "# Worklog\n- hello/hi\n"
            )
        session_memory._llm_client = mock_llm_client

        messages = [
            {"role": "user", "content": "hello", "_seq": 1},
            {"role": "assistant", "content": "hi", "_seq": 2},
        ]
        await session_memory.llm_update(messages, "step_1")

        # Verify .meta.json was updated by successful LLM update.
        # Initial last_summarized_seq is 0; after successful LLM update it becomes 2 (max _seq).
        meta_path = tmp_session_dir / "session_memory" / ".meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["last_summarized_seq"] == 2
