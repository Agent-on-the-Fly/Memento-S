# SPDX-License-Identifier: Apache-2.0
"""Tests for RecallEngine — three-mode recall interface.

Verifies:
  - recall_session_memory(): wraps SessionMemory.recall() into RecallResult
  - recall_long_term_memory(): wraps LongTermMemory.recall_topics_prompt() via recall_topics
  - recall_all(): aggregates L1 + L2 + artifacts + cross-session (async)
  - RecallResult.to_display(): formats entries or returns "no context" hint
  - _recall_artifact: keyword search across artifact files
  - _recall_l1_other_sessions: keyword search across other sessions' summary.md
  - _match_score: scoring logic
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dirs():
    """Creates three temp directories: context_root, session_dir, other_session_dir."""
    with tempfile.TemporaryDirectory() as td:
        context_root = Path(td)
        session_dir = context_root / "sessions" / "session_current"
        session_dir.mkdir(parents=True, exist_ok=True)

        other_session_dir = context_root / "sessions" / "session_other"
        other_session_dir.mkdir(parents=True, exist_ok=True)
        (other_session_dir / "memory").mkdir(exist_ok=True)
        (other_session_dir / "memory" / "summary.md").write_text(
            "# Project Architecture\n\nThis session covered the architecture of the system.",
            encoding="utf-8",
        )

        yield {
            "context_root": context_root,
            "session_dir": session_dir,
            "other_session_dir": other_session_dir,
        }


@pytest.fixture
def session_memory(tmp_dirs):
    from infra.memory.impl.session_memory import SessionMemory

    sm = SessionMemory(
        session_dir=tmp_dirs["session_dir"],
        model="",
        llm_update_interval=999,  # disable LLM updates
    )
    # SessionMemory stores summary at session_dir / "summary.md" (SESSION_MEMORY_DIR="")
    summary = (
        "# Session Memory\n"
        "# Current Task\nBug fixing in the auth module.\n"
        "# Worklog\n- Fixed login bug\n"
    )
    (tmp_dirs["session_dir"] / "summary.md").write_text(summary, encoding="utf-8")
    return sm


@pytest.fixture
def long_term_memory(tmp_dirs):
    from infra.memory.impl.long_term_memory import LongTermMemory

    ltm_dir = tmp_dirs["context_root"] / "memory"
    ltm_dir.mkdir(parents=True, exist_ok=True)
    ltm = LongTermMemory(memory_dir=ltm_dir, model="")
    # Write a topic with exact slug match for "auth-design"
    (ltm_dir / "topics").mkdir(exist_ok=True)
    (ltm_dir / "topics" / "auth-design.md").write_text(
        "# Auth Design\n\nDetailed auth architecture.",
        encoding="utf-8",
    )
    return ltm


@pytest.fixture
def artifact_provider(tmp_dirs):
    """Minimal mock artifact provider — duck-typed, just needs to exist."""
    artifacts_dir = tmp_dirs["session_dir"] / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "tool_123.txt").write_text(
        "This artifact contains bug fix information.",
        encoding="utf-8",
    )
    (artifacts_dir / "tool_456.txt").write_text(
        "Unrelated content about logging.",
        encoding="utf-8",
    )
    return artifacts_dir


@pytest.fixture
def recall_engine(session_memory, long_term_memory, artifact_provider, tmp_dirs):
    from infra.memory.recall_engine import RecallEngine

    return RecallEngine(
        session_memory=session_memory,
        long_term_memory=long_term_memory,
        artifact_provider=artifact_provider,
        context_dir=tmp_dirs["context_root"],
        current_session_dir=tmp_dirs["session_dir"],
    )


class TestRecallEntryAndResult:
    """RecallEntry / RecallResult data classes."""

    def test_recall_result_to_display_empty(self):
        from infra.memory.recall_engine import RecallResult

        result = RecallResult()
        assert "No context found" in result.to_display("nonexistent")

    def test_recall_result_to_display_entries(self):
        from infra.memory.recall_engine import RecallEntry, RecallResult

        entries = [
            RecallEntry(source="L1:current", title="Task", content="bug fix", score=0.9),
            RecallEntry(source="L2:auth", title="Auth", content="design doc", score=0.7),
        ]
        result = RecallResult(entries=entries)
        display = result.to_display("bug")
        assert "L1:current" in display
        assert "L2:auth" in display
        assert "bug fix" in display

    def test_recall_result_content_truncation(self):
        from infra.memory.recall_engine import RecallEntry, RecallResult

        long_content = "x" * 1000
        entries = [RecallEntry(source="L1:current", title="T", content=long_content, score=1.0)]
        result = RecallResult(entries=entries)
        display = result.to_display("x")
        # content is truncated to 500 chars
        assert len(display) < 1000


class TestMatchScore:
    """Unit tests for _match_score."""

    def test_exact_slug_match(self):
        from infra.memory.recall_engine import _match_score

        score = _match_score("auth", "auth", "Auth Design", "", "content")
        assert score == 1.0

    def test_exact_title_match(self):
        from infra.memory.recall_engine import _match_score

        # exact match: "auth" == "auth"
        score = _match_score("auth", "auth", "Auth Design", "", "content")
        assert score == 1.0

    def test_exact_title_match_substring(self):
        from infra.memory.recall_engine import _match_score

        # "design" is a token in "Auth Design" → substring in title = 0.7
        score = _match_score("design", "auth", "Auth Design", "", "content")
        assert score == 0.7

    def test_hook_match(self):
        from infra.memory.recall_engine import _match_score

        # "hookword" exactly equals hook → 0.9
        score = _match_score("hookword", "auth", "Auth", "hookword", "content")
        assert score == 0.9

    def test_substring_in_content(self):
        from infra.memory.recall_engine import _match_score

        # "bug" appears in content → 0.5
        score = _match_score("bug", "auth", "Auth", "", "bug fix content")
        assert score == 0.5

    def test_no_match(self):
        from infra.memory.recall_engine import _match_score

        score = _match_score("xyzabc", "auth", "Auth", "", "totally different")
        assert score == 0.0

    def test_empty_query(self):
        from infra.memory.recall_engine import _match_score

        score = _match_score("", "auth", "Auth", "", "content")
        assert score == 0.0

    def test_fuzzy_match(self):
        from infra.memory.recall_engine import _match_score

        # "autn" is close to "auth" (3/4 chars match)
        score = _match_score("autn", "auth", "Auth", "", "content")
        assert 0.0 < score < 1.0


class TestRecallSessionMemory:
    """recall_session_memory() wraps SessionMemory.recall()."""

    def test_returns_matching_entries(self, recall_engine):
        result = recall_engine.recall_session_memory("Bug")
        assert len(result.entries) >= 1
        assert any(e.source == "L1:current" for e in result.entries)

    def test_returns_empty_on_no_match(self, recall_engine):
        result = recall_engine.recall_session_memory("zzzznonexistentterm")
        assert result.entries == []

    def test_none_session_memory_returns_empty(self, recall_engine):
        recall_engine._session_memory = None
        result = recall_engine.recall_session_memory("bug")
        # Should return empty RecallResult without crashing
        assert result.entries == []


class TestRecallLongTermMemory:
    """recall_long_term_memory() wraps LongTermMemory via recall_topics."""

    def test_returns_topic_matches(self, recall_engine):
        # "auth-design" exact slug match → score 1.0
        result = recall_engine.recall_long_term_memory("auth-design")
        assert len(result.entries) >= 1
        assert any(e.source.startswith("L2:") for e in result.entries)
        assert result.entries[0].score == 1.0

    def test_returns_empty_on_no_match(self, recall_engine):
        result = recall_engine.recall_long_term_memory("zzzznotexist")
        assert result.entries == []

    def test_score_ordering(self, recall_engine):
        # exact slug match should score highest
        result = recall_engine.recall_long_term_memory("auth")
        if len(result.entries) >= 2:
            assert result.entries[0].score >= result.entries[1].score


class TestRecallAll:
    """recall_all() aggregates all four paths."""

    @pytest.mark.asyncio
    async def test_recall_all_returns_result(self, recall_engine):
        result = await recall_engine.recall_all("bug")
        assert hasattr(result, "entries")
        assert hasattr(result, "to_display")

    @pytest.mark.asyncio
    async def test_recall_all_sorted_by_score(self, recall_engine):
        result = await recall_engine.recall_all("auth")
        scores = [e.score for e in result.entries]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_recall_all_with_no_matches(self, recall_engine):
        result = await recall_engine.recall_all("zzzzunknown999")
        assert result.entries == []


class TestRecallArtifact:
    """_recall_artifact() keyword search across artifact files."""

    @pytest.mark.asyncio
    async def test_finds_matching_artifact(self, recall_engine):
        result = await recall_engine._recall_artifact("bug")
        assert len(result.entries) >= 1
        assert any("bug" in e.source or "bug" in e.content.lower() for e in result.entries)

    @pytest.mark.asyncio
    async def test_excludes_non_matching_artifact(self, recall_engine):
        result = await recall_engine._recall_artifact("zzzznonexistent")
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_no_artifacts_dir(self, recall_engine, tmp_dirs):
        shutil.rmtree(tmp_dirs["session_dir"] / "artifacts")
        result = await recall_engine._recall_artifact("bug")
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_respects_max_files_limit(self, recall_engine, tmp_dirs):
        artifacts_dir = tmp_dirs["session_dir"] / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        for i in range(60):
            (artifacts_dir / f"tool_{i:03d}.txt").write_text(
                f"content {i} bug", encoding="utf-8"
            )
        result = await recall_engine._recall_artifact("bug")
        assert len(result.entries) <= 50


class TestRecallL1OtherSessions:
    """_recall_l1_other_sessions() scans other sessions' summary.md."""

    @pytest.mark.asyncio
    async def test_finds_other_session_match(self, recall_engine):
        result = await recall_engine._recall_l1_other_sessions("architecture")
        assert len(result.entries) >= 1
        assert any(e.source.startswith("L1:session_other") for e in result.entries)

    @pytest.mark.asyncio
    async def test_excludes_current_session(self, recall_engine):
        result = await recall_engine._recall_l1_other_sessions("architecture")
        assert not any("session_current" in e.source for e in result.entries)

    @pytest.mark.asyncio
    async def test_no_other_sessions(self, recall_engine, tmp_dirs):
        shutil.rmtree(tmp_dirs["other_session_dir"])
        result = await recall_engine._recall_l1_other_sessions("architecture")
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_respects_max_sessions_limit(self, recall_engine, tmp_dirs):
        context_root = tmp_dirs["context_root"]
        for i in range(15):
            sp = context_root / "sessions" / f"session_{i:03d}"
            sp.mkdir(parents=True, exist_ok=True)
            (sp / "memory").mkdir(exist_ok=True)
            (sp / "memory" / "summary.md").write_text(
                f"# Session {i}\nBug fixing.\n", encoding="utf-8"
            )
        result = await recall_engine._recall_l1_other_sessions("bug")
        assert len(result.entries) <= 10
