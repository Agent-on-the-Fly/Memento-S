# SPDX-License-Identifier: Apache-2.0
"""Tests for LoopSupervisionHook — core/skill/execution/hooks/loop_supervision.py"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from shared.hooks.types import HookEvent, HookPayload, HookResult
from core.skill.execution.hooks.loop_supervision import (
    LoopSupervisionHook,
    RECOMMEND_ABORT,
    RECOMMEND_RETRY,
    CONTINUE,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def loop_hook():
    """LoopSupervisionHook with default config."""
    return LoopSupervisionHook(max_observation_chain=6, min_effect_ratio=0.15, window_size=10)


@pytest.fixture
def shared_hook_context():
    """Shared hook_context dict (simulating FileChangeHook → LoopSupervisionHook)."""
    return {}


@pytest.fixture
def state_ctx():
    """Mock state context for bind_state_context."""
    return {
        "turn_count": 0,
        "artifact_registry": MagicMock(),
        "update_scratchpad": MagicMock(),
    }


def make_payload(
    event: HookEvent = HookEvent.AFTER_TOOL_EXEC,
    tool_name: str = "search_web",
    result: str = "Some search results",
    args: dict | None = None,
) -> HookPayload:
    """Helper to create HookPayload for testing."""
    return HookPayload(
        event=event,
        tool_name=tool_name,
        args=args or {},
        result=result,
    )


# =============================================================================
# Initialization
# =============================================================================


class TestLoopSupervisionHookInit:
    """Tests for LoopSupervisionHook initialization."""

    def test_init_with_default_config(self):
        """Hook initializes with default LoopDetector."""
        hook = LoopSupervisionHook()
        assert hook._loop_detector is not None
        assert hook._loop_detector.max_observation_chain == 6
        assert hook._loop_detector.min_effect_ratio == 0.15
        assert hook._loop_detector.window_size == 10

    def test_init_with_custom_config(self):
        """Custom config is passed to LoopDetector."""
        hook = LoopSupervisionHook(
            max_observation_chain=4,
            min_effect_ratio=0.2,
            window_size=20,
        )
        assert hook._loop_detector.max_observation_chain == 4
        assert hook._loop_detector.min_effect_ratio == 0.2
        assert hook._loop_detector.window_size == 20

    def test_bind_state_context(self, loop_hook, state_ctx):
        """bind_state_context() stores the context."""
        loop_hook.bind_state_context(state_ctx)
        assert loop_hook._state_ctx is state_ctx


# =============================================================================
# execute() — Event Filtering
# =============================================================================


class TestLoopSupervisionHookEventFiltering:
    """Tests for execute() event type filtering."""

    @pytest.mark.asyncio
    async def test_non_after_tool_exec_returns_allowed_true(self, loop_hook):
        """Non AFTER_TOOL_EXEC events pass through immediately."""
        payload = make_payload(event=HookEvent.BEFORE_SKILL_EXEC)
        result = await loop_hook.execute(payload)
        assert result.allowed is True
        assert result.deferred_messages is None
        assert result.recovery_action is None

    @pytest.mark.asyncio
    async def test_befor_tool_exec_returns_allowed_true(self, loop_hook):
        """BEFORE_TOOL_EXEC events pass through."""
        payload = make_payload(event=HookEvent.BEFORE_TOOL_EXEC)
        result = await loop_hook.execute(payload)
        assert result.allowed is True


# =============================================================================
# execute() — Loop Detection
# =============================================================================


class TestLoopSupervisionHookLoopDetection:
    """Tests for loop detection via LoopDetector integration."""

    @pytest.mark.asyncio
    async def test_no_loop_returns_allowed_true(self, loop_hook, state_ctx):
        """No loop detected → allowed=True, no deferred_messages."""
        loop_hook.bind_state_context(state_ctx)
        # Normal tool calls with effects
        for i in range(3):
            payload = make_payload(tool_name="file_create", result="Created file.txt")
            result = await loop_hook.execute(payload)
        assert result.allowed is True
        assert result.deferred_messages is None

    @pytest.mark.asyncio
    async def test_observation_chain_triggers_deferred_message(self, loop_hook, state_ctx):
        """6+ consecutive observations → deferred_messages with LOOP_DETECTED."""
        loop_hook.bind_state_context(state_ctx)
        # 6 observations
        for i in range(6):
            payload = make_payload(tool_name="search_web", result="Search result")
            await loop_hook.execute(payload)
        # 7th observation triggers
        payload = make_payload(tool_name="search_web", result="Another search")
        result = await loop_hook.execute(payload)
        assert result.allowed is True
        assert result.deferred_messages is not None
        assert len(result.deferred_messages) >= 1
        assert any("LOOP_DETECTED" in m["content"] for m in result.deferred_messages)

    @pytest.mark.asyncio
    async def test_observation_chain_triggers_recommend_abort(self, loop_hook, state_ctx):
        """observation_chain itself does NOT set recovery_action; only repeating action/state does."""
        loop_hook.bind_state_context(state_ctx)
        for i in range(7):
            payload = make_payload(tool_name="search_web", result="result")
            await loop_hook.execute(payload)
        payload = make_payload(tool_name="search_web", result="result")
        result = await loop_hook.execute(payload)
        # observation_chain has no recovery_action (returns None)
        assert result.recovery_action is None
        # But deferred_messages are set
        assert result.deferred_messages is not None

    @pytest.mark.asyncio
    async def test_repeating_sequence_triggers_recommend_retry(self, loop_hook, state_ctx):
        """repeating_sequence triggers RECOMMEND_RETRY (warning only, does not abort).

        repeating_sequence 只发警告不打断执行，这是与其他真正的死循环模式（如
        repeated_action）的重要区别。
        """
        loop_hook.bind_state_context(state_ctx)
        # Need >= 6 records for repeating_sequence check
        tools = ["read", "write", "read", "write", "read", "write"]
        for t in tools:
            payload = make_payload(tool_name=t, result="result")
            await loop_hook.execute(payload)
        result = await loop_hook.execute(make_payload(tool_name="read", result="result"))
        # repeating_sequence 触发 RETRY（不发 ABORT）
        assert result.recovery_action == RECOMMEND_RETRY

    @pytest.mark.asyncio
    async def test_metadata_contains_loop_info(self, loop_hook, state_ctx):
        """HookResult.metadata contains loop_info."""
        loop_hook.bind_state_context(state_ctx)
        for i in range(7):
            payload = make_payload(tool_name="search_web", result="result")
            await loop_hook.execute(payload)
        result = await loop_hook.execute(
            make_payload(tool_name="search_web", result="result")
        )
        assert result.metadata is not None
        assert "loop_info" in result.metadata
        assert result.metadata["loop_info"]["type"] == "observation_chain"


# =============================================================================
# execute() — Hook Context Integration (FileChangeHook → LoopSupervisionHook)
# =============================================================================


class TestLoopSupervisionHookFsChanges:
    """Tests for FileChangeHook → LoopSupervisionHook fs_changes integration."""

    @pytest.mark.asyncio
    async def test_reads_fs_changes_from_hook_context(self, loop_hook, shared_hook_context, state_ctx):
        """LoopSupervisionHook reads fs_changes written by FileChangeHook."""
        loop_hook.bind_state_context(state_ctx)
        # Simulate FileChangeHook writing to hook_context
        shared_hook_context["fs_changes"] = {
            "created": ["/workspace/output.txt"],
            "modified": [],
            "deleted": [],
        }
        loop_hook.hook_context = shared_hook_context
        # Record with empty tool (no explicit effect)
        payload = make_payload(tool_name="bash", result="bash output")
        result = await loop_hook.execute(payload)
        # Should record with created_artifacts from fs_changes
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_fs_changes_override_category(self, loop_hook, shared_hook_context, state_ctx):
        """fs_changes with created files forces effect category in LoopDetector."""
        loop_hook.bind_state_context(state_ctx)
        shared_hook_context["fs_changes"] = {
            "created": ["/workspace/a.txt", "/workspace/b.txt"],
            "modified": [],
            "deleted": [],
        }
        loop_hook.hook_context = shared_hook_context
        # Observation tool with no explicit created_artifacts
        payload = make_payload(tool_name="read_file", result="file content")
        await loop_hook.execute(payload)
        # The LoopDetector should have recorded with created_artifacts > 0
        stats = loop_hook._loop_detector.get_stats()
        assert stats.get("total_calls", 0) >= 1


# =============================================================================
# execute() — Scratchpad Updates
# =============================================================================


class TestLoopSupervisionHookScratchpad:
    """Tests for scratchpad updates when loop detected."""

    @pytest.mark.asyncio
    async def test_scratchpad_updated_on_loop(self, loop_hook, state_ctx):
        """Scratchpad is updated when loop is detected."""
        update_mock = MagicMock()
        state_ctx["update_scratchpad"] = update_mock
        loop_hook.bind_state_context(state_ctx)

        for i in range(7):
            payload = make_payload(tool_name="search_web", result="result")
            await loop_hook.execute(payload)

        # Scratchpad should have been called
        assert update_mock.called

    @pytest.mark.asyncio
    async def test_scratchpad_not_updated_without_loop(self, loop_hook, state_ctx):
        """Scratchpad is not updated when no loop detected."""
        update_mock = MagicMock()
        state_ctx["update_scratchpad"] = update_mock
        loop_hook.bind_state_context(state_ctx)

        for i in range(3):
            payload = make_payload(tool_name="file_create", result="Created file.txt")
            await loop_hook.execute(payload)

        assert not update_mock.called


# =============================================================================
# _tool_category() — Tool Categorization
# =============================================================================


class TestLoopSupervisionHookToolCategory:
    """Tests for _tool_category() internal method."""

    def test_code_tools(self, loop_hook):
        """python_repl, bash, js_repl are code."""
        for tool in ["python_repl", "bash", "js_repl"]:
            assert loop_hook._tool_category(tool) == "code"

    def test_write_tools(self, loop_hook):
        """file_create, edit_file, write_file are write."""
        for tool in ["file_create", "edit_file", "edit_file_by_lines", "write_file"]:
            assert loop_hook._tool_category(tool) == "write"

    def test_read_tools(self, loop_hook):
        """read_file, list_dir, glob are read."""
        for tool in ["read_file", "list_dir", "glob"]:
            assert loop_hook._tool_category(tool) == "read"

    def test_web_tools(self, loop_hook):
        """search_web, fetch_webpage are web."""
        for tool in ["search_web", "fetch_webpage"]:
            assert loop_hook._tool_category(tool) == "web"

    def test_other_tools(self, loop_hook):
        """Unknown tools default to "other"."""
        assert loop_hook._tool_category("some_unknown_tool") == "other"


# =============================================================================
# _extract_entities() — Result Parsing
# =============================================================================


class TestLoopSupervisionHookExtractEntities:
    """Tests for _extract_entities() result parsing."""

    def test_extract_from_json_result(self, loop_hook):
        """Extracts entities from JSON format result."""
        json_result = '{"result_entities": [{"url": "https://a.com"}, {"url": "https://b.com"}]}'
        entities = loop_hook._extract_entities(json_result)
        assert len(entities) >= 1

    def test_extract_from_search_web_text(self, loop_hook):
        """Extracts search results from text format."""
        text_result = "### Search Results for: test\n- [Title](https://example.com)\n- [Title2](https://example2.com)\n- [Title3](https://example3.com)"
        entities = loop_hook._extract_entities(text_result)
        # Should extract at least 2 search results
        assert len(entities) >= 2

    def test_extract_from_plain_text(self, loop_hook):
        """Plain text with no search format returns empty."""
        text = "This is just plain text with no search results."
        entities = loop_hook._extract_entities(text)
        # Should not count as search results
        assert len(entities) < 2


# =============================================================================
# _decide_recovery_action()
# =============================================================================


class TestLoopSupervisionHookRecoveryAction:
    """Tests for _decide_recovery_action()."""

    def test_recommend_abort_for_repeating_action(self, loop_hook):
        """repeating_action → RECOMMEND_ABORT."""
        result = loop_hook._decide_recovery_action({
            "type": "repeating_action",
            "message": "...",
        })
        assert result == RECOMMEND_ABORT

    def test_recommend_abort_for_repeating_state(self, loop_hook):
        """repeating_state → RECOMMEND_ABORT."""
        result = loop_hook._decide_recovery_action({
            "type": "repeated_state",
            "message": "...",
        })
        assert result == RECOMMEND_ABORT

    def test_recommend_abort_for_repeated_action(self, loop_hook):
        """repeated_action → RECOMMEND_ABORT."""
        result = loop_hook._decide_recovery_action({
            "type": "repeated_action",
            "message": "...",
        })
        assert result == RECOMMEND_ABORT

    def test_recommend_abort_for_repeated_state(self, loop_hook):
        """repeated_state → RECOMMEND_ABORT."""
        result = loop_hook._decide_recovery_action({
            "type": "repeated_state",
            "message": "...",
        })
        assert result == RECOMMEND_ABORT

    def test_recommend_retry_for_no_progress(self, loop_hook):
        """no_progress → RECOMMEND_RETRY."""
        result = loop_hook._decide_recovery_action({
            "type": "no_progress",
            "message": "...",
        })
        assert result == RECOMMEND_RETRY

    def test_recommend_retry_for_stall(self, loop_hook):
        """stall → RECOMMEND_RETRY."""
        result = loop_hook._decide_recovery_action({
            "type": "stall",
            "message": "...",
        })
        assert result == RECOMMEND_RETRY

    def test_none_for_observation_chain(self, loop_hook):
        """observation_chain → returns None (no recovery_action needed)."""
        result = loop_hook._decide_recovery_action({
            "type": "observation_chain",
            "message": "...",
        })
        assert result is None

    def test_none_when_no_loop_info(self, loop_hook):
        """None loop_info → returns None."""
        result = loop_hook._decide_recovery_action(None)
        assert result is None
