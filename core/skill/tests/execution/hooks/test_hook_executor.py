# SPDX-License-Identifier: Apache-2.0
"""Tests for HookExecutor — shared/hooks/executor.py"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from shared.hooks.executor import HookDefinition, HookExecutor, CommandHook
from shared.hooks.types import HookEvent, HookPayload, HookResult


class MockHook(HookDefinition):
    """Simple mock hook for testing."""

    def __init__(self, allowed: bool = True, name: str = "mock"):
        super().__init__()
        self.allowed = allowed
        self.name = name
        self.call_count = 0
        self.last_payload = None

    async def execute(self, payload: HookPayload) -> HookResult:
        self.call_count += 1
        self.last_payload = payload
        return HookResult(allowed=self.allowed, reason="")


# =============================================================================
# HookExecutor Initialization
# =============================================================================


class TestHookExecutorInit:
    """Tests for HookExecutor initialization."""

    def test_empty_init(self):
        """Empty executor has no hooks registered."""
        executor = HookExecutor()
        assert executor._hooks == {}
        assert executor._hook_context == {}

    def test_init_with_hooks(self):
        """Executor can be initialized with hooks dict."""
        mock = MockHook()
        hooks = {HookEvent.BEFORE_TOOL_EXEC: [mock]}
        executor = HookExecutor(hooks=hooks)
        assert HookEvent.BEFORE_TOOL_EXEC in executor._hooks
        assert mock in executor._hooks[HookEvent.BEFORE_TOOL_EXEC]


# =============================================================================
# register() / unregister()
# =============================================================================


class TestHookExecutorRegistration:
    """Tests for hook registration and unregistration."""

    def test_register_single_hook(self):
        """register() adds a hook to the correct event."""
        executor = HookExecutor()
        mock = MockHook()
        executor.register(HookEvent.BEFORE_TOOL_EXEC, mock)
        assert mock in executor._hooks[HookEvent.BEFORE_TOOL_EXEC]

    def test_register_multiple_hooks_same_event(self):
        """Multiple hooks can be registered for the same event."""
        executor = HookExecutor()
        mock1 = MockHook(name="hook1")
        mock2 = MockHook(name="hook2")
        executor.register(HookEvent.AFTER_TOOL_EXEC, mock1)
        executor.register(HookEvent.AFTER_TOOL_EXEC, mock2)
        assert len(executor._hooks[HookEvent.AFTER_TOOL_EXEC]) == 2

    def test_register_same_hook_twice(self):
        """Registering the same hook twice adds it twice."""
        executor = HookExecutor()
        mock = MockHook()
        executor.register(HookEvent.BEFORE_SKILL_EXEC, mock)
        executor.register(HookEvent.BEFORE_SKILL_EXEC, mock)
        assert len(executor._hooks[HookEvent.BEFORE_SKILL_EXEC]) == 2

    def test_unregister_existing_hook(self):
        """unregister() removes a hook and returns True."""
        executor = HookExecutor()
        mock = MockHook()
        executor.register(HookEvent.BEFORE_TOOL_EXEC, mock)
        result = executor.unregister(HookEvent.BEFORE_TOOL_EXEC, mock)
        assert result is True
        assert mock not in executor._hooks[HookEvent.BEFORE_TOOL_EXEC]

    def test_unregister_nonexistent_hook(self):
        """unregister() returns False when hook not found."""
        executor = HookExecutor()
        mock = MockHook()
        result = executor.unregister(HookEvent.AFTER_TOOL_EXEC, mock)
        assert result is False

    def test_clear_all_hooks(self):
        """clear() without event clears all hooks."""
        executor = HookExecutor()
        mock1 = MockHook()
        mock2 = MockHook()
        executor.register(HookEvent.BEFORE_TOOL_EXEC, mock1)
        executor.register(HookEvent.AFTER_TOOL_EXEC, mock2)
        executor.clear()
        assert executor._hooks == {}

    def test_clear_specific_event(self):
        """clear(event) clears only that event's hooks."""
        executor = HookExecutor()
        mock1 = MockHook()
        mock2 = MockHook()
        executor.register(HookEvent.BEFORE_TOOL_EXEC, mock1)
        executor.register(HookEvent.AFTER_TOOL_EXEC, mock2)
        executor.clear(HookEvent.BEFORE_TOOL_EXEC)
        # AFTER_TOOL_EXEC should still exist
        assert HookEvent.AFTER_TOOL_EXEC in executor._hooks
        # BEFORE_TOOL_EXEC should be cleared
        assert executor._hooks.get(HookEvent.BEFORE_TOOL_EXEC, []) == []


# =============================================================================
# execute() — Core Behavior
# =============================================================================


class TestHookExecutorExecute:
    """Tests for HookExecutor.execute()."""

    @pytest.mark.asyncio
    async def test_executes_registered_hooks(self):
        """execute() calls all hooks registered for the event."""
        executor = HookExecutor()
        mock1 = MockHook(name="hook1")
        mock2 = MockHook(name="hook2")
        executor.register(HookEvent.AFTER_TOOL_EXEC, mock1)
        executor.register(HookEvent.AFTER_TOOL_EXEC, mock2)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC, tool_name="bash")
        result = await executor.execute(HookEvent.AFTER_TOOL_EXEC, payload)
        assert mock1.call_count == 1
        assert mock2.call_count == 1
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_no_hooks_registered(self):
        """execute() with no hooks returns allowed=True."""
        executor = HookExecutor()
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC)
        result = await executor.execute(HookEvent.AFTER_TOOL_EXEC, payload)
        assert result.allowed is True
        assert result.reason == ""

    @pytest.mark.asyncio
    async def test_hooks_called_in_registration_order(self):
        """Hooks are called in the order they were registered."""
        executor = HookExecutor()
        call_order = []
        for i in range(3):
            mock = MockHook(name=f"hook{i}")
            mock.execute = AsyncMock(side_effect=lambda m, o=call_order, n=i: o.append(n) or HookResult(allowed=True))
            executor.register(HookEvent.BEFORE_TOOL_EXEC, mock)
        payload = HookPayload(event=HookEvent.BEFORE_TOOL_EXEC)
        await executor.execute(HookEvent.BEFORE_TOOL_EXEC, payload)
        assert call_order == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_other_event_hooks_not_called(self):
        """Hooks registered for other events are not called."""
        executor = HookExecutor()
        mock_after = MockHook(name="after")
        mock_before = MockHook(name="before")
        executor.register(HookEvent.AFTER_TOOL_EXEC, mock_after)
        executor.register(HookEvent.BEFORE_TOOL_EXEC, mock_before)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC)
        await executor.execute(HookEvent.AFTER_TOOL_EXEC, payload)
        assert mock_after.call_count == 1
        assert mock_before.call_count == 0


# =============================================================================
# execute() — Aggregation
# =============================================================================


class TestHookExecutorAggregation:
    """Tests for HookResult aggregation in execute()."""

    @pytest.mark.asyncio
    async def test_aggregates_detected_artifacts(self):
        """detected_artifacts from all hooks are merged and deduplicated."""
        executor = HookExecutor()
        hook1 = MockHook()
        hook1.execute = AsyncMock(return_value=HookResult(
            allowed=True, detected_artifacts=["/path/a.txt", "/path/b.txt"]
        ))
        hook2 = MockHook()
        hook2.execute = AsyncMock(return_value=HookResult(
            allowed=True, detected_artifacts=["/path/b.txt", "/path/c.txt"]
        ))
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook1)
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook2)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC)
        result = await executor.execute(HookEvent.AFTER_TOOL_EXEC, payload)
        assert result.detected_artifacts == ["/path/a.txt", "/path/b.txt", "/path/c.txt"]

    @pytest.mark.asyncio
    async def test_aggregates_deferred_messages(self):
        """deferred_messages from all hooks are concatenated."""
        executor = HookExecutor()
        hook1 = MockHook()
        hook1.execute = AsyncMock(return_value=HookResult(
            allowed=True, deferred_messages=[{"role": "user", "content": "msg1"}]
        ))
        hook2 = MockHook()
        hook2.execute = AsyncMock(return_value=HookResult(
            allowed=True, deferred_messages=[{"role": "user", "content": "msg2"}]
        ))
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook1)
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook2)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC)
        result = await executor.execute(HookEvent.AFTER_TOOL_EXEC, payload)
        assert len(result.deferred_messages) == 2
        assert result.deferred_messages[0]["content"] == "msg1"
        assert result.deferred_messages[1]["content"] == "msg2"

    @pytest.mark.asyncio
    async def test_aggregates_fs_changes(self):
        """fs_changes from all hooks are merged."""
        executor = HookExecutor()
        hook1 = MockHook()
        hook1.execute = AsyncMock(return_value=HookResult(
            allowed=True,
            fs_changes={"created": ["/path/a.txt"], "modified": [], "deleted": []}
        ))
        hook2 = MockHook()
        hook2.execute = AsyncMock(return_value=HookResult(
            allowed=True,
            fs_changes={"created": [], "modified": ["/path/b.txt"], "deleted": []}
        ))
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook1)
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook2)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC)
        result = await executor.execute(HookEvent.AFTER_TOOL_EXEC, payload)
        assert result.fs_changes is not None
        assert "/path/a.txt" in result.fs_changes["created"]
        assert "/path/b.txt" in result.fs_changes["modified"]

    @pytest.mark.asyncio
    async def test_recovery_action_priority_recommend_abort(self):
        """RECOMMEND_ABORT has highest priority over RECOMMEND_RETRY."""
        executor = HookExecutor()
        hook1 = MockHook()
        hook1.execute = AsyncMock(return_value=HookResult(
            allowed=True, recovery_action="RECOMMEND_RETRY"
        ))
        hook2 = MockHook()
        hook2.execute = AsyncMock(return_value=HookResult(
            allowed=True, recovery_action="RECOMMEND_ABORT"
        ))
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook1)
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook2)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC)
        result = await executor.execute(HookEvent.AFTER_TOOL_EXEC, payload)
        assert result.recovery_action == "RECOMMEND_ABORT"

    @pytest.mark.asyncio
    async def test_recovery_action_priority_continue(self):
        """CONTINUE has lower priority than RECOMMEND_ABORT."""
        executor = HookExecutor()
        hook1 = MockHook()
        hook1.execute = AsyncMock(return_value=HookResult(
            allowed=True, recovery_action="CONTINUE"
        ))
        hook2 = MockHook()
        hook2.execute = AsyncMock(return_value=HookResult(
            allowed=True, recovery_action="RECOMMEND_ABORT"
        ))
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook1)
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook2)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC)
        result = await executor.execute(HookEvent.AFTER_TOOL_EXEC, payload)
        assert result.recovery_action == "RECOMMEND_ABORT"


# =============================================================================
# execute() — Blocking & Error Handling
# =============================================================================


class TestHookExecutorBlockingAndErrors:
    """Tests for blocking logic and exception handling."""

    @pytest.mark.asyncio
    async def test_first_blocking_hook_sets_reason(self):
        """First allowed=False sets blocked_reason, remaining hooks still run."""
        executor = HookExecutor()
        call_count = [0]

        async def blocking_hook(payload):
            call_count[0] += 1
            return HookResult(allowed=False, reason="dangerous command")

        hook1 = MockHook()
        hook1.execute = AsyncMock(side_effect=blocking_hook)
        hook2 = MockHook()
        executor.register(HookEvent.BEFORE_TOOL_EXEC, hook1)
        executor.register(HookEvent.BEFORE_TOOL_EXEC, hook2)
        payload = HookPayload(event=HookEvent.BEFORE_TOOL_EXEC, tool_name="rm")
        result = await executor.execute(HookEvent.BEFORE_TOOL_EXEC, payload)
        assert result.allowed is False
        assert "dangerous command" in result.reason
        # hook2 still ran
        assert hook2.call_count == 1

    @pytest.mark.asyncio
    async def test_all_hooks_run_even_if_first_blocks(self):
        """All hooks run even when first returns allowed=False."""
        executor = HookExecutor()
        call_counts = {"blocked": 0, "allowed": 0, "final": 0}

        async def always_block(payload):
            call_counts["blocked"] += 1
            return HookResult(allowed=False, reason="blocked")

        async def always_allow(payload):
            call_counts["allowed"] += 1
            return HookResult(allowed=True)

        hook1 = MockHook()
        hook1.execute = AsyncMock(side_effect=always_block)
        hook2 = MockHook()
        hook2.execute = AsyncMock(side_effect=always_allow)
        hook3 = MockHook()
        hook3.execute = AsyncMock(side_effect=always_allow)
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook1)
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook2)
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook3)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC)
        result = await executor.execute(HookEvent.AFTER_TOOL_EXEC, payload)
        assert result.allowed is False
        # All 3 hooks ran
        assert call_counts["blocked"] == 1
        assert call_counts["allowed"] == 2

    @pytest.mark.asyncio
    async def test_hook_exception_caught_and_returns_allowed_true(self):
        """Exception in hook does not crash executor; returns allowed=True with reason."""
        executor = HookExecutor()
        failing_hook = MockHook()
        failing_hook.execute = AsyncMock(side_effect=RuntimeError("hook crashed"))
        hook2 = MockHook()

        executor.register(HookEvent.AFTER_TOOL_EXEC, failing_hook)
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook2)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC)
        result = await executor.execute(HookEvent.AFTER_TOOL_EXEC, payload)
        # Hook2 still ran
        assert hook2.call_count == 1
        # Exception does not crash executor
        assert isinstance(result.reason, str)

    @pytest.mark.asyncio
    async def test_hook_context_is_shared_between_hooks(self):
        """hook_context is shared across hooks in the same execute() call."""
        executor = HookExecutor()
        context_captures = []

        async def capture_hook1(payload: HookPayload) -> HookResult:
            executor._hook_context["from_hook1"] = "shared_value"
            context_captures.append(dict(executor._hook_context))
            return HookResult(allowed=True)

        async def capture_hook2(payload: HookPayload) -> HookResult:
            context_captures.append(dict(executor._hook_context))
            return HookResult(allowed=True)

        hook1 = MockHook()
        hook1.execute = AsyncMock(side_effect=capture_hook1)
        hook2 = MockHook()
        hook2.execute = AsyncMock(side_effect=capture_hook2)
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook1)
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook2)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC)
        await executor.execute(HookEvent.AFTER_TOOL_EXEC, payload)
        # hook2 should see the value written by hook1
        assert len(context_captures) == 2
        assert "from_hook1" in context_captures[1]

    @pytest.mark.asyncio
    async def test_hook_context_reset_between_calls(self):
        """hook_context is cleared between different execute() calls."""
        executor = HookExecutor()
        context_values = []

        async def write_and_read(payload: HookPayload) -> HookResult:
            executor._hook_context["key"] = "value_from_call"
            context_values.append(dict(executor._hook_context))
            return HookResult(allowed=True)

        hook = MockHook()
        hook.execute = AsyncMock(side_effect=write_and_read)
        executor.register(HookEvent.AFTER_TOOL_EXEC, hook)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC)

        await executor.execute(HookEvent.AFTER_TOOL_EXEC, payload)
        await executor.execute(HookEvent.AFTER_TOOL_EXEC, payload)

        assert len(context_values) == 2
        # Each execute() call has its own isolated context snapshot
        assert context_values[0] == {"key": "value_from_call"}
        assert context_values[1] == {"key": "value_from_call"}


# =============================================================================
# CommandHook
# =============================================================================


class TestCommandHook:
    """Tests for CommandHook."""

    @pytest.mark.asyncio
    async def test_sync_function(self):
        """CommandHook works with synchronous function."""

        def validator(payload: HookPayload) -> HookResult:
            if payload.tool_name == "rm":
                return HookResult(allowed=False, reason="rm blocked")
            return HookResult(allowed=True)

        hook = CommandHook(validator)
        payload = HookPayload(event=HookEvent.BEFORE_TOOL_EXEC, tool_name="rm")
        result = await hook.execute(payload)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_async_function(self):
        """CommandHook works with async function."""

        async def validator(payload: HookPayload) -> HookResult:
            return HookResult(allowed=True, reason="ok")

        hook = CommandHook(validator)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC)
        result = await hook.execute(payload)
        assert result.allowed is True
