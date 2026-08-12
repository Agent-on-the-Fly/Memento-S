# SPDX-License-Identifier: Apache-2.0
"""Tests for ToolArgsValidationHook — core/skill/execution/hooks/tool_args_hook.py

注意：参数标准化已在 ToolArgsProcessor（adapter.py 先调用）中完成。
本 hook 仅做补充分析，不重复处理参数。
"""


from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from shared.hooks.types import HookEvent, HookPayload
from core.skill.execution.hooks.tool_args_hook import ToolArgsValidationHook


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hook():
    """ToolArgsValidationHook instance."""
    return ToolArgsValidationHook()


@pytest.fixture
def hook_executor_context():
    """Simulate hook_context shared across hooks in HookExecutor."""
    return {}


# =============================================================================
# Helpers
# =============================================================================


def make_payload(
    event: HookEvent = HookEvent.BEFORE_TOOL_EXEC,
    tool_name: str = "read_file",
    args: dict | None = None,
    context=None,
) -> HookPayload:
    """Helper to create HookPayload for testing."""
    return HookPayload(
        event=event,
        tool_name=tool_name,
        args=args or {},
        context=context,
    )


# =============================================================================
# Initialization
# =============================================================================


class TestToolArgsValidationHookInit:
    """Tests for ToolArgsValidationHook initialization."""

    def test_init(self):
        """Hook initializes without args_processor."""
        hook = ToolArgsValidationHook()
        # hook 不再持有 ToolArgsProcessor（processor 在 adapter.py 中先调用）
        assert hook is not None


# =============================================================================
# BEFORE_TOOL_EXEC — supplement only
# =============================================================================


class TestBeforeToolExecSupplement:
    """BEFORE_TOOL_EXEC: hook 做补充分析，不修改参数。"""

    @pytest.mark.asyncio
    async def test_before_tool_exec_returns_allowed_true(self, hook, hook_executor_context):
        """BEFORE_TOOL_EXEC 返回 allowed=True，不修改参数。"""
        hook.hook_context = hook_executor_context
        payload = make_payload(
            event=HookEvent.BEFORE_TOOL_EXEC,
            tool_name="read_file",
            args={"path": "/workspace/test.py", "skill_name": "my_skill"},
            context=MagicMock(),
        )
        result = await hook.execute(payload)
        assert result.allowed is True
        # hook 不返回 modified_args（参数已由 processor 处理）
        assert result.modified_args is None

    @pytest.mark.asyncio
    async def test_before_tool_exec_receives_processed_args(self, hook, hook_executor_context):
        """BEFORE_TOOL_EXEC 接收已处理的 args（来自 adapter.py 的 processor）。"""
        hook.hook_context = hook_executor_context
        processed_args = {"path": "/workspace/test.py", "skill_name": "my_skill", "work_dir": "/workspace"}
        payload = make_payload(
            event=HookEvent.BEFORE_TOOL_EXEC,
            tool_name="bash",
            args=processed_args,
            context=MagicMock(),
        )
        result = await hook.execute(payload)
        assert result.allowed is True
        # hook 接收已处理的 args，不重复处理


# =============================================================================
# Non-BEFORE_TOOL_EXEC events — pass through
# =============================================================================


class TestNonBeforeToolExecPassthrough:
    """Non-BEFORE_TOOL_EXEC 事件直接放行。"""

    @pytest.mark.asyncio
    async def test_after_tool_exec_passthrough(self, hook, hook_executor_context):
        """AFTER_TOOL_EXEC 返回 allowed=True。"""
        hook.hook_context = hook_executor_context
        payload = make_payload(
            event=HookEvent.AFTER_TOOL_EXEC,
            tool_name="read_file",
            args={"path": "test.py"},
            context=MagicMock(),
        )
        result = await hook.execute(payload)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_before_skill_exec_passthrough(self, hook, hook_executor_context):
        """BEFORE_SKILL_EXEC 返回 allowed=True。"""
        hook.hook_context = hook_executor_context
        payload = make_payload(
            event=HookEvent.BEFORE_SKILL_EXEC,
            args={},
            context=None,
        )
        result = await hook.execute(payload)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_on_loop_detected_passthrough(self, hook, hook_executor_context):
        """ON_LOOP_DETECTED 返回 allowed=True。"""
        hook.hook_context = hook_executor_context
        payload = make_payload(
            event=HookEvent.ON_LOOP_DETECTED,
            args={},
            context=None,
        )
        result = await hook.execute(payload)
        assert result.allowed is True
