# SPDX-License-Identifier: Apache-2.0
"""Tests for ToolResultSupervisionHook — core/skill/execution/hooks/tool_result_supervision.py

注意：错误分类已在 ToolResultProcessor（adapter.py 先调用）中完成。
本 hook 仅做补充分析，不重复分类逻辑。
"""


from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from shared.hooks.types import HookEvent, HookPayload
from core.skill.execution.hooks.tool_result_supervision import (
    ToolResultSupervisionHook,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hook():
    """ToolResultSupervisionHook instance."""
    return ToolResultSupervisionHook()


@pytest.fixture
def hook_executor_context():
    """Simulate hook_context shared across hooks in HookExecutor."""
    return {}


# =============================================================================
# Helpers
# =============================================================================


def make_payload(
    event: HookEvent = HookEvent.AFTER_TOOL_EXEC,
    tool_name: str = "bash",
    result: str = "done",
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


class TestToolResultSupervisionHookInit:
    """Tests for ToolResultSupervisionHook initialization."""

    def test_init(self):
        """Hook initializes correctly."""
        hook = ToolResultSupervisionHook()
        assert hook is not None


# =============================================================================
# Non-AFTER_TOOL_EXEC events — pass through
# =============================================================================


class TestNonAfterToolExecPassthrough:
    """Non-AFTER_TOOL_EXEC 事件直接放行。"""

    @pytest.mark.asyncio
    async def test_before_tool_exec_passthrough(self, hook, hook_executor_context):
        """BEFORE_TOOL_EXEC 返回 allowed=True。"""
        hook.hook_context = hook_executor_context
        payload = make_payload(event=HookEvent.BEFORE_TOOL_EXEC, result="")
        result = await hook.execute(payload)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_before_skill_exec_passthrough(self, hook, hook_executor_context):
        """BEFORE_SKILL_EXEC 返回 allowed=True。"""
        hook.hook_context = hook_executor_context
        payload = make_payload(event=HookEvent.BEFORE_SKILL_EXEC, result="")
        result = await hook.execute(payload)
        assert result.allowed is True


# =============================================================================
# AFTER_TOOL_EXEC — supplement only (no classification)
# =============================================================================


class TestAfterToolExecSupplement:
    """AFTER_TOOL_EXEC: hook 做补充分析，不重复分类逻辑。"""

    @pytest.mark.asyncio
    async def test_after_tool_exec_returns_allowed_true(self, hook, hook_executor_context):
        """AFTER_TOOL_EXEC 返回 allowed=True，不重复分类。"""
        hook.hook_context = hook_executor_context
        payload = make_payload(
            tool_name="bash",
            result="File created successfully at /tmp/out.txt",
        )
        result = await hook.execute(payload)
        assert result.allowed is True
        # hook 不返回 metadata（含分类结果）—— 分类由 processor 完成
        assert result.metadata is None

    @pytest.mark.asyncio
    async def test_after_tool_exec_error_result(self, hook, hook_executor_context):
        """错误结果也不在 hook 中重复分类（processor 已完成）。"""
        hook.hook_context = hook_executor_context
        payload = make_payload(
            tool_name="bash",
            result="Error: permission denied to write /etc/passwd",
        )
        result = await hook.execute(payload)
        assert result.allowed is True
        # metadata 为 None —— 分类已在 processor 中完成
        assert result.metadata is None

    @pytest.mark.asyncio
    async def test_after_tool_exec_receives_result(self, hook, hook_executor_context):
        """AFTER_TOOL_EXEC 接收 tool_result 做补充分析。"""
        hook.hook_context = hook_executor_context
        payload = make_payload(
            tool_name="read_file",
            result="File content here...",
            args={"path": "/workspace/test.py"},
        )
        result = await hook.execute(payload)
        assert result.allowed is True
        # hook 可读取 result 做补充（当前实现仅为 allowed=True）
