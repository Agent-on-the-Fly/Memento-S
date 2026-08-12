# SPDX-License-Identifier: Apache-2.0
"""ToolResultSupervisionHook — ToolResultProcessor 的补充层。

在 _result_processor.process() 之后执行，对已分类的结果做补充分析。

注意：
  - ToolResultProcessor 是主流程（本文件不再重复调用 classify/assess_execution_state）
  - 本 hook 做补充分析，如：错误模式检测（ErrorPatternSupervisionHook 职责）、deferred_messages 注入等
  - hook 将补充信息写入 HookResult.metadata / hook_context
"""

from __future__ import annotations

from shared.hooks.executor import HookDefinition
from shared.hooks.types import HookEvent, HookPayload, HookResult
from utils.logger import get_logger

logger = get_logger(__name__)


class ToolResultSupervisionHook(HookDefinition):
    """
    Tool Result 监督 Hook — 在 processor 处理后的结果上做补充分析。

    注意：错误分类、状态评估已在 ToolResultProcessor 中完成。
    本 hook 仅做补充分析，如有必要可扩展。

    使用方式：
        hook_executor.register(HookEvent.AFTER_TOOL_EXEC, ToolResultSupervisionHook())
    """

    async def execute(self, payload: HookPayload) -> HookResult:
        """执行补充分析。AFTER_TOOL_EXEC 之外直接放行。"""
        if payload.event != HookEvent.AFTER_TOOL_EXEC:
            return HookResult(allowed=True)

        tool_name = payload.tool_name
        result = payload.result or ""

        logger.debug(
            f"[ToolResultSupervisionHook] supplement analysis for tool={tool_name}, "
            f"result_preview='{str(result)[:100]}'"
        )

        return HookResult(allowed=True)

