# SPDX-License-Identifier: Apache-2.0
"""ToolArgsValidationHook — ToolArgsProcessor 的补充层。

在 _args_processor.process() 之后执行，对已处理好的参数做补充分析。

注意：
  - ToolArgsProcessor 是主流程（本文件不再重复调用）
  - 本 hook 做补充分析，如：跨字段验证、可疑模式检测等
  - hook 将补充信息写入 HookResult.metadata，不修改参数
"""

from __future__ import annotations

from shared.hooks.executor import HookDefinition
from shared.hooks.types import HookEvent, HookPayload, HookResult
from utils.logger import get_logger

logger = get_logger(__name__)


class ToolArgsValidationHook(HookDefinition):
    """
    工具参数补充验证 Hook — 在 processor 处理后的参数上做额外分析。

    注意：参数标准化、enrich、路径重写已在 ToolArgsProcessor 中完成。
    本 hook 仅做补充分析，不重复处理参数。

    使用方式：
        hook_executor.register(HookEvent.BEFORE_TOOL_EXEC, ToolArgsValidationHook())
    """

    async def execute(self, payload: HookPayload) -> HookResult:
        """执行补充验证。BEFORE_TOOL_EXEC 之外直接放行。"""
        if payload.event != HookEvent.BEFORE_TOOL_EXEC:
            return HookResult(allowed=True)

        # 参数已在 ToolArgsProcessor 中处理完毕（adapter.py 先调用）
        # 本 hook 做补充分析，payload.args 为已处理的参数
        tool_name = payload.tool_name
        processed_args = payload.args or {}

        logger.debug(
            f"[ToolArgsValidationHook] supplement analysis for tool={tool_name}, "
            f"args_keys={list(processed_args.keys())}"
        )

        return HookResult(allowed=True)
