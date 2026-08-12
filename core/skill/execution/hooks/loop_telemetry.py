# SPDX-License-Identifier: Apache-2.0
"""LoopTelemetryHook — Loop 检测遥测 Hook。

监听 ON_LOOP_DETECTED 事件，记录遥测数据（日志、metrics）。
不影响 agent 决策，纯被动监控。

使用方式：
    hook_executor.register(HookEvent.ON_LOOP_DETECTED, LoopTelemetryHook())

触发方式：
    SkillAgent 内检测到 loop 时，通过 hook_executor 主动触发：
        await hook_executor.execute(HookEvent.ON_LOOP_DETECTED, payload)
"""

from __future__ import annotations

from typing import Any

from shared.hooks.executor import HookDefinition
from shared.hooks.types import HookEvent, HookPayload, HookResult
from utils.logger import get_logger

logger = get_logger(__name__)


class LoopTelemetryHook(HookDefinition):
    """
    Loop 检测遥测 Hook — 被动记录，不影响 agent 决策。

    在 ON_LOOP_DETECTED 时记录：
    - loop 类型和详细信息
    - 触发时的工具名和参数
    - 当前 turn 编号
    """

    def __init__(self, log_level: str = "info") -> None:
        """
        Args:
            log_level: 日志级别（debug/info/warning）。
        """
        super().__init__()
        self._log_level = log_level

    async def execute(self, payload: HookPayload) -> HookResult:
        """执行遥测记录。BEFORE_TOOL_EXEC 之外直接放行。"""
        if payload.event != HookEvent.ON_LOOP_DETECTED:
            return HookResult(allowed=True)

        metadata = payload.metadata or {}
        loop_info = metadata.get("loop_info", {})
        tool_name = payload.tool_name
        args = payload.args or {}

        log_msg = (
            f"[LoopTelemetry] ON_LOOP_DETECTED: "
            f"type={loop_info.get('type', 'unknown')}, "
            f"message={loop_info.get('message', '')}, "
            f"tool={tool_name}, "
            f"turn={metadata.get('turn', '?')}, "
            f"args_keys={list(args.keys())}"
        )

        if self._log_level == "debug":
            logger.debug(log_msg)
        elif self._log_level == "warning":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # 可扩展：上报 metrics、发送 webhook 等
        self._report_metrics(loop_info, tool_name, metadata)

        return HookResult(allowed=True)

    def _report_metrics(
        self,
        loop_info: dict,
        tool_name: str,
        metadata: dict,
    ) -> None:
        """扩展点：上报遥测数据（可接入 Prometheus、DataDog 等）。"""
        # 目前只记录到日志
        # TODO: 可扩展为：
        #   - 指标库计数
        #   - 事件追踪（opentelemetry span）
        #   - 告警（连续多次 RECOMMEND_ABORT）
        pass
