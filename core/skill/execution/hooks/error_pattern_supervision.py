# SPDX-License-Identifier: Apache-2.0
"""ErrorPatternSupervisionHook — StatefulErrorPatternDetector 的 Hook 封装。

职责：
  1. record_error 到 state
  2. StatefulErrorPatternDetector.analyze()
  3. 构建 deferred_messages（recovery 部分）
  4. 更新 scratchpad
"""

from __future__ import annotations

from typing import Any

from shared.hooks.executor import HookDefinition
from shared.hooks.types import HookEvent, HookPayload, HookResult
from core.skill.execution.error_recovery import StatefulErrorPatternDetector
from utils.log_config import log_preview
from utils.logger import get_logger

logger = get_logger(__name__)

CONTINUE = "CONTINUE"


class ErrorPatternSupervisionHook(HookDefinition):
    """
    Error Pattern 监督 Hook — 封装 StatefulErrorPatternDetector 的检测逻辑。

    在 AFTER_TOOL_EXEC 时执行，执行顺序：
      1. record_error 到 state
      2. StatefulErrorPatternDetector.analyze()
      3. 构建 deferred_messages（recovery 部分）
      4. 更新 scratchpad
    """

    def __init__(self) -> None:
        """初始化 ErrorPatternSupervisionHook。"""
        super().__init__()
        self._state_ctx: dict[str, Any] | None = None

    def bind_state_context(self, state_ctx: dict) -> None:
        """由 SkillAgent 在初始化时调用，传入必要的状态引用。

        Args:
            state_ctx: 包含以下键的字典：
                - turn_count: int（当前 turn 编号）
                - error_history: list（错误历史）
                - action_history: list（操作历史）
                - record_error: callable
                - should_inject_recovery_hint: callable
                - mark_recovery_hint_injected: callable
                - update_scratchpad: callable
        """
        self._state_ctx = state_ctx

    async def execute(self, payload: HookPayload) -> HookResult:
        """执行错误模式检测逻辑。BEFORE_TOOL_EXEC 之外直接放行。"""
        if payload.event != HookEvent.AFTER_TOOL_EXEC:
            return HookResult(allowed=True)

        tool_name = payload.tool_name
        result = payload.result or ""
        state_ctx = self._state_ctx or {}

        # ── 1. 错误检测 ───────────────────────────────────────────────────
        recovery_hints: list[dict[str, Any]] = []
        raw_error = self._extract_error(result)

        if raw_error:
            record_error = state_ctx.get("record_error")
            if record_error:
                try:
                    record_error(
                        error=raw_error,
                        tool_name=tool_name,
                        hint_injected=False,
                    )
                except Exception as e:
                    logger.warning(
                        "[ErrorPatternSupervisionHook] record_error failed: {}", e
                    )

            # ── 2. 分析错误模式 ───────────────────────────────────────────
            should_inject = state_ctx.get("should_inject_recovery_hint")
            if should_inject and callable(should_inject):
                try:
                    if should_inject():
                        recovery_hints = StatefulErrorPatternDetector.analyze(
                            state_ctx.get("error_history", []),
                            action_history=state_ctx.get("action_history", []),
                        )
                        mark_injected = state_ctx.get("mark_recovery_hint_injected")
                        if mark_injected and callable(mark_injected):
                            mark_injected()
                except Exception as e:
                    logger.warning(
                        "[ErrorPatternSupervisionHook] recovery hint check failed: {}", e
                    )

        # ── 3. deferred_messages ─────────────────────────────────────────
        deferred_messages: list[dict[str, Any]] = []
        for hint in recovery_hints:
            deferred_messages.append({
                "role": "user",
                "content": (
                    f"[System] ERROR_RECOVERY_HINT [{hint['pattern']}]: "
                    f"{hint['hint']}"
                ),
            })
            self._update_scratchpad(
                state_ctx,
                f"[RECOVERY] {hint['pattern']}: "
                f"{log_preview(hint.get('hint', ''), default=100)}",
            )

        # ── 4. recovery_action ───────────────────────────────────────────
        recovery_action = CONTINUE if recovery_hints else None

        return HookResult(
            allowed=True,
            deferred_messages=deferred_messages if deferred_messages else None,
            recovery_action=recovery_action,
            metadata={"recovery_hints": recovery_hints},
        )

    # ── 内部辅助方法 ──────────────────────────────────────────────────────

    def _extract_error(self, result: str) -> str:
        """从工具结果中提取错误信息。"""
        if result and ("error" in result.lower()[:200] or "err:" in result[:200]):
            return result[:500]
        return ""

    def _update_scratchpad(self, state_ctx: dict, text: str) -> None:
        """更新 scratchpad。"""
        update_scratchpad = state_ctx.get("update_scratchpad")
        if update_scratchpad and callable(update_scratchpad):
            try:
                update_scratchpad(text)
            except Exception:
                pass
