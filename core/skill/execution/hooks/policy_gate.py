# SPDX-License-Identifier: Apache-2.0
"""PolicyGateHook — ToolGate 的 Hook 化改造，统一 Policy 策略入口。

将 SkillToolAdapter 内独立的 ToolGate.check() 调用重构为 HookDefinition，
通过全局 HookExecutor 统一管理，在 BEFORE_TOOL_EXEC 事件时执行策略检查。

替代：SkillToolAdapter 内的独立 ToolGate.check() 调用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shared.hooks.executor import HookDefinition
from shared.hooks.types import HookEvent, HookPayload, HookResult

if TYPE_CHECKING:
    from shared.security.policy import PolicyManager


class PolicyGateHook(HookDefinition):
    """
    Policy Gate Hook — 在 BEFORE_TOOL_EXEC 时执行 PolicyManager 策略检查。

    所有 Tool 级别的策略检查（危险命令黑名单、速率限制等）统一通过此 Hook 执行。
    """

    def __init__(self, policy_manager: "PolicyManager | None" = None) -> None:
        super().__init__()
        from core.skill.execution.policy.tool_gate import ToolGate

        self._gate = ToolGate(policy_manager=policy_manager)

    async def execute(self, payload: HookPayload) -> HookResult:
        """执行策略检查。BEFORE_TOOL_EXEC 之外的事件直接放行。"""
        if payload.event != HookEvent.BEFORE_TOOL_EXEC:
            return HookResult(allowed=True)

        decision = self._gate.check(
            tool_name=payload.tool_name,
            args=payload.args or {},
        )
        return HookResult(
            allowed=decision.allowed,
            reason=decision.reason or "",
            error_context=decision.detail,
        )
