# SPDX-License-Identifier: Apache-2.0
"""SkillPolicyHook — Skill 级别的策略检查 Hook。

在 BEFORE_SKILL_EXEC 时执行，封装 `run_pre_execute_gate` 的所有检查逻辑：
- API 密钥缺失检查
- Skill 结构完整性检查
- 允许的工具列表检查
- 输入参数 schema 验证

向后兼容：`run_pre_execute_gate` 的内部实现保持不变，
此 Hook 只是将其包装为全局 Hook 系统的一部分。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shared.hooks.executor import HookDefinition
from shared.hooks.types import HookEvent, HookPayload, HookResult
from core.skill.execution.policy.pre_execute import run_pre_execute_gate

if TYPE_CHECKING:
    from core.skill.schema import Skill


class SkillPolicyHook(HookDefinition):
    """
    Skill 级别策略检查 Hook — BEFORE_SKILL_EXEC 时执行。

    封装 run_pre_execute_gate 的所有检查逻辑，向 Hook 系统提供统一的
    BEFORE_SKILL_EXEC 事件处理。

    使用方式：
        hook_executor.register(HookEvent.BEFORE_SKILL_EXEC, SkillPolicyHook())

        # SkillGateway 中：
        result = await hook_executor.execute(
            HookEvent.BEFORE_SKILL_EXEC,
            HookPayload(skill=skill, skill_params=params),
        )
        if not result.allowed:
            # 返回 policy 拒绝响应
    """

    async def execute(self, payload: HookPayload) -> HookResult:
        """执行 Skill 级别策略检查。"""
        if payload.event != HookEvent.BEFORE_SKILL_EXEC:
            return HookResult(allowed=True)

        skill = payload.skill
        if skill is None:
            return HookResult(allowed=True)

        params = payload.skill_params or {}

        gate_result = run_pre_execute_gate(skill, params=params)

        if not gate_result.allowed:
            detail = gate_result.detail or {}
            return HookResult(
                allowed=False,
                reason=gate_result.reason,
                error_context={
                    "error_type": detail.get("error_type"),
                    "category": detail.get("category"),
                    "missing_keys": detail.get("missing_keys"),
                    "stage": "pre_execute",
                },
            )

        return HookResult(allowed=True)