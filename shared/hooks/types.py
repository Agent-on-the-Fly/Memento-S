# SPDX-License-Identifier: Apache-2.0
"""Hook 类型定义 — 生命周期事件枚举和 Payload/Result 结构。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.skill.execution.tool_context import RuntimeToolContext
    from core.skill.schema import Skill


class HookEvent(str, Enum):
    """SkillAgent/SkillDispatcher 层可拦截的生命周期事件。"""

    #: adapter 工具执行前
    BEFORE_TOOL_EXEC = "before_tool_exec"
    #: adapter 工具执行后（无论成功或失败）
    AFTER_TOOL_EXEC = "after_tool_exec"
    #: skill 执行前（gateway.execute 前）
    BEFORE_SKILL_EXEC = "before_skill_exec"
    #: skill 执行后（gateway.execute 返回后）
    AFTER_SKILL_EXEC = "after_skill_exec"
    #: 循环检测到时
    ON_LOOP_DETECTED = "on_loop_detected"


@dataclass(frozen=True, slots=True)
class HookPayload:
    """传递给 hook 的事件负载。"""

    event: HookEvent

    #: 工具名（tool_exec 事件）
    tool_name: str = ""
    #: 工具参数（tool_exec 事件）
    args: dict[str, Any] = None

    #: 运行时上下文（tool_exec 事件）
    context: "RuntimeToolContext | None" = None

    #: Skill 实例（skill_exec 事件）
    skill: "Skill | None" = None
    #: Skill 参数（skill_exec 事件）
    skill_params: dict[str, Any] = None

    #: 执行结果字符串（after_tool_exec / after_skill_exec）
    result: str | None = None
    #: 错误详情字典（after_tool_exec / after_skill_exec，失败时有值）
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.args is None:
            object.__setattr__(self, "args", {})


@dataclass
class HookResult:
    """Hook 执行结果。

    allowed=False 表示阻止操作继续。

    扩展字段：
    - error_context: 错误上下文（由 PolicyGateHook 返回）
    - detected_artifacts: 检测到的产物路径列表（由 FileChangeHook 返回）
    - deferred_messages: 延迟注入的 system 消息（由 ReactSupervisionHook 返回）
    - recovery_action: 恢复建议（由 ReactSupervisionHook 返回）
    """

    #: True = 允许继续，False = 阻止操作
    allowed: bool = True
    #: 阻止原因（当 allowed=False 时）
    reason: str = ""
    #: 修改后的参数（当 allowed=True 且需要修改参数时）
    modified_args: dict[str, Any] | None = None

    # ── 扩展字段 ────────────────────────────────────────────────────────

    #: 错误上下文字典（由 PolicyGateHook 返回，包含 error_type / category 等）
    error_context: dict[str, Any] | None = None

    #: 检测到的产物文件路径列表（由 FileChangeHook 返回，用于统一注册）
    detected_artifacts: list[str] | None = None

    #: 延迟注入的 system 消息列表（由各 SupervisionHook 返回，供 SkillAgent 注入 prompt）
    deferred_messages: list[dict[str, Any]] | None = None

    #: 恢复动作建议（由各 SupervisionHook 返回）
    #: 值：RECOMMEND_ABORT / RECOMMEND_RETRY / CONTINUE
    recovery_action: str | None = None

    #: 本次工具执行产生的文件系统变化（由 FileChangeHook 在 AFTER_TOOL_EXEC 时填充）
    #: 结构：{"created": [...], "modified": [...], "deleted": [...]}
    #: agent.py 用此字段作为 state_delta 的首选来源，不再依赖 regex 解析
    fs_changes: dict[str, list[str]] | None = None

    #: 扩展元数据（由各 Hook 自由填充）
    metadata: dict[str, Any] | None = None
