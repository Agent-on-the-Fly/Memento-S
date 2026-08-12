# SPDX-License-Identifier: Apache-2.0
"""全局 Hook 注册中心 — 单例模式，所有层共享。

所有 Hook 在系统初始化时注册一次，各层按事件类型自动过滤：

  - BEFORE_TOOL_EXEC  → ToolArgsValidationHook, PolicyGateHook, PathPolicyHook
  - AFTER_TOOL_EXEC   → ToolResultSupervisionHook, ErrorPatternSupervisionHook
  - BEFORE_SKILL_EXEC → SkillPolicyHook
  - ON_LOOP_DETECTED  → LoopTelemetryHook（遥测用）

注意：LoopSupervisionHook 和 SaturationSupervisionHook 已移至 Per-execution 层
（每次 SkillAgent.run() 时动态注册），确保 LoopDetector / InfoSaturationDetector
在每次 skill 执行时重置状态，避免跨 skill 的状态污染。

使用方式：
    from core.skill.execution.hooks.registry import (
        global_skill_agent_hook_executor,
        register_skill_agent_hooks,
    )

    # 初始化时注册（通常在 bootstrap.init_skill_system() 中调用一次）
    register_skill_agent_hooks(workspace_root=Path("..."), policy_manager=...)

    # 各层获取同一实例，按事件类型自动过滤
    executor = global_skill_agent_hook_executor()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.hooks.executor import HookExecutor
from shared.hooks.types import HookEvent

if TYPE_CHECKING:
    from shared.security.policy import PolicyManager

# 全局单例（模块级延迟初始化）
_global_skill_agent_hook_executor: HookExecutor | None = None

# 全局配置（在 register_skill_agent_hooks 时传入）
_global_workspace_root: Path | None = None
_global_policy_manager: Any = None


def global_skill_agent_hook_executor() -> HookExecutor:
    """获取全局 HookExecutor 实例（延迟初始化）。

    在调用 register_skill_agent_hooks() 之前返回空实例（无任何 hook 注册）。
    """
    global _global_skill_agent_hook_executor
    if _global_skill_agent_hook_executor is None:
        _global_skill_agent_hook_executor = HookExecutor()
    return _global_skill_agent_hook_executor


def is_global_hooks_registered() -> bool:
    """检查全局 Hook 是否已注册（防止重复注册）。"""
    global _global_policy_manager
    return _global_policy_manager is not None


def register_skill_agent_hooks(
    workspace_root: Path | None = None,
    policy_manager: "PolicyManager | None" = None,
) -> None:
    """在系统初始化时注册所有 SkillAgent 层全局 Hook。

    推荐在 bootstrap.init_skill_system() 时调用一次。

    Args:
        workspace_root: 工作区根目录，FileChangeHook 需要。
        policy_manager: PolicyManager 实例，PolicyGateHook 需要。
    """
    global _global_workspace_root, _global_policy_manager

    if is_global_hooks_registered():
        return

    _global_workspace_root = workspace_root
    _global_policy_manager = policy_manager

    executor = global_skill_agent_hook_executor()

    # ── BEFORE_TOOL_EXEC ─────────────────────────────────────────────────

    # 0. 工具参数处理 Hook（从 ToolArgsProcessor 迁移）
    from core.skill.execution.hooks.tool_args_hook import ToolArgsValidationHook

    executor.register(HookEvent.BEFORE_TOOL_EXEC, ToolArgsValidationHook())

    # 1. Policy Gate Hook（替代 SkillToolAdapter 内的 ToolGate）
    from core.skill.execution.hooks.policy_gate import PolicyGateHook

    executor.register(HookEvent.BEFORE_TOOL_EXEC, PolicyGateHook(policy_manager=policy_manager))

    # 2. 路径策略 Hook
    from core.skill.execution.hooks.path_policy import PathPolicyHook

    executor.register(HookEvent.BEFORE_TOOL_EXEC, PathPolicyHook())

    # ── AFTER_TOOL_EXEC ──────────────────────────────────────────────────

    # 注意：LoopSupervisionHook 和 SaturationSupervisionHook 已从全局永久注册移除
    # （见上方 "注意" 说明）。它们的 Per-execution 注册在 agent.py 中进行。

    # 2. 工具结果处理 Hook（从 ToolResultProcessor 迁移）
    from core.skill.execution.hooks.tool_result_supervision import ToolResultSupervisionHook

    executor.register(HookEvent.AFTER_TOOL_EXEC, ToolResultSupervisionHook())

    # 3. Error Pattern 监督 Hook（StatefulErrorPatternDetector 封装 — 无状态，可保留全局）
    from core.skill.execution.hooks.error_pattern_supervision import ErrorPatternSupervisionHook

    executor.register(HookEvent.AFTER_TOOL_EXEC, ErrorPatternSupervisionHook())

    # 4. Skill 级别策略检查 Hook
    from core.skill.execution.hooks.skill_policy import SkillPolicyHook

    executor.register(HookEvent.BEFORE_SKILL_EXEC, SkillPolicyHook())

    # ── ON_LOOP_DETECTED ─────────────────────────────────────────────────

    # 7. Loop 遥测 Hook（遥测用，pass-through）
    from core.skill.execution.hooks.loop_telemetry import LoopTelemetryHook

    executor.register(HookEvent.ON_LOOP_DETECTED, LoopTelemetryHook())

    # ── 注意 ─────────────────────────────────────────────────────────────
    # 以下两个 Hook 已从全局永久注册中移除，
    # 改为在 SkillAgent.run() 中每次执行时动态注册（per-execution 层），
    # 确保 LoopDetector / InfoSaturationDetector 在每次 skill 执行时重置状态，
    # 避免跨 skill 的状态污染：
    #   - LoopSupervisionHook
    #   - SaturationSupervisionHook
    #
    # FileChangeHook 需要 workspace_root，在每次 SkillAgent.run() 时动态注册


def get_workspace_root() -> Path | None:
    """获取注册时的工作区根目录。"""
    return _global_workspace_root


def get_policy_manager() -> Any:
    """获取注册时的 PolicyManager 实例。"""
    return _global_policy_manager


# ── 向后兼容别名 ────────────────────────────────────────────────────────────

# 保留旧名称以兼容已有调用方
get_global_hook_executor = global_skill_agent_hook_executor
register_global_hooks = register_skill_agent_hooks


# ── 统一导出 ────────────────────────────────────────────────────────────────

from shared.hooks.types import HookEvent, HookPayload, HookResult

__all__ = [
    # 新名称优先
    "global_skill_agent_hook_executor",
    "register_skill_agent_hooks",
    "is_global_hooks_registered",
    "get_workspace_root",
    "get_policy_manager",
    "HookEvent",
    "HookPayload",
    "HookResult",
    # 向后兼容别名
    "get_global_hook_executor",
    "register_global_hooks",
]
