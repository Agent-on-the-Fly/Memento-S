# SPDX-License-Identifier: Apache-2.0
"""Hook 生命周期系统 — 细粒度事件钩子。

HookExecutor 作为 SkillAgent/SkillDispatcher 层的细粒度生命周期控制，
与上层 Runner Emitter 组成粗细两层。

分层架构（两类注册来源）：
  - 全局永久层（registry.py）：无状态 Hook，在系统初始化时注册一次
    ToolArgsValidationHook, PolicyGateHook, PathPolicyHook,
    ToolResultSupervisionHook, ErrorPatternSupervisionHook,
    SkillPolicyHook, LoopTelemetryHook
    - Per-execution 层（agent.py）：状态性 Hook，在每次 SkillAgent.run() 时动态注册
    LoopSupervisionHook, SaturationSupervisionHook, StallSupervisionHook,
    FileChangeHook, SandboxAuditHook
    → 每次 skill 执行时创建新实例，确保状态重置，避免 cross-skill 污染

文件结构：
  shared/hooks/types.py   — HookEvent 枚举、HookPayload、HookResult
  shared/hooks/executor.py — HookExecutor 实现
  registry.py       — 全局单例 HookExecutor 注册中心（无状态 Hook）
  loop_supervision.py  — LoopSupervisionHook（Per-execution，含 result-aware 重复检测）
  saturation_supervision.py — SaturationSupervisionHook（Per-execution）
  stall_supervision.py — StallSupervisionHook（Per-execution，多维进展检测）
  error_pattern_supervision.py — ErrorPatternSupervisionHook（全局）
  path_policy.py   — PathPolicyHook（全局）
  sandbox_audit.py — SandboxAuditHook（Per-execution）
  file_change_hook.py — FileChangeHook（Per-execution）
  loop_telemetry.py   — LoopTelemetryHook（全局）
  tool_args_hook.py   — ToolArgsValidationHook（全局）
  tool_result_supervision.py — ToolResultSupervisionHook（全局）
  __init__.py      — 统一导出
"""

from shared.hooks.types import HookEvent, HookPayload, HookResult
from shared.hooks.executor import HookExecutor, HookDefinition
from core.skill.execution.hooks.policy_gate import PolicyGateHook
from core.skill.execution.hooks.path_policy import PathPolicyHook
from core.skill.execution.hooks.sandbox_audit import SandboxAuditHook
from core.skill.execution.hooks.file_change_hook import FileChangeHook
from core.skill.execution.hooks.loop_supervision import LoopSupervisionHook
from core.skill.execution.hooks.saturation_supervision import SaturationSupervisionHook
from core.skill.execution.hooks.stall_supervision import StallSupervisionHook
from core.skill.execution.hooks.error_pattern_supervision import ErrorPatternSupervisionHook
from core.skill.execution.hooks.loop_telemetry import LoopTelemetryHook
from shared.fs.snapshot import FsSnapshotManager as SnapshotManager
from core.skill.execution.hooks.skill_policy import SkillPolicyHook
from core.skill.execution.hooks.tool_args_hook import ToolArgsValidationHook
from core.skill.execution.hooks.tool_result_supervision import ToolResultSupervisionHook
from core.skill.execution.hooks import registry

# 新名称优先
global_skill_agent_hook_executor = registry.global_skill_agent_hook_executor
register_skill_agent_hooks = registry.register_skill_agent_hooks
is_global_hooks_registered = registry.is_global_hooks_registered
get_workspace_root = registry.get_workspace_root
get_policy_manager = registry.get_policy_manager

# 向后兼容别名
get_global_hook_executor = registry.get_global_hook_executor
register_global_hooks = registry.register_global_hooks

__all__ = [
    # 核心类型
    "HookEvent",
    "HookPayload",
    "HookResult",
    "HookDefinition",
    "HookExecutor",
    # Hook 实现类
    "PathPolicyHook",
    "SandboxAuditHook",
    "FileChangeHook",
    "PolicyGateHook",
    "LoopSupervisionHook",
    "SaturationSupervisionHook",
    "StallSupervisionHook",
    "ErrorPatternSupervisionHook",
    "LoopTelemetryHook",
    "SnapshotManager",
    "SkillPolicyHook",
    "ToolArgsValidationHook",
    "ToolResultSupervisionHook",
    # 全局单例（新名称）
    "global_skill_agent_hook_executor",
    "register_skill_agent_hooks",
    "is_global_hooks_registered",
    "get_workspace_root",
    "get_policy_manager",
    # 向后兼容别名
    "get_global_hook_executor",
    "register_global_hooks",
]
