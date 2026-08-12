# SPDX-License-Identifier: Apache-2.0
"""HookExecutor — 通用生命周期 hook 执行器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from shared.hooks.types import HookEvent, HookPayload, HookResult

if TYPE_CHECKING:
    from core.skill.execution.tool_context import RuntimeToolContext


class HookDefinition(ABC):
    """Hook 定义基类 — 支持不同类型的 hook 实现。"""

    def __init__(self) -> None:
        # 共享上下文：同一 execute() 调用内所有 hook 共享同一个字典。
        # 由 HookExecutor 在每次 execute() 前注入。
        self.hook_context: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, payload: HookPayload) -> HookResult:
        """执行 hook，返回 allowed/reason。"""
        ...


class CommandHook(HookDefinition):
    """
    命令式 hook — 同步/异步函数作为 hook。

    Usage:
        def my_hook(payload: HookPayload) -> HookResult:
            if payload.tool_name == "rm":
                return HookResult(allowed=False, reason="rm is disabled")
            return HookResult(allowed=True)

        executor.register(HookEvent.BEFORE_TOOL_EXEC, CommandHook(my_hook))
    """

    def __init__(self, fn: Any) -> None:
        super().__init__()
        self._fn = fn

    async def execute(self, payload: HookPayload) -> HookResult:
        result = self._fn(payload)
        if hasattr(result, "__await__"):
            return await result
        if result is None:
            return HookResult(allowed=True, reason="CommandHook fn returned None")
        return result


class HookExecutor:
    """
    通用生命周期 hook 执行器。

    在每个生命周期事件触发时，依次执行所有已注册的 hook。
    任何 hook 返回 allowed=False，剩余 hook 仍会执行（忽略其结果），
    但操作本身被阻止。

    共享上下文机制：
    - 所有同一事件内的 hook 共享同一个 hook_context 字典。
    - FileChangeHook 在 AFTER_TOOL_EXEC 时写入 fs_changes。
    - LoopSupervisionHook 在 AFTER_TOOL_EXEC 时读取 fs_changes，
      用真实文件系统变化覆盖 record() 的 category 判断。
    - hook_context 在每次 execute() 调用时重置，确保状态隔离。

    Usage:
        executor = HookExecutor()
        executor.register(HookEvent.BEFORE_TOOL_EXEC, CommandHook(my_validator))

        result = await executor.execute(HookEvent.BEFORE_TOOL_EXEC, payload)
        if not result.allowed:
            raise PermissionError(result.reason)
    """

    def __init__(
        self,
        hooks: dict[HookEvent, list[HookDefinition]] | None = None,
    ) -> None:
        self._hooks: dict[HookEvent, list[HookDefinition]] = hooks or {}
        # 共享上下文：在同一 execute() 调用内的所有 hook 之间共享。
        # 结构：dict[str, Any]，键名由各 hook 约定。
        # 生命周期：每次 execute() 时清空并重建。
        self._hook_context: dict[str, Any] = {}

    def register(
        self,
        event: HookEvent,
        definition: HookDefinition,
    ) -> None:
        """注册一个 hook 到指定事件。"""
        self._hooks.setdefault(event, []).append(definition)

    def unregister(
        self,
        event: HookEvent,
        definition: HookDefinition,
    ) -> bool:
        """取消注册一个 hook。返回是否找到并移除。"""
        hooks = self._hooks.get(event, [])
        try:
            hooks.remove(definition)
            return True
        except ValueError:
            return False

    def clear(self, event: HookEvent | None = None) -> None:
        """清除指定事件的 hook（或所有 hook）。"""
        if event is None:
            self._hooks.clear()
        elif event in self._hooks:
            self._hooks[event].clear()

    async def execute(
        self,
        event: HookEvent,
        payload: HookPayload,
    ) -> HookResult:
        """
        执行所有注册到该事件的 hook。

        聚合返回：
        - 任一 hook allowed=False → blocked_reason 为第一个阻止的原因，
          剩余 hook 仍执行
        - 所有 hook 的 detected_artifacts → 合并去重
        - 所有 hook 的 deferred_messages → 合并
        - 所有 hook 的 recovery_action → 优先级 RECOMMEND_ABORT > RECOMMEND_RETRY > CONTINUE > None
        """
        blocked_result: HookResult | None = None

        # 聚合字段
        all_detected_artifacts: list[str] = []
        all_deferred_messages: list[dict[str, Any]] = []
        recovery_priority: int = 99  # 越小优先级越高
        final_recovery_action: str | None = None
        # 合并 fs_changes：只合并 created/modified，deleted 不参与 stall 检测
        merged_fs_changes: dict[str, list[str]] = {"created": [], "modified": [], "deleted": []}
        merged_metadata: dict[str, Any] = {}

        RECOVERY_PRIORITY: dict[str, int] = {
            "RECOMMEND_ABORT": 0,
            "RECOMMEND_RETRY": 1,
            "CONTINUE": 2,
        }

        # 每次 execute() 开始时重置共享上下文，确保状态隔离
        self._hook_context.clear()

        for hook_def in self._hooks.get(event, []):
            # 注入共享上下文：同一事件内所有 hook 共享同一个字典实例
            hook_def.hook_context = self._hook_context

            try:
                result = await hook_def.execute(payload)
            except Exception as exc:
                result = HookResult(allowed=True, reason=f"hook error: {exc}")

            # 防御：某些 CommandHook 包装的同步函数可能静默返回 None
            if result is None:
                result = HookResult(allowed=True, reason="hook returned None")

            # 阻止逻辑（保留第一个阻止的）
            if not result.allowed and blocked_result is None:
                blocked_result = result

            # 聚合 detected_artifacts
            if result.detected_artifacts:
                for path in result.detected_artifacts:
                    if path not in all_detected_artifacts:
                        all_detected_artifacts.append(path)

            # 聚合 deferred_messages
            if result.deferred_messages:
                all_deferred_messages.extend(result.deferred_messages)

            # 聚合 recovery_action（优先级最高者）
            if result.recovery_action:
                p = RECOVERY_PRIORITY.get(result.recovery_action, 99)
                if p < recovery_priority:
                    recovery_priority = p
                    final_recovery_action = result.recovery_action

            # 聚合 fs_changes（只合并 created/modified）
            if result.fs_changes:
                for path in result.fs_changes.get("created", []):
                    if path not in merged_fs_changes["created"]:
                        merged_fs_changes["created"].append(path)
                for path in result.fs_changes.get("modified", []):
                    if path not in merged_fs_changes["modified"]:
                        merged_fs_changes["modified"].append(path)
                for path in result.fs_changes.get("deleted", []):
                    if path not in merged_fs_changes["deleted"]:
                        merged_fs_changes["deleted"].append(path)

            # 合并 metadata：按注册顺序，后面的 hook 覆盖前面的同名 key
            if result.metadata:
                merged_metadata.update(result.metadata)

        # 构建最终结果
        final_fs_changes: dict[str, list[str]] | None = None
        if merged_fs_changes["created"] or merged_fs_changes["modified"]:
            final_fs_changes = merged_fs_changes

        final = HookResult(
            allowed=(blocked_result is None),
            reason=blocked_result.reason if blocked_result else "",
            detected_artifacts=all_detected_artifacts or None,
            deferred_messages=all_deferred_messages or None,
            recovery_action=final_recovery_action,
            fs_changes=final_fs_changes,
            metadata=merged_metadata if merged_metadata is not None else None,
        )

        return final
