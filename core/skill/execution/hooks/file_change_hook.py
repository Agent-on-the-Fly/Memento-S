# SPDX-License-Identifier: Apache-2.0
"""FileChangeHook — 集成 ExecutionFileTracker 的文件变化检测 Hook。

在 BEFORE_TOOL_EXEC 时拍快照，在 AFTER_TOOL_EXEC 时对比、分类、决策。

设计决策：
- 仅检测显著变化（文件创建、修改、删除）
- 不阻止任何操作（审计模式）
- 通过 HookResult.detected_artifacts 返回检测到的产物路径，
  由 SkillAgent 层统一注册（保持 turn 正确性）
- 补充检测：从不返回文件路径的工具输出文本中提取文件路径
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.hooks.executor import HookDefinition
from shared.hooks.types import HookEvent, HookPayload, HookResult
from core.skill.execution.detection.config import DetectionConfig, LifecyclePolicy
from core.skill.execution.detection.execution_tracker import ExecutionFileTracker
from utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# 需要检测的"文件创建类"工具
FILE_CREATION_TOOLS: frozenset[str] = frozenset({
    "bash",
    "python_repl",
    "js_repl",
    "file_create",
    "write_file",
    "edit_file",
    "edit_file_by_lines",
    "copy_file",
    "move_file",
})

# 从工具输出文本中提取文件路径的正则模式
_RESULT_FILE_PATH_PATTERNS = [
    # 匹配 / 开头且长度大于 5 的路径（排除 URLs）
    re.compile(r"(?<![a-zA-Z0-9_/])((?:/[a-zA-Z0-9_\-.]){3,})"),
    # 匹配 output/path/file/dest 标记的值
    re.compile(r"(?:output|path|file|dest)\s*[:=]\s*'?([^\s'\";]+)'?", re.IGNORECASE),
]


def _looks_like_path(p: Path) -> bool:
    """判断字符串是否看起来像文件路径。"""
    s = str(p)
    if s.startswith("/") or s.startswith("\\"):
        return True
    if "." in s.split("/")[-1] or "/" in s or "\\" in s:
        return True
    return False


def _extract_paths_from_text(text: str) -> set[Path]:
    """从文本中提取文件路径。"""
    paths: set[Path] = set()
    for pattern in _RESULT_FILE_PATH_PATTERNS:
        for match in pattern.finditer(text):
            path_str = match.group(1).strip()
            if path_str and len(path_str) > 3:
                try:
                    paths.add(Path(path_str))
                except Exception:
                    pass
    return paths


def _extract_paths_from_dict(data: dict) -> set[Path]:
    """从字典中递归提取文件路径。"""
    paths: set[Path] = set()
    path_keys = {
        "path", "file", "filepath", "file_path",
        "output", "output_path", "destination",
        "result", "result_path", "created", "saved",
        "uri", "url",
    }
    for key, value in data.items():
        if key.lower() in path_keys and isinstance(value, str):
            try:
                p = Path(value)
                if _looks_like_path(p):
                    paths.add(p)
            except Exception:
                pass
        # 递归搜索
        if isinstance(value, str):
            paths.update(_extract_paths_from_text(value))
        elif isinstance(value, dict):
            paths.update(_extract_paths_from_dict(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    paths.update(_extract_paths_from_dict(item))
    return paths


def extract_result_file_paths(result: Any, workspace_root: Path) -> list[Path]:
    """
    从工具执行结果中提取文件路径（补充快照检测遗漏的路径）。

    支持纯文本、JSON 结构和 list of content。
    只返回 workspace 内存在的文件。
    """
    if result is None:
        return []

    paths: set[Path] = set()

    if isinstance(result, str):
        paths.update(_extract_paths_from_text(result))
    elif isinstance(result, dict):
        paths.update(_extract_paths_from_dict(result))
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, str):
                paths.update(_extract_paths_from_text(item))
            elif isinstance(item, dict):
                paths.update(_extract_paths_from_dict(item))

    # 过滤：只保留 workspace 内的文件
    valid_paths = []
    ws_str = str(workspace_root.resolve())
    for p in paths:
        try:
            resolved = p.resolve()
            if str(resolved).startswith(ws_str) and resolved.is_file():
                valid_paths.append(resolved)
        except Exception:
            pass

    return valid_paths


class FileChangeHook(HookDefinition):
    """
    文件变化检测 Hook — 集成 ExecutionFileTracker。

    功能：
    1. BEFORE_TOOL_EXEC：拍摄 workspace 快照
    2. AFTER_TOOL_EXEC：
       - 对比快照，检测变化
       - 分类生命周期（临时文件 / 产物 / 未知）
       - 执行策略（清理 / 注册 / 保留）
       - 通过 HookResult.detected_artifacts 返回检测到的产物路径

    产物不再直接注册，而是由 SkillAgent 层统一注册（保持 turn 正确性）。

    使用方式：
        hook_executor.register(HookEvent.BEFORE_TOOL_EXEC, file_change_hook)
        hook_executor.register(HookEvent.AFTER_TOOL_EXEC, file_change_hook)
    """

    def __init__(
        self,
        workspace_root: Path,
        config: DetectionConfig | None = None,
        policy: LifecyclePolicy | None = None,
        enabled: bool = True,
    ):
        """
        初始化 FileChangeHook。

        Args:
            workspace_root: 工作区根目录。
            config: 检测配置（可选）。
            policy: 生命周期策略（可选）。
            enabled: 是否启用（禁用时跳过检测）。
        """
        super().__init__()
        self._workspace_root = workspace_root.resolve()
        self._enabled = enabled

        if self._enabled:
            self._tracker = ExecutionFileTracker(
                workspace_root=self._workspace_root,
                config=config,
                policy=policy,
            )
        else:
            self._tracker = None

        logger.info(
            "[FileChangeHook] Initialized for workspace: {}, enabled={}",
            self._workspace_root, enabled
        )

    @property
    def tracker(self) -> ExecutionFileTracker | None:
        """获取 ExecutionFileTracker 实例。"""
        return self._tracker

    def bind_artifact_registry(self, registry: Any) -> None:
        """
        绑定 ArtifactRegistry（保留接口签名，向后兼容）。

        注意：产物注册已改为通过 HookResult.detected_artifacts 返回，
        由 SkillAgent 层统一注册，不再直接调用 registry。
        """
        logger.debug(
            "[FileChangeHook] ArtifactRegistry binding (legacy): {}",
            type(registry).__name__ if registry else None
        )

    async def execute(self, payload: HookPayload) -> HookResult:
        """
        执行 hook。

        根据事件类型执行不同操作：
        - BEFORE_TOOL_EXEC: 拍快照
        - AFTER_TOOL_EXEC: 对比、分类、决策
        """
        if not self._enabled or self._tracker is None:
            return HookResult(allowed=True)

        tool_name = payload.tool_name

        # 仅对文件创建类工具进行检测
        if tool_name not in FILE_CREATION_TOOLS:
            return HookResult(allowed=True)

        if payload.event == HookEvent.BEFORE_TOOL_EXEC:
            return await self._handle_before_exec(payload)

        elif payload.event == HookEvent.AFTER_TOOL_EXEC:
            return await self._handle_after_exec(payload)

        return HookResult(allowed=True)

    async def _handle_before_exec(
        self,
        payload: HookPayload,
    ) -> HookResult:
        """BEFORE_TOOL_EXEC：拍摄快照，记录输入文件。"""
        tool_name = payload.tool_name
        args = payload.args or {}

        # 提取输入文件路径
        input_paths = self._extract_input_paths(tool_name, args)

        try:
            execution_id = await self._tracker.before_execute(
                tool_name=tool_name,
                args=args,
                input_paths=input_paths,
            )

            logger.debug(
                "[FileChangeHook] BEFORE_EXEC: tool={}, execution_id={}",
                tool_name, execution_id
            )

        except Exception as e:
            logger.warning(
                "[FileChangeHook] before_execute failed: {}",
                e
            )

        return HookResult(allowed=True)

    async def _handle_after_exec(
        self,
        payload: HookPayload,
    ) -> HookResult:
        """AFTER_TOOL_EXEC：对比快照，分类，决策。产物通过 detected_artifacts 返回。"""
        tool_name = payload.tool_name
        args = payload.args or {}
        result = payload.result

        if not self._tracker._execution_stack:
            # This should NOT happen after the adapter.py fix: SkillToolAdapter no longer
            # fires AFTER_TOOL_EXEC for retry shim calls (_skip_hooks=True).
            # If seen, it means a direct adapter call without an agent wrapper,
            # or an unmatched AFTER call.
            logger.warning(
                "[FileChangeHook] AFTER_EXEC without before execution "
                "(direct adapter call or unmatched AFTER), skipping"
            )
            return HookResult(allowed=True)

        detected_paths: list[str] = []
        temporary_paths: list[str] = []
        audit_paths: list[str] = []

        try:
            # Do NOT pass execution_id — let ExecutionFileTracker.after_execute()
            # auto-pop from its own _execution_stack. This is the sole owner of the stack.
            record = await self._tracker.after_execute(
                tool_name=tool_name,
                args=args,
                result=result,
            )

            # 补充检测：从结果文本中提取文件路径（快照 diff 遗漏的情况）
            extracted_paths = extract_result_file_paths(result, self._workspace_root)
            existing_paths = {c.path for c in (record.change_set.changes if record.change_set else [])}
            for path in extracted_paths:
                if path not in existing_paths:
                    logger.debug(
                        "[FileChangeHook] Extracted file path from result (not in snapshot diff): {}",
                        path,
                    )
                    # 补充到 detected_paths，让产物能被注册
                    detected_paths.append(str(path))

            # ── 构建 fs_changes（供 SkillAgent 直接作为 state_delta 来源）────────
            # 不依赖 regex 解析，直接用快照 diff 结果
            fs_changes: dict[str, list[str]] = {"created": [], "modified": [], "deleted": []}
            if record.change_set and record.change_set.changes:
                from shared.fs.types import ChangeType
                for change in record.change_set.changes:
                    change_str = str(change.path)
                    if change.change_type == ChangeType.CREATED:
                        fs_changes["created"].append(change_str)
                    elif change.change_type == ChangeType.MODIFIED:
                        fs_changes["modified"].append(change_str)
                    elif change.change_type == ChangeType.DELETED:
                        fs_changes["deleted"].append(change_str)

            # 通过 HookResult.detected_artifacts 返回产物路径
            # 由 SkillAgent 层统一注册（保持 turn 正确性）
            if record.artifact_paths:
                for path in record.artifact_paths:
                    path_str = str(path)
                    if path_str not in detected_paths:
                        detected_paths.append(path_str)

            if record.temporary_paths:
                for path in record.temporary_paths:
                    temporary_paths.append(str(path))

            if record.audit_paths:
                for path in record.audit_paths:
                    audit_paths.append(str(path))

            # 记录日志摘要
            self._log_execution_summary(record)

        except Exception as e:
            logger.warning(
                "[FileChangeHook] after_execute failed: {}",
                e
            )
            return HookResult(allowed=True)

        # 写入共享上下文：供后续执行的 LoopSupervisionHook 读取。
        self.hook_context["fs_changes"] = (
            fs_changes if (fs_changes["created"] or fs_changes["modified"]) else None
        )

        return HookResult(
            allowed=True,
            detected_artifacts=detected_paths if detected_paths else None,
            fs_changes=fs_changes if (fs_changes["created"] or fs_changes["modified"]) else None,
            metadata={
                "temporary_paths": temporary_paths if temporary_paths else None,
                "audit_paths": audit_paths if audit_paths else None,
                "tool_name": tool_name,
            },
        )

    def _extract_input_paths(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> list[Path]:
        """从工具参数中提取输入文件路径。"""
        input_paths: list[Path] = []
        path_keys = {
            "path", "file", "target", "source", "src",
            "input", "input_path", "file_path", "target_path",
            "source_path",
        }

        for key, value in args.items():
            if key.lower() in path_keys and isinstance(value, str):
                try:
                    p = Path(value)
                    if p.exists() and p.is_file():
                        input_paths.append(p)
                except Exception:
                    pass

        return input_paths

    def _log_execution_summary(
        self,
        record: "ExecutionRecord",  # noqa: F821
    ) -> None:
        """记录执行摘要日志。"""
        changes = record.change_set
        if not changes:
            return

        logger.info(
            "[FileChangeHook] Execution complete: tool={}, "
            "changes={}, artifacts={}, temporary={}, audit={}",
            record.tool_name,
            len(changes.changes),
            len(record.artifact_paths),
            len(record.temporary_paths),
            len(record.audit_paths),
        )

        # 详细日志：每个变化
        for change in changes.changes:
            logger.debug(
                "[FileChangeHook] Change: type={}, path={}, size={}",
                change.change_type.value,
                change.path,
                change.size_bytes,
            )

    def get_tracker(self) -> ExecutionFileTracker | None:
        """获取 ExecutionFileTracker（向后兼容）。"""
        return self._tracker

    def get_last_record(self) -> "ExecutionRecord | None":  # noqa: F821
        """获取最近一次执行的记录。"""
        if self._tracker:
            history = self._tracker.get_history()
            if history:
                return history[-1]
        return None

    def cleanup_temporary(self) -> int:
        """清理所有临时文件。"""
        if self._tracker:
            return self._tracker.cleanup_temporary()
        return 0
