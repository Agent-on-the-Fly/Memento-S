# SPDX-License-Identifier: Apache-2.0
"""SandboxAuditHook — Sandbox 快照对比审计（BEFORE_TOOL_EXEC + AFTER_TOOL_EXEC）。

可选集成 FsSnapshotManager 以复用 FileChangeHook 的快照，
避免同一工具执行时重复遍历 workspace。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from shared.hooks.executor import HookDefinition
from shared.hooks.types import HookEvent, HookPayload, HookResult
from shared.fs.snapshot import FsSnapshotManager
from shared.fs.types import is_safe_within_workspace
from utils.logger import get_logger

logger = get_logger(__name__)


class SandboxAuditHook(HookDefinition):
    """
    Sandbox 快照对比审计 — BEFORE_TOOL_EXEC + AFTER_TOOL_EXEC

    策略：宽松模式，放行但记录日志
    审计范围：所有可能创建文件的工具（bash, python_repl, file_create, write_file 等）

    工作流程：
    1. BEFORE_TOOL_EXEC 时记录 workspace 快照（优先用 FsSnapshotManager）
    2. AFTER_TOOL_EXEC 时对比快照，检测新增的文件
    3. 如果新增文件在系统目录（workspace 外），记录警告日志但放行

    可选使用 FsSnapshotManager 复用 FileChangeHook 的快照：
        snapshot_mgr = FsSnapshotManager(workspace_root=workspace_root)
        hook = SandboxAuditHook(workspace_root=workspace_root, snapshot_manager=snapshot_mgr)
    """

    FILE_CREATION_TOOLS: frozenset[str] = frozenset({
        "bash", "python_repl", "file_create", "write_file", "edit_file",
        "copy_file", "move_file",
    })

    def __init__(
        self,
        workspace_root: Path,
        snapshot_manager: "FsSnapshotManager | None" = None,
    ):
        """
        Args:
            workspace_root: 工作区根目录。
            snapshot_manager: 可选的 FsSnapshotManager 实例，优先复用其快照。
        """
        super().__init__()
        self._workspace_root = workspace_root.resolve()
        self._snapshot_manager = snapshot_manager
        self._local_snapshots: dict[str, frozenset[str]] = {}

    async def execute(self, payload: HookPayload) -> HookResult:
        tool_name = payload.tool_name

        # 只审计文件创建类工具
        if tool_name not in self.FILE_CREATION_TOOLS:
            return HookResult(allowed=True)

        if payload.event == HookEvent.BEFORE_TOOL_EXEC:
            # BEFORE_TOOL_EXEC：记录快照
            snapshot_key = tool_name
            if self._snapshot_manager is not None:
                execution_id = payload.metadata.get("execution_id", snapshot_key)
                self._snapshot_manager.take_snapshot(execution_id)
            else:
                self._local_snapshots[snapshot_key] = self._take_snapshot()
            return HookResult(allowed=True)

        elif payload.event == HookEvent.AFTER_TOOL_EXEC:
            # AFTER_TOOL_EXEC：对比快照
            snapshot_key = tool_name
            if self._snapshot_manager is not None:
                execution_id = payload.metadata.get("execution_id", snapshot_key)
                diff = self._snapshot_manager.diff(execution_id)
                created_files = diff.created
            else:
                before = self._local_snapshots.get(snapshot_key)
                if before is None:
                    return HookResult(allowed=True)
                after = self._take_snapshot()
                created_files = list(after - before)
                self._local_snapshots.pop(snapshot_key, None)

            for file_path in created_files:
                # 使用跨平台的 workspace 外检测
                if not is_safe_within_workspace(Path(file_path), self._workspace_root):
                    logger.warning(
                        "[SandboxAudit] File created outside workspace by '{}': {}",
                        tool_name, file_path
                    )
                else:
                    logger.debug(
                        "[SandboxAudit] File created in workspace by '{}': {}",
                        tool_name, file_path
                    )

        return HookResult(allowed=True)

    def _take_snapshot(self) -> frozenset[str]:
        """获取当前 workspace 的文件快照。"""
        if not self._workspace_root.exists():
            return frozenset()

        files = []
        try:
            for p in self._workspace_root.rglob("*"):
                if p.is_file():
                    files.append(str(p))
        except PermissionError:
            logger.warning("[SandboxAudit] Cannot access workspace: {}", self._workspace_root)
            return frozenset()

        return frozenset(files)


# 向后兼容别名
SnapshotManager = FsSnapshotManager
