# SPDX-License-Identifier: Apache-2.0
"""shared/fs/monitor - 统一的文件监控接口。

提供 FsMonitorProtocol 的默认实现：
- 基于轮询（os.walk）的工作方式
- 与 ArtifactRegistry 可选集成
- 支持 session/skill agent 级别的隔离
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

from shared.fs.snapshot import FsSnapshotManager, SnapshotConfig
from shared.fs.types import (
    ChangeType,
    DiffResult,
    ExecutionRecord,
    FsChange,
    FsMonitorProtocol,
    IS_POSIX,
    resolve_path_safe,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# FsMonitor - 默认监控实现
# =============================================================================

class FsMonitor:
    """
    统一的文件监控器。

    基于 FsSnapshotManager 的监控实现，提供 before_tool/after_tool 接口。
    支持可选绑定 ArtifactRegistry 实现自动产物注册。

    使用方式：
        # 基本用法
        monitor = FsMonitor(workspace_root=Path("/path/to/workspace"))
        exec_id = monitor.before_tool("bash", {"command": "echo hello"}, [])
        # ... tool executes ...
        record = monitor.after_tool(exec_id, "bash", {"command": "echo hello"}, result)

        # 带自动注册
        monitor = FsMonitor(workspace_root=Path("/path/to/workspace"))
        monitor.bind_artifact_registry(artifact_registry)
        exec_id = monitor.before_tool("bash", {"command": "echo hello"}, [])
        record = monitor.after_tool(exec_id, "bash", {"command": "echo hello"}, result)
        # 产物会自动注册到 artifact_registry
    """

    def __init__(
        self,
        workspace_root: Path,
        config: SnapshotConfig | None = None,
        compute_hash: bool = False,
    ) -> None:
        """
        Args:
            workspace_root: 工作区根目录
            config: 快照配置
            compute_hash: 是否计算文件 hash（默认 False，提高性能）
        """
        self._root = workspace_root.resolve()
        self._config = config or SnapshotConfig(
            workspace_root=self._root,
            compute_hash=compute_hash,
        )
        self._manager = FsSnapshotManager(self._root, self._config)

        # 当前执行上下文
        self._current_execution_id: str | None = None
        self._current_tool_name: str = ""
        self._current_args: dict[str, Any] = {}
        self._input_paths: list[Path] = []

        # 可选的 ArtifactRegistry 绑定
        self._artifact_registry: Any = None

    @property
    def workspace_root(self) -> Path:
        """获取 workspace 根目录。"""
        return self._root

    @property
    def snapshot_manager(self) -> FsSnapshotManager:
        """获取底层快照管理器。"""
        return self._manager

    # =========================================================================
    # ArtifactRegistry 集成
    # =========================================================================

    def bind_artifact_registry(self, registry: Any) -> None:
        """
        绑定 ArtifactRegistry，实现自动产物注册。

        Args:
            registry: ArtifactRegistry 实例
        """
        self._artifact_registry = registry

    def unbind_artifact_registry(self) -> None:
        """解绑 ArtifactRegistry。"""
        self._artifact_registry = None

    # =========================================================================
    # FsMonitorProtocol 实现
    # =========================================================================

    def before_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        input_paths: list[Path],
    ) -> str:
        """
        工具执行前调用，拍摄快照。

        Args:
            tool_name: 工具名称
            args: 工具参数
            input_paths: 输入文件路径

        Returns:
            execution_id 用于关联 after_tool
        """
        execution_id = self._manager.generate_execution_id()
        self._current_execution_id = execution_id
        self._current_tool_name = tool_name
        self._current_args = args or {}
        self._input_paths = input_paths or []

        # 拍摄快照
        self._manager.take_full_snapshot(execution_id)

        return execution_id

    def after_tool(
        self,
        execution_id: str,
        tool_name: str,
        args: dict[str, Any],
        result: Any = None,
    ) -> ExecutionRecord:
        """
        工具执行后调用，对比快照并返回执行记录。

        Args:
            execution_id: before_tool 返回的 ID
            tool_name: 工具名称
            args: 工具参数
            result: 工具执行结果

        Returns:
            ExecutionRecord 包含所有文件变化
        """
        # 拍摄执行后的快照
        after_id = f"{execution_id}_after"
        self._manager.take_full_snapshot(after_id)

        # 对比快照
        before = self._manager.get_snapshot(execution_id)
        after = self._manager.get_snapshot(after_id)

        changes: list[FsChange] = []
        if before is not None and after is not None:
            changes = before.diff(after)

        # 构建执行记录
        record = ExecutionRecord(
            execution_id=execution_id,
            tool_name=tool_name,
            timestamp=0.0,  # 由调用方填充
            input_paths=self._input_paths,
            changes=changes,
            artifact_paths=[],
            temporary_paths=[],
            audit_paths=[],
        )

        # 分类变化
        self._classify_changes(record)

        # 自动注册产物
        if self._artifact_registry is not None:
            self._auto_register_artifacts(record)

        # 清理快照
        self._manager.release(execution_id)
        self._manager.release(after_id)

        # 重置上下文
        self._current_execution_id = None
        self._current_tool_name = ""
        self._current_args = {}
        self._input_paths = []

        return record

    # =========================================================================
#   内部方法
# =========================================================================

    def _classify_changes(self, record: ExecutionRecord) -> None:
        """
        分类文件变化。

        根据工具类型和文件特征，将变化分为：
        - artifact_paths: 最终产物
        - temporary_paths: 临时文件
        - audit_paths: 需审计的文件

        Args:
            record: 执行记录（会被修改）
        """
        # 基础分类：按变化类型
        for change in record.changes:
            if change.change_type == ChangeType.DELETED:
                continue

            # 检查是否为显著变化
            if not change.is_significant:
                continue

            # 检查路径是否在 workspace 内
            if not self._is_safe_path(change.path):
                record.audit_paths.append(change.path)
                continue

            # 检查是否为临时文件
            if self._is_temporary_path(change.path):
                record.temporary_paths.append(change.path)
                continue

            # 默认视为产物
            record.artifact_paths.append(change.path)

    def _is_safe_path(self, path: Path) -> bool:
        """检查路径是否在 workspace 内。"""
        try:
            path.resolve().relative_to(self._root)
            return True
        except ValueError:
            return False

    def _is_temporary_path(self, path: Path) -> bool:
        """检查路径是否为临时文件。"""
        name = path.name.lower()

        # 临时文件模式
        temp_patterns = {
            "tmp", "temp", "cache", ".cache",
            "__pycache__", "__runner__",
            ".pytest_cache", ".tox",
        }

        for pattern in temp_patterns:
            if pattern in name:
                return True

        # __pycache__ 目录下的文件
        try:
            path.relative_to(self._root)
            for parent in path.parents:
                if parent.name == "__pycache__":
                    return True
                if parent.name.startswith(".pytest"):
                    return True
        except ValueError:
            pass

        return False

    def _auto_register_artifacts(self, record: ExecutionRecord) -> None:
        """自动将产物注册到 ArtifactRegistry。"""
        if self._artifact_registry is None:
            return

        for path in record.artifact_paths:
            try:
                # 调用 ArtifactRegistry.register()
                self._artifact_registry.register(
                    path=str(path),
                    tool=record.tool_name,
                    turn=0,  # turn 由调用方填充
                    source="fs_monitor",
                )
            except Exception:
                pass

    # =========================================================================
    # 兼容性别名
    # =========================================================================

    def cleanup(self) -> int:
        """清理资源。"""
        self._manager.clear()
        return 0


# =============================================================================
# SessionFsMonitor - Session 级别的隔离监控器
# =============================================================================

class SessionFsMonitor:
    """
    Session 级别的隔离文件监控器。

    每个 session/skill agent 绑定独立的 FsMonitor 实例，
    提供完整的隔离和生命周期管理。

    使用方式：
        session_monitor = SessionFsMonitor(
            workspace_root=Path("/path/to/workspace"),
            session_id="session-123",
        )
        monitor = session_monitor.get_monitor()
        # 使用 monitor...
        session_monitor.cleanup()
    """

    def __init__(
        self,
        workspace_root: Path,
        session_id: str | None = None,
        config: SnapshotConfig | None = None,
    ) -> None:
        """
        Args:
            workspace_root: 工作区根目录
            session_id: Session ID（自动生成如果未提供）
            config: 快照配置
        """
        self._root = workspace_root.resolve()
        self._session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        self._config = config
        self._monitor: FsMonitor | None = None

    @property
    def session_id(self) -> str:
        """获取 Session ID。"""
        return self._session_id

    @property
    def workspace_root(self) -> Path:
        """获取 workspace 根目录。"""
        return self._root

    def get_monitor(self) -> FsMonitor:
        """获取或创建 FsMonitor 实例。"""
        if self._monitor is None:
            self._monitor = FsMonitor(
                workspace_root=self._root,
                config=self._config,
            )
        return self._monitor

    def bind_artifact_registry(self, registry: Any) -> None:
        """绑定 ArtifactRegistry。"""
        self.get_monitor().bind_artifact_registry(registry)

    def cleanup(self) -> int:
        """清理资源，返回清理的文件数。"""
        if self._monitor is not None:
            return self._monitor.cleanup()
        return 0


# =============================================================================
# 兼容性别名
# =============================================================================

SnapshotManager = FsSnapshotManager
