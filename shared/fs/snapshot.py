# SPDX-License-Identifier: Apache-2.0
"""shared/fs/snapshot - 统一的快照管理器。

整合以下三套快照实现：
1. SnapshotManager (core/skill/execution/hooks/snapshot_manager.py)
   - 简单高效，仅记录 path: size
2. FileChangeDetector (core/skill/execution/detection/file_change_detector.py)
   - 功能完整，支持 hash 和 mtime
3. SandboxArtifactCollector (middleware/sandbox/artifacts.py)
   - 简单快照，用于沙箱产物收集

设计特点：
- 支持按需计算 hash（compute_hash 选项）
- 统一的遍历方式（os.walk）
- 支持 ignore_dirs 和 ignore_patterns
- 支持 execution_id 绑定
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.fs.types import (
    ChangeType,
    DiffResult,
    ExecutionRecord,
    FsChange,
    FsSnapshot,
    FsSnapshotEntry,
    IS_POSIX,
    DEFAULT_IGNORE_DIRS,
    DEFAULT_IGNORE_PATTERNS,
    resolve_path_safe,
)


class FsChangeCollection(list[FsChange]):
    """List of changes with the legacy ``.changes`` access path."""

    @property
    def changes(self) -> "FsChangeCollection":
        return self


# =============================================================================
# 配置
# =============================================================================

@dataclass
class SnapshotConfig:
    """快照配置。"""
    # 是否计算 hash（计算 hash 会增加 I/O）
    compute_hash: bool = False
    # 小文件阈值（小于此大小才计算 hash）
    small_file_threshold: int = 1024 * 1024  # 1MB
    # 忽略的目录
    ignore_dirs: frozenset[str] = DEFAULT_IGNORE_DIRS
    # 忽略的文件模式
    ignore_patterns: frozenset[str] = DEFAULT_IGNORE_PATTERNS
    # workspace 根目录
    workspace_root: Path | None = None

    def is_ignored_dir(self, dirname: str) -> bool:
        """检查目录是否应被忽略。"""
        return dirname.startswith(".") or dirname in self.ignore_dirs

    def is_ignored_file(self, filename: str) -> bool:
        """检查文件是否应被忽略。"""
        import fnmatch
        return any(fnmatch.fnmatch(filename, p) for p in self.ignore_patterns)


# =============================================================================
# FsSnapshotManager - 统一快照管理器
# =============================================================================

class FsSnapshotManager:
    """
    统一的文件系统快照管理器。

    整合了三套快照系统的功能：
    - 快速快照（仅 size，用于快速对比）
    - 完整快照（包含 mtime/hash，用于精确变化检测）
    - 沙箱快照（简单实现，用于产物收集）

    使用方式：
        manager = FsSnapshotManager(workspace_root=Path("/path/to/workspace"))

        # 方式 1: 快速快照对比
        manager.take_snapshot("exec-123")
        # ... tool executes ...
        diff = manager.diff("exec-123")

        # 方式 2: 完整快照（带 hash）
        config = SnapshotConfig(compute_hash=True)
        manager2 = FsSnapshotManager(workspace_root=Path("/path/to/workspace"), config=config)
        snapshot1 = manager2.take_full_snapshot("exec-456")
        # ... tool executes ...
        snapshot2 = manager2.take_full_snapshot("exec-456")
        changes = snapshot1.diff(snapshot2)

        # 方式 3: 沙箱产物收集
        pre = SandboxSnapshot.take(workspace_root)
        # ... sandbox executes ...
        artifacts = SandboxSnapshot.collect_diff(pre, workspace_root)
    """

    def __init__(
        self,
        workspace_root: Path,
        config: SnapshotConfig | None = None,
    ) -> None:
        """
        Args:
            workspace_root: 工作区根目录
            config: 快照配置
        """
        self._root = workspace_root.resolve()
        self._config = config or SnapshotConfig(workspace_root=self._root)
        if self._config.workspace_root is None:
            self._config.workspace_root = self._root

        # 快照存储：execution_id -> FsSnapshot
        self._snapshots: dict[str, FsSnapshot] = {}

    @property
    def workspace_root(self) -> Path:
        """获取 workspace 根目录。"""
        return self._root

    # =========================================================================
    # 快速快照 API（来自 SnapshotManager）
    # =========================================================================

    def take_snapshot(self, execution_id: str) -> FsSnapshot:
        """
        拍摄快速快照（仅记录 path: size）。

        如果已存在同名快照，返回已有的。

        Args:
            execution_id: 执行 ID

        Returns:
            FsSnapshot 实例
        """
        if execution_id in self._snapshots:
            return self._snapshots[execution_id]

        snapshot = FsSnapshot(
            timestamp=time.time(),
            workspace_root=self._root,
            execution_id=execution_id,
        )

        self._walk_and_snapshot(snapshot, compute_hash=False)
        self._snapshots[execution_id] = snapshot
        return snapshot

    def diff(self, execution_id: str) -> DiffResult:
        """
        对比当前工作区与快照的差异（快速版本）。

        Args:
            execution_id: 快照对应的执行 ID

        Returns:
            DiffResult 实例
        """
        snapshot = self._snapshots.get(execution_id)
        if snapshot is None:
            return DiffResult()

        current_files: dict[str, int] = {}
        current_dirs: dict[str, frozenset[str]] = {}

        try:
            for dirpath, dirnames, filenames in os.walk(self._root):
                dirpath = Path(dirpath)
                # 过滤忽略目录
                dirnames[:] = [d for d in dirnames if not self._config.is_ignored_dir(d)]

                for filename in filenames:
                    try:
                        fpath = dirpath / filename
                        rel = str(fpath.relative_to(self._root))
                        current_files[rel] = fpath.stat().st_size
                    except OSError:
                        pass

                # 记录子目录
                for dirname in dirnames:
                    try:
                        dpath = dirpath / dirname
                        rel = str(dpath.relative_to(self._root))
                        sub = frozenset(
                            str(p.relative_to(dpath)) for p in dpath.iterdir() if p.is_dir()
                        )
                        current_dirs[rel] = sub
                    except OSError:
                        pass

        except OSError:
            pass

        snap_files = snapshot.entries
        current_files_rels = {rel: size for rel, entry in snapshot.entries.items() for size in [entry.size]}

        # 重新组织 snapshot.files 格式
        snap_files_dict: dict[str, int] = {}
        for rel, entry in snapshot.entries.items():
            snap_files_dict[rel] = entry.size

        created = [f for f in current_files if f not in snap_files_dict]
        deleted = [f for f in snap_files_dict if f not in current_files]
        modified = [
            f for f in current_files
            if f in snap_files_dict and current_files[f] != snap_files_dict[f]
        ]
        new_dirs = [d for d in current_dirs if d not in snapshot.entries]

        return DiffResult(
            created=created,
            modified=modified,
            deleted=deleted,
            new_dirs=new_dirs,
        )

    # =========================================================================
    # 完整快照 API（来自 FileChangeDetector）
    # =========================================================================

    def take_full_snapshot(self, execution_id: str) -> FsSnapshot:
        """
        拍摄完整快照（包含 mtime 和可选 hash）。

        与 take_snapshot 不同，此方法会记录每个文件的详细信息。

        Args:
            execution_id: 执行 ID

        Returns:
            FsSnapshot 实例
        """
        if execution_id in self._snapshots:
            return self._snapshots[execution_id]

        snapshot = FsSnapshot(
            timestamp=time.time(),
            workspace_root=self._root,
            execution_id=execution_id,
        )

        self._walk_and_snapshot(snapshot, compute_hash=self._config.compute_hash)
        self._snapshots[execution_id] = snapshot
        return snapshot

    def compare(
        self,
        before_id: str,
        after_id: str,
    ) -> FsChangeCollection:
        """
        对比两个快照，返回详细变化列表。

        Args:
            before_id: 执行前的快照 ID
            after_id: 执行后的快照 ID

        Returns:
            FsChange 列表
        """
        before = self._snapshots.get(before_id)
        after = self._snapshots.get(after_id)

        if before is None or after is None:
            return FsChangeCollection()

        return FsChangeCollection(before.diff(after))

    def get_snapshot(self, execution_id: str) -> FsSnapshot | None:
        """获取之前拍摄的快照。"""
        return self._snapshots.get(execution_id)

    # =========================================================================
    # 工具方法
    # =========================================================================

    def release(self, execution_id: str) -> None:
        """释放指定 execution_id 的快照，释放内存。"""
        self._snapshots.pop(execution_id, None)

    def clear_snapshot(self, execution_id: str) -> None:
        """Backward-compatible alias for releasing one named snapshot."""
        self.release(execution_id)

    def clear(self) -> None:
        """清空所有快照。"""
        self._snapshots.clear()

    def generate_execution_id(self) -> str:
        """生成唯一的执行 ID。"""
        return f"fs_{uuid.uuid4().hex[:12]}"

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _walk_and_snapshot(
        self,
        snapshot: FsSnapshot,
        compute_hash: bool,
    ) -> None:
        """遍历 workspace 并填充快照。"""
        try:
            for dirpath, dirnames, filenames in os.walk(self._root):
                dirpath = Path(dirpath)
                # 过滤忽略目录（原地修改）
                dirnames[:] = [
                    d for d in dirnames
                    if not self._config.is_ignored_dir(d)
                ]

                for filename in filenames:
                    # 过滤忽略文件
                    if self._config.is_ignored_file(filename):
                        continue

                    try:
                        fpath = dirpath / filename
                        stat = fpath.stat()
                        rel = str(fpath.relative_to(self._root))

                        # 按需计算 hash
                        hash_sha256 = None
                        if compute_hash and stat.st_size < self._config.small_file_threshold:
                            hash_sha256 = self._compute_sha256(fpath)

                        snapshot.entries[rel] = FsSnapshotEntry(
                            path=fpath,
                            mtime=stat.st_mtime,
                            size=stat.st_size,
                            hash_sha256=hash_sha256,
                        )
                    except OSError:
                        pass

        except OSError:
            pass

    @staticmethod
    def _compute_sha256(filepath: Path) -> str | None:
        """计算文件的 SHA256 hash。"""
        try:
            h = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except (OSError, PermissionError):
            return None


# =============================================================================
# SandboxSnapshot - 沙箱产物快照（来自 middleware/sandbox/artifacts.py）
# =============================================================================

class SandboxSnapshot:
    """
    简化的沙箱产物快照。

    提供静态方法用于快速拍摄快照和收集产物差异。
    用于沙箱执行环境的产物检测。
    """

    @staticmethod
    def take(directory: Path) -> dict[str, dict[str, Any]]:
        """
        拍摄目录的简单快照。

        Args:
            directory: 要快照的目录

        Returns:
            快照字典 {rel_path: {mtime, size}}
        """
        if not directory.exists():
            return {}

        snapshot: dict[str, dict[str, Any]] = {}
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    snapshot[str(file_path.relative_to(directory))] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                    }
        except PermissionError:
            pass

        return snapshot

    @staticmethod
    def collect_diff(
        pre_snapshot: dict[str, dict[str, Any]],
        work_dir: Path,
    ) -> list[str]:
        """
        收集快照差异，返回新增或修改的文件路径。

        Args:
            pre_snapshot: 执行前的快照
            work_dir: 工作目录

        Returns:
            新增或修改的文件路径列表
        """
        if not work_dir.exists():
            return []

        artifacts: list[str] = []
        current = SandboxSnapshot.take(work_dir)

        for rel_path, current_info in current.items():
            if rel_path not in pre_snapshot:
                # 新增文件
                artifacts.append(str(work_dir / rel_path))
            elif current_info["mtime"] != pre_snapshot[rel_path]["mtime"]:
                # 修改文件
                artifacts.append(str(work_dir / rel_path))

        return artifacts


# =============================================================================
# 兼容性别名
# =============================================================================

DirectorySnapshot = FsSnapshot
