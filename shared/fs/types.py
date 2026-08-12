# SPDX-License-Identifier: Apache-2.0
"""shared/fs/types - 统一的文件系统监控类型定义。

整合并统一以下来源的类型：
- SnapshotManager.DirectorySnapshot / DiffResult
- FileChangeDetector.FileSnapshotEntry / FileSnapshot / FileChange / ChangeType / ChangeSet
- ExecutionRecord

跨平台设计：
- 使用 pathlib.Path 处理所有路径（自动适配 Windows/Unix）
- 禁止使用字符串拼接路径
- workspace 外检测与平台无关
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    pass


# =============================================================================
# 跨平台常量
# =============================================================================

# 统一忽略目录（整合三套快照系统的忽略规则）
DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset({
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    ".pytest_cache",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    ".hypothesis",
    ".eggs",
    "*.egg-info",
    ".venv",
    "venv",
    "env",
    ".env",
})

# 统一忽略文件模式
DEFAULT_IGNORE_PATTERNS: frozenset[str] = frozenset({
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
    "Thumbs.db",
    "*.swp",
    "*.swo",
    "*~",
    ".coverage",
})

# 平台检测
IS_WINDOWS = os.name == "nt"
IS_POSIX = os.name == "posix"


# =============================================================================
# 跨平台工具函数
# =============================================================================

def is_safe_within_workspace(path: Path, workspace: Path) -> bool:
    """跨平台：检查路径是否在 workspace 内。

    使用 Path.resolve() 进行标准化，然后尝试 relative_to()。
    与平台无关，在 Windows/Linux/macOS 上行为一致。

    Args:
        path: 要检查的路径
        workspace: workspace 根目录

    Returns:
        True 如果 path 在 workspace 内或其子目录中
    """
    try:
        path.resolve().relative_to(workspace.resolve())
        return True
    except ValueError:
        return False


def paths_equal(p1: Path, p2: Path) -> bool:
    """跨平台：比较两个路径是否相等（resolve 后）。

    Args:
        p1: 第一个路径
        p2: 第二个路径

    Returns:
        True 如果两个路径指向同一文件/目录
    """
    return p1.resolve() == p2.resolve()


def resolve_path_safe(path: Path) -> Path | None:
    """安全地 resolve 路径，失败时返回 None。

    Args:
        path: 要 resolve 的路径

    Returns:
        resolve 后的路径，失败时返回 None
    """
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return None


# =============================================================================
# ChangeType 枚举
# =============================================================================

class ChangeType(Enum):
    """文件变化类型枚举。"""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


# =============================================================================
# FsSnapshotEntry - 单个文件的快照条目
# =============================================================================

@dataclass
class FsSnapshotEntry:
    """快照中的单个文件条目。

    整合了 FileChangeDetector.FileSnapshotEntry 的功能。
    支持按需计算 hash（仅小文件）。
    """
    path: Path
    mtime: float
    size: int
    hash_sha256: str | None = None  # None 表示未计算（适用于大文件）

    def is_same_content(self, other: FsSnapshotEntry) -> bool:
        """检查内容是否相同（优先使用 hash，其次 mtime）。"""
        if self.size != other.size:
            return False
        if self.hash_sha256 is not None and other.hash_sha256 is not None:
            return self.hash_sha256 == other.hash_sha256
        # Fallback: mtime 比较（精确到毫秒）
        return abs(self.mtime - other.mtime) < 0.001

    def compute_hash(self, chunk_size: int = 8192) -> str | None:
        """计算文件 hash（如果尚未计算）。"""
        if self.hash_sha256 is not None:
            return self.hash_sha256

        if not self.path.exists():
            return None

        try:
            h = hashlib.sha256()
            with open(self.path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    h.update(chunk)
            self.hash_sha256 = h.hexdigest()
            return self.hash_sha256
        except (OSError, PermissionError):
            return None

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "path": str(self.path),
            "mtime": self.mtime,
            "size": self.size,
            "hash": self.hash_sha256,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FsSnapshotEntry:
        """从字典反序列化。"""
        return cls(
            path=Path(d["path"]),
            mtime=d["mtime"],
            size=d["size"],
            hash_sha256=d.get("hash"),
        )


# =============================================================================
# FsSnapshot - 快照容器
# =============================================================================

@dataclass
class FsSnapshot:
    """目录树的快照。

    整合了 FileChangeDetector.FileSnapshot 的功能。
    使用相对路径存储，节省内存且与 workspace 无关。
    """
    timestamp: float
    workspace_root: Path
    entries: dict[str, FsSnapshotEntry] = field(default_factory=dict)
    # 执行 ID（用于关联 before/after 快照对）
    execution_id: str = ""

    def get(self, rel_path: str) -> FsSnapshotEntry | None:
        """通过相对路径获取条目。"""
        return self.entries.get(rel_path)

    def diff(self, other: FsSnapshot) -> list[FsChange]:
        """与另一个快照对比，返回变化列表（after - before）。

        Args:
            other: 之后的快照

        Returns:
            FsChange 列表
        """
        changes: list[FsChange] = []

        # 遍历新快照，找创建和修改
        for rel_path, entry in other.entries.items():
            if rel_path not in self.entries:
                changes.append(FsChange(
                    path=entry.path,
                    rel_path=rel_path,
                    change_type=ChangeType.CREATED,
                    size_bytes=entry.size,
                    size_delta=None,
                    is_significant=entry.size > 0,
                    before_entry=None,
                    after_entry=entry,
                ))
            else:
                before_entry = self.entries[rel_path]
                if not before_entry.is_same_content(entry):
                    changes.append(FsChange(
                        path=entry.path,
                        rel_path=rel_path,
                        change_type=ChangeType.MODIFIED,
                        size_bytes=entry.size,
                        size_delta=entry.size - before_entry.size,
                        is_significant=True,
                        before_entry=before_entry,
                        after_entry=entry,
                    ))

        # 遍历旧快照，找删除
        for rel_path, entry in self.entries.items():
            if rel_path not in other.entries:
                changes.append(FsChange(
                    path=entry.path,
                    rel_path=rel_path,
                    change_type=ChangeType.DELETED,
                    size_bytes=0,
                    size_delta=-entry.size,
                    is_significant=entry.size > 0,
                    before_entry=entry,
                    after_entry=None,
                ))

        return changes

    def get_created(self) -> list[FsChange]:
        """获取所有创建的变化。"""
        return [c for c in self.entries.values() if c.path.exists()] if hasattr(self, 'entries') else []

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "timestamp": self.timestamp,
            "workspace_root": str(self.workspace_root),
            "execution_id": self.execution_id,
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }


# =============================================================================
# FsChange - 单个文件变化
# =============================================================================

@dataclass
class FsChange:
    """表示单个文件的变化。

    整合了 FileChangeDetector.FileChange 的功能。
    """
    path: Path
    rel_path: str
    change_type: ChangeType
    size_bytes: int
    size_delta: int | None
    is_significant: bool
    before_entry: FsSnapshotEntry | None = None
    after_entry: FsSnapshotEntry | None = None

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "path": str(self.path),
            "rel_path": self.rel_path,
            "change_type": self.change_type.value,
            "size_bytes": self.size_bytes,
            "size_delta": self.size_delta,
            "is_significant": self.is_significant,
            "before": self.before_entry.to_dict() if self.before_entry else None,
            "after": self.after_entry.to_dict() if self.after_entry else None,
        }


# =============================================================================
# DiffResult - 快照对比结果
# =============================================================================

@dataclass
class DiffResult:
    """快照对比结果（来自 SnapshotManager.DiffResult）。

    提供简化的变化摘要，适合快速检查。
    """
    created: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    new_dirs: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """是否有任何变化。"""
        return bool(self.created or self.modified or self.deleted or self.new_dirs)

    @property
    def artifact_paths(self) -> list[str]:
        """获取创建和修改的文件路径（排除目录）。"""
        return self.created + self.modified

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "created": self.created,
            "modified": self.modified,
            "deleted": self.deleted,
            "new_dirs": self.new_dirs,
            "has_changes": self.has_changes,
        }


# =============================================================================
# ExecutionRecord - 执行记录
# =============================================================================

@dataclass
class ExecutionRecord:
    """单次工具执行的记录（来自 execution_tracker.ExecutionRecord）。

    聚合了文件变化、策略决策和产物信息。
    """
    execution_id: str
    tool_name: str
    timestamp: float
    input_paths: list[Path] = field(default_factory=list)
    # 变化集合
    changes: list[FsChange] = field(default_factory=list)
    # 产物路径（被识别为最终产物的文件）
    artifact_paths: list[Path] = field(default_factory=list)
    # 临时文件路径（执行产生的临时文件）
    temporary_paths: list[Path] = field(default_factory=list)
    # 审计路径（需要审计的文件）
    audit_paths: list[Path] = field(default_factory=list)
    # 错误信息
    error: str | None = None

    @property
    def created_paths(self) -> list[Path]:
        """获取创建的文件路径。"""
        return [c.path for c in self.changes if c.change_type == ChangeType.CREATED]

    @property
    def modified_paths(self) -> list[Path]:
        """获取修改的文件路径。"""
        return [c.path for c in self.changes if c.change_type == ChangeType.MODIFIED]

    @property
    def deleted_paths(self) -> list[Path]:
        """获取删除的文件路径。"""
        return [c.path for c in self.changes if c.change_type == ChangeType.DELETED]

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "execution_id": self.execution_id,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            "input_paths": [str(p) for p in self.input_paths],
            "artifact_paths": [str(p) for p in self.artifact_paths],
            "temporary_paths": [str(p) for p in self.temporary_paths],
            "audit_paths": [str(p) for p in self.audit_paths],
            "error": self.error,
            "change_count": len(self.changes),
            "created_count": len(self.created_paths),
            "modified_count": len(self.modified_paths),
            "deleted_count": len(self.deleted_paths),
        }


# =============================================================================
# FsMonitorProtocol - 监控协议
# =============================================================================

@runtime_checkable
class FsMonitorProtocol(Protocol):
    """文件监控接口协议。

    定义文件监控的标准接口，支持多种实现：
    - 轮询实现（默认，基于 os.walk）
    - fsnotify 实现（Linux）
    - FSEvents 实现（macOS）
    - ReadDirectoryChangesW 实现（Windows）
    """

    def before_tool(self, tool_name: str, args: dict[str, Any], input_paths: list[Path]) -> str:
        """工具执行前调用，返回 execution_id。

        Args:
            tool_name: 工具名称
            args: 工具参数
            input_paths: 输入文件路径

        Returns:
            execution_id 用于关联 before/after 快照
        """
        ...

    def after_tool(
        self,
        execution_id: str,
        tool_name: str,
        args: dict[str, Any],
        result: Any = None,
    ) -> ExecutionRecord:
        """工具执行后调用，返回执行记录。

        Args:
            execution_id: before_tool 返回的 ID
            tool_name: 工具名称
            args: 工具参数
            result: 工具执行结果

        Returns:
            ExecutionRecord 包含所有文件变化和分类
        """
        ...

    def cleanup(self) -> int:
        """清理资源，返回清理的文件数。"""
        ...
