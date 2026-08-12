# SPDX-License-Identifier: Apache-2.0
"""shared/fs - 统一的文件系统监控抽象层。

整合三套快照系统：
- SnapshotManager (core/skill/execution/hooks/snapshot_manager.py)
- FileChangeDetector (core/skill/execution/detection/file_change_detector.py)
- SandboxArtifactCollector (middleware/sandbox/artifacts.py)

支持跨平台（Windows/Linux/macOS），提供统一的快照、变化检测和监控接口。
"""

from shared.fs.types import (
    # 类型定义
    ChangeType,
    FsSnapshotEntry,
    FsSnapshot,
    FsChange,
    DiffResult,
    ExecutionRecord,
    # 工具函数
    is_safe_within_workspace,
    paths_equal,
    resolve_path_safe,
    DEFAULT_IGNORE_DIRS,
    DEFAULT_IGNORE_PATTERNS,
)
from shared.fs.snapshot import FsSnapshotManager
from shared.fs.monitor import FsMonitor, FsMonitorProtocol

__all__ = [
    # 类型
    "ChangeType",
    "FsSnapshotEntry",
    "FsSnapshot",
    "FsChange",
    "DiffResult",
    "ExecutionRecord",
    # 管理器
    "FsSnapshotManager",
    # 监控接口
    "FsMonitor",
    "FsMonitorProtocol",
    # 工具函数
    "is_safe_within_workspace",
    "paths_equal",
    "resolve_path_safe",
    "DEFAULT_IGNORE_DIRS",
    "DEFAULT_IGNORE_PATTERNS",
]
