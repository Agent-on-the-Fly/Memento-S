# SPDX-License-Identifier: Apache-2.0
"""Configuration models for file change detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Platform = Literal["windows", "linux", "darwin"]

# Default ignored directory names (platform-agnostic)
DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset({
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".idea",
    ".vscode",
    ".cursor",
    ".codex",
})


@dataclass
class LifecyclePolicy:
    """Policy for file lifecycle management.

    Determines what happens to files based on their lifecycle classification.
    """

    # 是否自动清理临时文件
    auto_cleanup_temporary: bool = True

    # 清理前是否备份
    backup_before_cleanup: bool = False

    # 备份目录（None 表示不备份）
    backup_dir: Path | None = None

    # 未知文件保留的最大数量（防止误报堆积）
    max_unknown_files: int = 50

    # 未知文件保留的最大总大小（MB）
    max_unknown_size_mb: int = 100

    # 显著变化阈值（字节），小于此值的变化被视为微小变化
    significant_size_threshold: int = 10 * 1024

    # 是否在显著变化时自动注册到 ArtifactRegistry
    auto_register_artifacts: bool = True


@dataclass
class DetectionConfig:
    """Configuration for file change detection."""

    # 工作区根目录
    workspace_root: Path

    # 是否启用文件变化检测
    enabled: bool = True

    # 是否启用生命周期分类
    enable_lifecycle_classification: bool = True

    # 快照是否包含 SHA256 哈希（小文件）
    compute_hash_for_small_files: bool = True

    # 小文件阈值（字节），小于此值的文件计算 SHA256
    small_file_threshold: int = 1024 * 1024  # 1MB

    # 忽略的目录名称
    ignore_dirs: frozenset[str] = field(default_factory=lambda: DEFAULT_IGNORE_DIRS)

    # 忽略的文件扩展名
    ignore_extensions: frozenset[str] = field(default_factory=lambda: frozenset({
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".class",
        ".o",
        ".obj",
        ".swp",
        ".swo",
        ".bak",
        ".tmp",
        ".log",
    }))

    # 忽略的文件名模式（正则表达式）
    ignore_patterns: tuple[str, ...] = ("__pycache__", ".DS_Store", "Thumbs.db")

    # 生命周期策略
    lifecycle_policy: LifecyclePolicy = field(default_factory=LifecyclePolicy)

    # 每个 execution_id 的最大快照数量
    max_snapshots_per_execution: int = 2  # before + after

    # 是否记录执行历史（用于调试）
    record_history: bool = False

    def is_ignored(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        # Check by directory name
        for part in path.parts:
            if part in self.ignore_dirs or part.startswith("."):
                return True

        # Check by extension
        if path.suffix.lower() in self.ignore_extensions:
            return True

        return False
