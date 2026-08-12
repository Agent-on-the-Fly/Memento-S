# SPDX-License-Identifier: Apache-2.0
"""Shared types for file change detection."""

from __future__ import annotations

from enum import Enum


class FileLifecycle(Enum):
    """Lifecycle classification of a file."""
    # 临时文件，执行后应自动清理
    TEMPORARY = "temporary"
    # 最终产物，应保留并注册
    ARTIFACT = "artifact"
    # 未知，需要进一步分析或由 Agent 决定
    UNKNOWN = "unknown"
