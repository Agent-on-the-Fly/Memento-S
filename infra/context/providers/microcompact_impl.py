# SPDX-License-Identifier: Apache-2.0
"""Microcompact — 零 LLM 调用的旧 tool result 内容清除。

迁移自 core/context/microcompact.py。
委托 infra/shared/compact.py 处理（唯一真实来源）。
保留此文件作为向后兼容的导入入口。
"""

from __future__ import annotations

from infra.shared.compact import (
    CLEARED_MESSAGE,
    _SKIP_PREFIXES,
    _compact_content,
    microcompact_messages,
)

__all__ = [
    "CLEARED_MESSAGE",
    "_SKIP_PREFIXES",
    "_compact_content",
    "microcompact_messages",
]
