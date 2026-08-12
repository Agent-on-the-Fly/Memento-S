# SPDX-License-Identifier: Apache-2.0
"""infra/shared — infra 内部共享工具模块。

无 core/ 依赖，所有压缩/提取逻辑集中存放。
从 infra/compact/strategies/zero_llm.py 和
infra/context/providers/microcompact_impl.py 合并而来。
"""

from infra.shared.compact import (
    CLEARED_MESSAGE,
    _SKIP_PREFIXES,
    _compact_content,
    microcompact_messages,
)
from infra.shared.extract import smart_extract_content

__all__ = [
    "CLEARED_MESSAGE",
    "_SKIP_PREFIXES",
    "_compact_content",
    "microcompact_messages",
    "smart_extract_content",
]
