# SPDX-License-Identifier: Apache-2.0
"""智能内容提取 — 分层提取策略。

Layer 0: 结构化提取 (零 LLM)
Layer 1: LLM 总结 (按需)
Layer 2: 截断 (fallback)

委托 infra/shared/extract.py 的 smart_extract_content 处理（唯一真实来源）。
"""

from __future__ import annotations

from infra.shared.extract import smart_extract_content as _smart_extract_content

# 向后兼容的别名
from infra.shared.extract import (
    _extract_structured,
    extract_key_content,
    estimate_tokens_fast,
)

__all__ = [
    "smart_extract_content",
    "_extract_structured",
    "extract_key_content",
    "estimate_tokens_fast",
]


async def smart_extract_content(
    content: str,
    budget_tokens: int,
    *,
    model: str = "",
    tool_name: str = "",
) -> str:
    """分层内容提取。委托 shared/extract.py 处理。"""
    from middleware.llm.llm_client import chat_completions_async

    return await _smart_extract_content(
        content,
        budget_tokens,
        model=model,
        tool_name=tool_name,
        llm_client=lambda msgs, **kw: chat_completions_async(**kw, messages=msgs),
    )
