# SPDX-License-Identifier: Apache-2.0
"""共享压缩工具 — microcompact。

从 infra/context/providers/microcompact_impl.py 迁移。
无 core/ 依赖，所有压缩逻辑集中存放。
"""

from __future__ import annotations

from typing import Any

from utils.logger import get_logger

from infra.shared.extract import _build_digest

logger = get_logger(__name__)


CLEARED_MESSAGE: str = "[Old tool result content cleared]"

_SKIP_PREFIXES: tuple[str, ...] = (
    "[Output persisted",
    "[Extracted from",
    "[summarized]",
)


def _compact_content(content: str) -> str:
    """Replace tool result with structured digest if possible, else empty placeholder."""
    try:
        import json
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            output_val = parsed.get("output")
            if isinstance(output_val, dict):
                return _build_digest(parsed, output_val)
    except Exception:
        pass
    return CLEARED_MESSAGE


def _estimate_message_tokens_rough(messages: list[dict[str, Any]]) -> int:
    """粗略估算消息 token 数（不调用 tokenizer）。"""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    total_chars += len(part)
                elif isinstance(part, dict):
                    total_chars += len(str(part.get("text", "") or ""))
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            total_chars += len(func.get("name", ""))
            args = func.get("arguments", "")
            total_chars += len(args) if isinstance(args, str) else len(str(args))
    return int(total_chars / 3)


def microcompact_messages(
    messages: list[dict[str, Any]],
    keep_recent: int,
    compactable_tools: set[str],
) -> tuple[list[dict[str, Any]], int]:
    """清除旧的 compactable tool result 内容。

    跳过已落盘/已提取的 preview 消息（不重复清除）。
    保留最近 keep_recent 个可清除消息不清除。

    Returns:
        (处理后的新 messages 列表, 估计节省的 token 数)
    """
    compactable_indices: list[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        if msg.get("name", "") not in compactable_tools:
            continue

        content = msg.get("content", "")
        if content == CLEARED_MESSAGE:
            continue

        if isinstance(content, str) and any(
            content.startswith(prefix) for prefix in _SKIP_PREFIXES
        ):
            continue

        compactable_indices.append(i)

    if len(compactable_indices) <= keep_recent:
        return messages, 0

    to_clear = set(compactable_indices[:-keep_recent] if keep_recent > 0 else compactable_indices)

    saved_tokens = 0
    result: list[dict[str, Any]] = []

    for i, msg in enumerate(messages):
        if i in to_clear:
            old_tokens = _estimate_message_tokens_rough([msg])
            replacement = _compact_content(msg.get("content", ""))
            new_tokens = _estimate_message_tokens_rough([{**msg, "content": replacement}])
            saved_tokens += max(0, old_tokens - new_tokens)
            result.append({**msg, "content": replacement})
            logger.debug(
                "Microcompact cleared tool result at index {} (tool={})",
                i, msg.get("name", ""),
            )
        else:
            result.append(msg)

    if saved_tokens > 0:
        logger.info(
            "Microcompact cleared {} tool results, freed ~{} tokens",
            len(to_clear), saved_tokens,
        )

    return result, saved_tokens
