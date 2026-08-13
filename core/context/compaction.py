# SPDX-License-Identifier: Apache-2.0
"""Small compatibility compaction helpers for callers of the legacy API."""

from __future__ import annotations

from typing import Any

from middleware.llm import chat_completions_async
from utils.token_utils import count_tokens, count_tokens_messages


async def compress_message(
    message: dict[str, Any],
    *,
    max_msg_tokens: int = 3000,
    summary_tokens: int = 500,
) -> dict[str, Any]:
    content = message.get("content", "")
    if not isinstance(content, str) or not content:
        return message
    if count_tokens(content) <= max_msg_tokens:
        return message
    try:
        summary = await chat_completions_async(
            messages=[
                {"role": "system", "content": "Compress the message faithfully."},
                {"role": "user", "content": content},
            ],
            max_tokens=summary_tokens,
        )
    except Exception:  # noqa: BLE001 - compaction must preserve the original on any provider failure
        return message
    result = dict(message)
    result["content"] = f"[compressed]\n{summary}"
    return result


async def compact_messages(
    messages: list[dict[str, Any]], *, summary_tokens: int = 2000
) -> tuple[list[dict[str, Any]], int]:
    if len(messages) <= 1:
        return messages, count_tokens_messages(messages)
    system = messages[0] if messages[0].get("role") == "system" else None
    rest = messages[1:] if system is not None else messages
    if not rest:
        return messages, count_tokens_messages(messages)
    transcript = "\n\n".join(
        f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in rest
    )
    try:
        summary = await chat_completions_async(
            messages=[
                {"role": "system", "content": "Summarize the conversation history."},
                {"role": "user", "content": transcript},
            ],
            max_tokens=summary_tokens,
        )
    except Exception:  # noqa: BLE001 - compaction must preserve the original on any provider failure
        return messages, count_tokens_messages(messages)
    compacted: list[dict[str, Any]] = []
    if system is not None:
        compacted.append(system)
    compacted.append(
        {"role": "system", "content": f"[历史摘要]\n{summary}"}
    )
    return compacted, count_tokens_messages(compacted)
