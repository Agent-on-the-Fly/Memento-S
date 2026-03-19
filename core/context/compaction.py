"""消息压缩层 — compress（单条摘要）+ compact（全量合并）。

两个独立操作:
  compress_message  — 单条消息超过 token 阈值时，LLM 摘要为更短的等价消息
  compact_messages  — 总 token 超阈值时，全量合并非 system 消息为一条摘要

由 ContextManager.append() 在追加消息时自动调用，外部不需要直接使用。
"""
from __future__ import annotations

from typing import Any

from utils.logger import get_logger
from utils.token_utils import count_tokens, count_tokens_messages
from middleware.llm.utils import chat_completions_async
from core.prompts.templates import SUMMARIZE_CONVERSATION_PROMPT

logger = get_logger(__name__)


async def compress_message(
    msg: dict[str, Any],
    max_msg_tokens: int,
    summary_tokens: int = 800,
) -> dict[str, Any]:
    """当单条消息 token 数超过 max_msg_tokens 时，用 LLM 压缩。

    Returns:
        压缩后的消息（role 不变），或原消息（未超限时）。
    """
    content = msg.get("content", "")
    if not isinstance(content, str) or not content:
        return msg

    tokens = count_tokens(content)
    if tokens <= max_msg_tokens:
        return msg

    role = msg.get("role", "user")
    logger.info("Compress: role={} tokens={} > {}", role, tokens, max_msg_tokens)

    try:
        summary = await chat_completions_async(
            system=(
                "You are a precise summarizer. Compress the following message "
                "while preserving all key facts, data and intent. "
                "Return ONLY the compressed text."
            ),
            messages=[{"role": "user", "content": content}],
            max_tokens=summary_tokens,
        )
        result = dict(msg)
        result["content"] = f"[compressed]\n{summary.strip()}"
        logger.info("Compress: {} -> {} tokens", tokens, count_tokens(result["content"]))
        return result
    except Exception:
        logger.warning("compress_message failed, keeping original", exc_info=True)
        return msg


async def compact_messages(
    messages: list[dict[str, Any]],
    summary_tokens: int = 2000,
    scratchpad_path: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """全量压缩：保留 system message，其余全部合并为一条摘要。

    Args:
        scratchpad_path: 归档文件路径，会写入摘要 hint 供 agent 回查。

    Returns:
        (compacted_messages, new_total_tokens)
    """
    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    start = 1 if system_msg else 0
    rest = messages[start:]

    if not rest:
        return messages, count_tokens_messages(messages)

    context_parts: list[str] = []
    for msg in rest:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        tag = "TOOL_RESULT" if role == "tool" else role
        context_parts.append(f"[{tag}]: {content}")

    full_context = "\n".join(context_parts)
    prompt = SUMMARIZE_CONVERSATION_PROMPT.format(
        max_tokens=summary_tokens,
        context=full_context,
    )

    try:
        summary = await chat_completions_async(
            system=(
                "You are a precise summarizer. Return only the essential information. "
                "CRITICAL: Preserve [TOOL_RESULT] content as completely as possible."
            ),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=summary_tokens,
        )
        hint = "[历史摘要"
        if scratchpad_path:
            hint += f" — 完整记录已归档: {scratchpad_path}"
        hint += "]"
        summary_msg: dict[str, Any] = {
            "role": "system",
            "content": f"{hint}\n{summary.strip()}",
        }
        result = ([system_msg] if system_msg else []) + [summary_msg]
        new_total = count_tokens_messages(result)
        logger.info(
            "Compact: {} -> {} msgs, tokens -> {}",
            len(messages), len(result), new_total,
        )
        return result, new_total
    except Exception:
        logger.warning("compact_messages failed, keeping original", exc_info=True)
        return messages, count_tokens_messages(messages)
