"""compress_message 单条消息 LLM 摘要测试。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from core.context.compaction import compress_message


def test_compress_short_message_unchanged():
    """短消息不触发 compress，原样返回。"""
    msg = {"role": "user", "content": "hello world"}
    result = asyncio.run(compress_message(msg, max_msg_tokens=3000))
    assert result == msg


def test_compress_long_message_triggers_llm():
    """超长消息触发 LLM 摘要。"""
    long_content = "x " * 5000
    msg = {"role": "assistant", "content": long_content}

    with patch(
        "core.context.compaction.chat_completions_async",
        new_callable=AsyncMock,
        return_value="compressed summary",
    ):
        result = asyncio.run(compress_message(msg, max_msg_tokens=500, summary_tokens=200))

    assert "[compressed]" in result["content"]
    assert "compressed summary" in result["content"]
    assert result["role"] == "assistant"


def test_compress_empty_content_unchanged():
    """空内容消息不触发 compress。"""
    msg = {"role": "user", "content": ""}
    result = asyncio.run(compress_message(msg, max_msg_tokens=100))
    assert result == msg


def test_compress_llm_failure_returns_original():
    """LLM 调用失败时返回原消息。"""
    long_content = "x " * 5000
    msg = {"role": "user", "content": long_content}

    with patch(
        "core.context.compaction.chat_completions_async",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM unavailable"),
    ):
        result = asyncio.run(compress_message(msg, max_msg_tokens=500))

    assert result == msg
