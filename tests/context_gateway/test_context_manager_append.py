"""ContextManager.append() 追加消息 + 自动 compress/compact 测试。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch


def test_append_short_messages_no_compress(context_manager):
    """短消息追加不触发 compress，token 计数正确更新。"""
    context_manager.init_budget(80000)

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    context_manager.sync_tokens(msgs)

    new_msgs = [{"role": "assistant", "content": "hi there"}]
    result = asyncio.run(context_manager.append(msgs, new_msgs))

    assert len(result) == 3
    assert result[-1]["content"] == "hi there"
    assert context_manager.total_tokens > 0


def test_append_triggers_compact_when_over_budget(context_manager):
    """总 token 超阈值时触发 compact（compact_trigger = context_max * 0.7）。"""
    context_manager.init_budget(2000)
    context_manager._total_tokens = 1500

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "msg1 " + "x" * 3000},
        {"role": "assistant", "content": "reply1 " + "y" * 3000},
        {"role": "user", "content": "msg2 " + "x" * 3000},
        {"role": "assistant", "content": "reply2 " + "y" * 3000},
        {"role": "user", "content": "msg3 " + "x" * 3000},
        {"role": "assistant", "content": "reply3 " + "y" * 3000},
        {"role": "user", "content": "msg4 " + "x" * 3000},
        {"role": "assistant", "content": "reply4 " + "y" * 3000},
        {"role": "user", "content": "msg5 " + "x" * 3000},
        {"role": "assistant", "content": "reply5 " + "y" * 3000},
    ]

    new_msgs = [{"role": "user", "content": "new msg " + "z" * 3000}]

    with patch(
        "core.context.compaction.chat_completions_async",
        new_callable=AsyncMock,
        return_value="compacted summary of old messages",
    ):
        result = asyncio.run(context_manager.append(msgs, new_msgs))

    assert len(result) < len(msgs) + 1
    summary_msgs = [m for m in result if "历史摘要" in str(m.get("content", ""))]
    assert len(summary_msgs) == 1
