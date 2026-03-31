"""Compact 超阈值路径: LLM 全量合并消息。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from core.context.compaction import compact_messages


def test_compact_over_threshold_generates_summary():
    """超阈值时触发 LLM 摘要，保留 system + 摘要。"""
    msgs = [{"role": "system", "content": "system prompt"}]
    for i in range(10):
        msgs.append({"role": "user", "content": f"msg {i} " + "x" * 5000})
        msgs.append({"role": "assistant", "content": f"reply {i} " + "y" * 5000})

    mock_summary = "Summary: user asked about X, agent did Y."

    with patch(
        "core.context.compaction.chat_completions_async",
        new_callable=AsyncMock,
        return_value=mock_summary,
    ):
        result, new_total = asyncio.run(compact_messages(
            msgs,
            summary_tokens=2000,
        ))

    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "system prompt"

    summary_msgs = [m for m in result if "历史摘要" in str(m.get("content", ""))]
    assert len(summary_msgs) == 1
    assert mock_summary in summary_msgs[0]["content"]
    assert new_total > 0
