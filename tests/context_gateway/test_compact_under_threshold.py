"""Compact 空消息路径: 原样返回。"""
from __future__ import annotations

import asyncio

from core.context.compaction import compact_messages


def test_compact_empty_rest_returns_unchanged():
    """只有 system 消息时不触发 compact。"""
    msgs = [{"role": "system", "content": "sys"}]

    result, total = asyncio.run(compact_messages(
        msgs,
        summary_tokens=2000,
    ))

    assert result == msgs
    assert total > 0
