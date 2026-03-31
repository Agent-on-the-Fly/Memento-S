from __future__ import annotations

import json

from core.context.manager import ContextManager


def test_persist_tool_result_returns_inline(context_manager: ContextManager):
    """persist_tool_result returns full result inline without writing to disk."""
    result = json.dumps({
        "ok": True,
        "results": [{"tool": "list_dir", "args": {"path": "/src"}, "result": "file1.py"}],
    })

    msg = context_manager.persist_tool_result("call-1", "filesystem", result)

    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call-1"
    assert msg["content"] == result

    # scratchpad should NOT contain the data (no disk write until compact)
    sp_content = context_manager.scratchpad_path.read_text(encoding="utf-8")
    assert "list_dir" not in sp_content


def test_write_to_scratchpad_via_manager(context_manager: ContextManager):
    """ContextManager.write_to_scratchpad delegates to scratchpad.write."""
    ref = context_manager.write_to_scratchpad("Test Section", "test data here")

    assert "scratchpad#section-1" in ref
    sp_content = context_manager.scratchpad_path.read_text(encoding="utf-8")
    assert "test data here" in sp_content
