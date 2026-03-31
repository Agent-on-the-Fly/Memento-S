from __future__ import annotations

from core.context.scratchpad import Scratchpad


def test_persist_result_inline_no_disk_write(scratchpad: Scratchpad):
    """All results stay inline without disk write."""
    initial_content = scratchpad.path.read_text(encoding="utf-8")

    short_result = f'{{"ok": true, "output": "content of {scratchpad.path.name}..."}}'

    msg = scratchpad.persist_tool_result("call-3", "filesystem", short_result)

    assert msg["role"] == "tool"
    assert msg["content"] == short_result
    assert scratchpad.path.read_text(encoding="utf-8") == initial_content
