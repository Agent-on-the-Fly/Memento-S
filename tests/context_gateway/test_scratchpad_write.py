from __future__ import annotations

from core.context.scratchpad import Scratchpad


def test_scratchpad_write_returns_anchor(scratchpad: Scratchpad):
    """write() appends section and returns anchor reference."""
    ref = scratchpad.write("Test Section", "some content")
    assert "scratchpad#section-1" in ref

    content = scratchpad.path.read_text(encoding="utf-8")
    assert "Test Section" in content
    assert "some content" in content


def test_scratchpad_write_anchors_increment(scratchpad: Scratchpad):
    """Multiple writes produce incrementing anchors."""
    ref1 = scratchpad.write("First", "aaa")
    ref2 = scratchpad.write("Second", "bbb")
    ref3 = scratchpad.write("Third", "ccc")

    assert "section-1" in ref1
    assert "section-2" in ref2
    assert "section-3" in ref3
