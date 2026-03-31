from __future__ import annotations

from core.context.scratchpad import Scratchpad


def test_scratchpad_init(scratchpad: Scratchpad):
    """Scratchpad file is created on init with correct header."""
    assert scratchpad.path.exists()
    content = scratchpad.path.read_text(encoding="utf-8")
    assert "# Session Scratchpad" in content
    assert "test-session" in content
