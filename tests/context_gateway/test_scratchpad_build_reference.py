from __future__ import annotations

from core.context.scratchpad import Scratchpad


def test_build_reference_no_sections(scratchpad: Scratchpad):
    """No archived sections -> empty reference (compact never triggered)."""
    ref = scratchpad.build_reference()
    assert ref == ""


def test_build_reference_after_archive(scratchpad: Scratchpad):
    """After write (archive), returns reference with path and instructions."""
    scratchpad.write("Archived", "x" * 500)

    ref = scratchpad.build_reference()
    assert "## Scratchpad (archived context)" in ref
    assert str(scratchpad.path) in ref
    assert "filesystem" in ref
    assert "search_grep" in ref
