from __future__ import annotations

from core.context.manager import ContextManager


def test_context_section_empty_state(context_manager: ContextManager):
    """No large scratchpad -> empty context section."""
    section = context_manager._get_context_section()
    assert section == ""


def test_context_section_with_scratchpad(context_manager: ContextManager):
    """Large scratchpad produces reference in context section."""
    context_manager.write_to_scratchpad("Big", "x" * 500)

    section = context_manager._get_context_section()
    assert "Scratchpad" in section
