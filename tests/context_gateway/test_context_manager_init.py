from __future__ import annotations

from core.context.manager import ContextManager


def test_context_manager_paths_exist(context_manager: ContextManager):
    """ContextManager creates scratchpad file on init."""
    assert context_manager.scratchpad_path.exists()
    assert context_manager.scratchpad_path.name.startswith("scratchpad_")


def test_context_manager_session_id(context_manager: ContextManager):
    """ContextManager stores session_id correctly."""
    assert context_manager.session_id == "test-session"
