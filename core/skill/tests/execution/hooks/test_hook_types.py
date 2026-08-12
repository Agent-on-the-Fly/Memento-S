# SPDX-License-Identifier: Apache-2.0
"""Tests for Hook types and events — shared/hooks/types.py"""

from __future__ import annotations

import pytest

from shared.hooks.types import HookEvent, HookPayload, HookResult


class TestHookEventValues:
    """Tests for HookEvent enum values."""

    def test_all_events_are_strings(self):
        """All HookEvent values are strings."""
        for event in HookEvent:
            assert isinstance(event.value, str)

    def test_before_tool_exec_value(self):
        """BEFORE_TOOL_EXEC has correct value."""
        assert HookEvent.BEFORE_TOOL_EXEC.value == "before_tool_exec"

    def test_after_tool_exec_value(self):
        """AFTER_TOOL_EXEC has correct value."""
        assert HookEvent.AFTER_TOOL_EXEC.value == "after_tool_exec"

    def test_before_skill_exec_value(self):
        """BEFORE_SKILL_EXEC has correct value."""
        assert HookEvent.BEFORE_SKILL_EXEC.value == "before_skill_exec"

    def test_after_skill_exec_value(self):
        """AFTER_SKILL_EXEC has correct value."""
        assert HookEvent.AFTER_SKILL_EXEC.value == "after_skill_exec"

    def test_on_loop_detected_value(self):
        """ON_LOOP_DETECTED has correct value."""
        assert HookEvent.ON_LOOP_DETECTED.value == "on_loop_detected"


class TestHookPayload:
    """Tests for HookPayload dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        payload = HookPayload(event=HookEvent.BEFORE_TOOL_EXEC)
        assert payload.event == HookEvent.BEFORE_TOOL_EXEC
        assert payload.tool_name == ""
        assert payload.args == {}
        assert payload.result is None
        assert payload.error is None
        assert payload.skill is None
        assert payload.skill_params is None

    def test_args_defaults_to_empty_dict(self):
        """args defaults to {} if None is set."""
        payload = HookPayload(event=HookEvent.BEFORE_TOOL_EXEC, args=None)
        # __post_init__ converts None to {}
        assert payload.args == {}

    def test_full_payload(self):
        """Full payload with all fields."""
        payload = HookPayload(
            event=HookEvent.AFTER_TOOL_EXEC,
            tool_name="bash",
            args={"command": "ls"},
            result="file1.txt\nfile2.txt",
            error={"error_type": "timeout"},
        )
        assert payload.tool_name == "bash"
        assert payload.args == {"command": "ls"}
        assert payload.result == "file1.txt\nfile2.txt"
        assert payload.error["error_type"] == "timeout"

    def test_skill_payload(self):
        """Skill execution payload."""
        payload = HookPayload(
            event=HookEvent.BEFORE_SKILL_EXEC,
            skill=MagicMock(),
            skill_params={"query": "test"},
        )
        assert payload.skill is not None
        assert payload.skill_params == {"query": "test"}


class TestHookResult:
    """Tests for HookResult dataclass."""

    def test_default_values(self):
        """Default values: allowed=True, no blocking."""
        result = HookResult()
        assert result.allowed is True
        assert result.reason == ""
        assert result.modified_args is None
        assert result.detected_artifacts is None
        assert result.deferred_messages is None
        assert result.recovery_action is None
        assert result.fs_changes is None

    def test_blocked_result(self):
        """blocked result has allowed=False and reason."""
        result = HookResult(allowed=False, reason="dangerous command")
        assert result.allowed is False
        assert result.reason == "dangerous command"

    def test_with_detected_artifacts(self):
        """Result with detected_artifacts."""
        result = HookResult(
            detected_artifacts=["/path/a.txt", "/path/b.txt"]
        )
        assert result.detected_artifacts == ["/path/a.txt", "/path/b.txt"]

    def test_with_deferred_messages(self):
        """Result with deferred_messages."""
        result = HookResult(
            deferred_messages=[
                {"role": "user", "content": "LOOP_DETECTED"},
                {"role": "user", "content": "ERROR_RECOVERY"},
            ]
        )
        assert len(result.deferred_messages) == 2

    def test_with_recovery_action(self):
        """Result with recovery_action."""
        result = HookResult(recovery_action="RECOMMEND_ABORT")
        assert result.recovery_action == "RECOMMEND_ABORT"

    def test_with_fs_changes(self):
        """Result with fs_changes."""
        result = HookResult(
            fs_changes={
                "created": ["/path/a.txt"],
                "modified": ["/path/b.txt"],
                "deleted": [],
            }
        )
        assert result.fs_changes is not None
        assert "/path/a.txt" in result.fs_changes["created"]
        assert "/path/b.txt" in result.fs_changes["modified"]

    def test_combined_result(self):
        """Result with all fields combined."""
        result = HookResult(
            allowed=True,
            reason="",
            detected_artifacts=["/path/a.txt"],
            deferred_messages=[{"role": "user", "content": "msg"}],
            recovery_action="RECOMMEND_ABORT",
            fs_changes={"created": [], "modified": [], "deleted": []},
        )
        assert result.allowed is True
        assert result.detected_artifacts is not None
        assert result.deferred_messages is not None
        assert result.recovery_action == "RECOMMEND_ABORT"
        assert result.fs_changes is not None


# Need this for MagicMock import
from unittest.mock import MagicMock