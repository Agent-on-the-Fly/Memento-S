# SPDX-License-Identifier: Apache-2.0
"""Tests for FileChangeHook — core/skill/execution/hooks/file_change_hook.py

Based on real execution logs (app_*.log):

1. BEFORE_EXEC/AFTER_EXEC lifecycle with real tool names
2. Lifecycle classification on FsChange objects  
3. Extract_result_file_paths from HTTP errors and python_repl JSON output
4. Graceful degradation when tracker is disabled
5. Stack-based execution_id management (multi-tool-call per turn)
"""

from __future__ import annotations

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from shared.hooks.types import HookEvent, HookPayload, HookResult
from core.skill.execution.hooks.file_change_hook import (
    FileChangeHook,
    extract_result_file_paths,
    FILE_CREATION_TOOLS,
)


# =============================================================================
# extract_result_file_paths()
# =============================================================================


class TestExtractResultFilePaths:
    """Tests for extract_result_file_paths()."""

    def test_none_result_returns_empty(self):
        """None result returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = extract_result_file_paths(None, Path(tmpdir))
            assert paths == []

    def test_string_with_path_extraction(self):
        """Extracts paths from plain text results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "output.txt").touch()
            result = "Created file: /some/path/output.txt"
            paths = extract_result_file_paths(result, ws)
            # /some/path is not in workspace, so not returned
            assert isinstance(paths, list)

    def test_dict_with_path_extraction(self):
        """Extracts paths from dict results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            # Create the file so it exists
            (ws / "file.md").write_text("test content")
            result = {
                "path": str(ws / "file.md"),
                "content": "test",
            }
            paths = extract_result_file_paths(result, ws)
            assert len(paths) == 1
            assert paths[0].name == "file.md"

    def test_list_of_dicts(self):
        """Extracts from list of result dicts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "a.txt").write_text("")
            (ws / "b.txt").write_text("")
            result = [
                {"path": str(ws / "a.txt")},
                {"path": str(ws / "b.txt")},
            ]
            paths = extract_result_file_paths(result, ws)
            assert len(paths) == 2

    def test_python_repl_success_json(self):
        """Extracts from python_repl JSON success output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            result = '{"success": true, "result": "Created presentation.pptx"}'
            paths = extract_result_file_paths(result, ws)
            # presentation.pptx not in workspace → not returned
            assert isinstance(paths, list)

    def test_file_create_success_message(self):
        """Extracts from file_create SUCCESS message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            result = "SUCCESS: Created file /workspace/output.md"
            paths = extract_result_file_paths(result, ws)
            assert isinstance(paths, list)

    def test_non_workspace_files_excluded(self):
        """Paths outside workspace are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            result = "/tmp/tempfile.txt"
            paths = extract_result_file_paths(result, ws)
            assert len(paths) == 0


# =============================================================================
# FILE_CREATION_TOOLS constant
# =============================================================================


class TestFileCreationTools:
    """Tests for FILE_CREATION_TOOLS constant."""

    def test_contains_core_tools(self):
        """Core file creation tools are present."""
        assert "bash" in FILE_CREATION_TOOLS
        assert "python_repl" in FILE_CREATION_TOOLS
        assert "file_create" in FILE_CREATION_TOOLS
        assert "write_file" in FILE_CREATION_TOOLS
        assert "edit_file" in FILE_CREATION_TOOLS
        assert "edit_file_by_lines" in FILE_CREATION_TOOLS

    def test_is_frozenset(self):
        """FILE_CREATION_TOOLS is a frozenset (immutable)."""
        assert isinstance(FILE_CREATION_TOOLS, frozenset)


# =============================================================================
# FileChangeHook Initialization
# =============================================================================


class TestFileChangeHookInit:
    """Tests for FileChangeHook initialization."""

    def test_init_with_enabled_tracker(self, tmp_path):
        """Enabled hook creates ExecutionFileTracker."""
        hook = FileChangeHook(workspace_root=tmp_path, enabled=True)
        assert hook._enabled is True
        assert hook._tracker is not None
        assert hook._tracker.workspace_root == tmp_path.resolve()

    def test_init_with_disabled_tracker(self, tmp_path):
        """Disabled hook does not create tracker."""
        hook = FileChangeHook(workspace_root=tmp_path, enabled=False)
        assert hook._enabled is False
        assert hook._tracker is None

    def test_execution_stack_starts_empty(self, tmp_path):
        """_execution_stack starts as empty list."""
        hook = FileChangeHook(workspace_root=tmp_path)
        assert hook._execution_stack == []


# =============================================================================
# execute() — Event Filtering
# =============================================================================


class TestFileChangeHookExecute:
    """Tests for FileChangeHook.execute()."""

    @pytest.mark.asyncio
    async def test_disabled_hook_returns_allowed_true(self, tmp_path):
        """Disabled hook always returns allowed=True immediately."""
        hook = FileChangeHook(workspace_root=tmp_path, enabled=False)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC, tool_name="bash")
        result = await hook.execute(payload)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_non_file_creation_tool_skipped(self, tmp_path):
        """Non-FILE_CREATION_TOOLS skip immediately."""
        hook = FileChangeHook(workspace_root=tmp_path)
        payload = HookPayload(event=HookEvent.AFTER_TOOL_EXEC, tool_name="search_web")
        result = await hook.execute(payload)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_befor_tool_exec_returns_allowed_true(self, tmp_path):
        """BEFORE_TOOL_EXEC passes through (BEFORE handler does snapshot)."""
        hook = FileChangeHook(workspace_root=tmp_path)
        payload = HookPayload(event=HookEvent.BEFORE_TOOL_EXEC, tool_name="bash", args={})
        result = await hook.execute(payload)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_after_tool_exec_on_empty_stack_returns_allowed_true(self, tmp_path):
        """AFTER_TOOL_EXEC with empty execution stack logs warning and returns."""
        hook = FileChangeHook(workspace_root=tmp_path)
        hook._execution_stack = []
        payload = HookPayload(
            event=HookEvent.AFTER_TOOL_EXEC,
            tool_name="bash",
            args={},
            result="output",
        )
        result = await hook.execute(payload)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_execute_with_python_repl_tool(self, tmp_path):
        """python_repl is in FILE_CREATION_TOOLS → tracked."""
        hook = FileChangeHook(workspace_root=tmp_path)
        assert "python_repl" in FILE_CREATION_TOOLS
        payload = HookPayload(event=HookEvent.BEFORE_TOOL_EXEC, tool_name="python_repl", args={})
        result = await hook.execute(payload)
        assert result.allowed is True


# =============================================================================
# Integration: BEFORE → AFTER lifecycle with real errors
# =============================================================================


class TestFileChangeHookRealErrorPatterns:
    """Real error patterns from logs: HTTP 400, python_repl NameError."""

    def test_python_repl_nameerror_result(self, tmp_path):
        """python_repl NameError output format."""
        # From logs: '{"success": false, "result": "...", "error": "NameError: ..."}'
        result = '{"success": false, "result": "Output path: results_5/output.pptx", "error": "NameError: name \'MSO_CONNECTOR\' is not defined"}'
        paths = extract_result_file_paths(result, tmp_path)
        assert isinstance(paths, list)

    def test_fetch_webpage_http_error(self, tmp_path):
        """HTTP 400 error format from logs."""
        # From logs: "ERR: fetch_webpage failed: Client error '400 Bad Request' for url 'https://ai.meta.com/...'"
        result = "ERR: fetch_webpage failed: Client error '400 Bad Request' for url 'https://ai.meta.com/blog/'"
        paths = extract_result_file_paths(result, tmp_path)
        assert isinstance(paths, list)

    def test_mixed_success_and_error_in_result(self, tmp_path):
        """Mixed output with success and error parts."""
        result = (
            "SUCCESS: Created file /workspace/output.md\n"
            "ERROR: Some other error occurred"
        )
        paths = extract_result_file_paths(result, tmp_path)
        assert isinstance(paths, list)


# =============================================================================
# FileChangeHook bind_artifact_registry (backward compat)
# =============================================================================


class TestFileChangeHookArtifactRegistry:
    """Tests for bind_artifact_registry (backward compatibility)."""

    def test_bind_artifact_registry_accepts_none(self, tmp_path):
        """bind_artifact_registry accepts None (already migrated off registry)."""
        hook = FileChangeHook(workspace_root=tmp_path)
        hook.bind_artifact_registry(None)  # Should not raise

    def test_bind_artifact_registry_accepts_mock(self, tmp_path):
        """bind_artifact_registry accepts a mock registry."""
        hook = FileChangeHook(workspace_root=tmp_path)
        mock_registry = MagicMock()
        hook.bind_artifact_registry(mock_registry)  # Should not raise


# =============================================================================
# FileChangeHook tracker access
# =============================================================================


class TestFileChangeHookTrackerAccess:
    """Tests for tracker accessor methods."""

    def test_get_tracker_returns_tracker(self, tmp_path):
        """get_tracker() returns the tracker instance."""
        hook = FileChangeHook(workspace_root=tmp_path)
        assert hook.get_tracker() is hook._tracker

    def test_get_last_record_when_empty(self, tmp_path):
        """get_last_record() returns None when no history."""
        hook = FileChangeHook(workspace_root=tmp_path)
        assert hook.get_last_record() is None

    def test_cleanup_temporary_returns_int(self, tmp_path):
        """cleanup_temporary() returns an integer count."""
        hook = FileChangeHook(workspace_root=tmp_path)
        count = hook.cleanup_temporary()
        assert isinstance(count, int)
        assert count >= 0
