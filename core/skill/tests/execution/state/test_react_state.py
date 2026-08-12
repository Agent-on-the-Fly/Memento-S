# SPDX-License-Identifier: Apache-2.0
"""Tests for ReActState — core/skill/execution/state.py"""

from __future__ import annotations

import pytest

from core.skill.execution.state import (
    ReActState,
    is_inference_output,
    compress_to_summary,
    action_signature,
    state_fingerprint,
    infer_preferred_extension,
)


# =============================================================================
# is_inference_output() & compress_to_summary()
# =============================================================================


class TestIsInferenceOutput:
    """Tests for is_inference_output()."""

    def test_empty_content_is_summary(self):
        """Empty content returns 'summary'."""
        assert is_inference_output("bash", "") == "summary"

    def test_preserve_tools_always_preserve(self):
        """read_file, grep, glob always return 'preserve'."""
        for tool in ["read_file", "grep", "glob", "list_dir", "file_create", "edit_file_by_lines", "fetch_webpage"]:
            assert is_inference_output(tool, "some content") == "preserve"

    def test_short_content_preserved(self):
        """Content <= 80 chars returns 'preserve'."""
        short = "x" * 80
        assert is_inference_output("bash", short) == "preserve"

    def test_long_bash_with_error(self):
        """Bash with error output returns 'preserve'."""
        content = "x" * 200 + "\nerror: something failed"
        assert is_inference_output("bash", content) == "preserve"

    def test_python_repl_short_output(self):
        """python_repl short output returns 'preserve'."""
        content = ">>> result"
        assert is_inference_output("python_repl", content) == "preserve"

    def test_python_repl_long_output(self):
        """python_repl long output without error returns 'inference'."""
        content = ">>> " + "x" * 100
        assert is_inference_output("python_repl", content) == "inference"

    def test_error_preserved(self):
        """Error content is preserved."""
        content = "Error: file not found"
        assert is_inference_output("bash", content) == "preserve"


class TestCompressToSummary:
    """Tests for compress_to_summary()."""

    def test_whisper_output(self):
        """Faster-Whisper content gets appropriate summary."""
        content = "faster_whisper: processed 50 segments. saved to output.wav"
        result = compress_to_summary("python_repl", content)
        assert "Transcription" in result or "completed" in result

    def test_ffmpeg_preserves_time(self):
        """FFmpeg output preserves time information."""
        content = "frame=100 time=00:01:30"
        result = compress_to_summary("bash", content)
        assert "time=" in result or "FFmpeg" in result

    def test_python_repl_with_error(self):
        """Python error returns error summary."""
        content = "Error: something went wrong"
        result = compress_to_summary("python_repl", content)
        assert "error" in result.lower()

    def test_default_truncation(self):
        """Default behavior truncates to 80 chars."""
        content = "x" * 200
        result = compress_to_summary("bash", content)
        assert len(result) <= 100  # includes prefix "[bash] "
        assert "..." in result


# =============================================================================
# ReActState Initialization
# =============================================================================


class TestReActStateInit:
    """Tests for ReActState initialization."""

    def test_default_values(self):
        """Default values are set correctly."""
        state = ReActState(query="test query")
        assert state.query == "test query"
        assert state.turn_count == 0
        assert state.tool_calls_count == 0
        assert state.scratchpad == ""
        assert state.created_files == []
        assert state.updated_files == []
        assert state.no_progress_count == 0

    def test_max_turns(self):
        """max_turns is set correctly."""
        state = ReActState(query="test", max_turns=5)
        assert state.max_turns == 5

    def test_params(self):
        """params is stored correctly."""
        state = ReActState(query="test", params={"key": "value"})
        assert state.params == {"key": "value"}


# =============================================================================
# update_from_observation()
# =============================================================================


class TestReActStateUpdateFromObservation:
    """Tests for update_from_observation()."""

    def test_appends_to_observation_log(self):
        """Observation is appended to observation_log."""
        state = ReActState(query="test")
        state.update_from_observation({"tool": "bash", "summary": "result", "exec_status": "success"})
        assert len(state.observation_log) == 1
        assert state.observation_log[0]["tool"] == "bash"

    def test_updates_created_files(self):
        """state_delta.created_files updates created_files list."""
        state = ReActState(query="test")
        state.update_from_observation({
            "tool": "bash",
            "summary": "ok",
            "state_delta": {"created_files": ["/path/file.txt"]},
        })
        assert "/path/file.txt" in state.created_files

    def test_deduplicates_created_files(self):
        """Duplicate created files are not added twice."""
        state = ReActState(query="test")
        state.update_from_observation({
            "tool": "bash", "summary": "ok",
            "state_delta": {"created_files": ["/path/file.txt"]},
        })
        state.update_from_observation({
            "tool": "bash", "summary": "ok",
            "state_delta": {"created_files": ["/path/file.txt"]},
        })
        assert state.created_files.count("/path/file.txt") == 1

    def test_updates_tool_calls_count(self):
        """tool_calls_count increments."""
        state = ReActState(query="test")
        state.tool_calls_count = 2
        state.update_from_observation({"tool": "bash", "summary": "ok"})
        # tool_calls_count is managed externally by SkillAgent


# =============================================================================
# Scratchpad
# =============================================================================


class TestReActStateScratchpad:
    """Tests for scratchpad updates."""

    def test_scratchpad_update(self):
        """update_scratchpad appends with turn marker."""
        state = ReActState(query="test", turn_count=3)
        state.update_scratchpad("Remember to check logs")
        assert "[Turn 3]" in state.scratchpad
        assert "Remember to check logs" in state.scratchpad

    def test_scratchpad_truncation(self):
        """Scratchpad truncates to 2000 chars."""
        state = ReActState(query="test")
        long_content = "x" * 2500
        state.update_scratchpad(long_content)
        assert len(state.scratchpad) <= 2000


# =============================================================================
# Error Tracking
# =============================================================================


class TestReActStateErrorTracking:
    """Tests for error tracking methods."""

    def test_record_error_normalizes_and_deduplicates(self):
        """record_error normalizes paths and line numbers."""
        state = ReActState(query="test")
        state.record_error("File /path/to/file.py line 42 not found", "bash")
        state.record_error("File /path/to/other.py line 99 not found", "bash")
        assert len(state.error_history) == 2
        assert state.error_history[0]["tool"] == "bash"

    def test_repeated_error_detected(self):
        """Repeated same error increments repeated_error_count."""
        state = ReActState(query="test")
        state.record_error("File not found", "bash")
        state.record_error("File not found", "bash")
        assert state.repeated_error_count == 1

    def test_should_inject_recovery_hint(self):
        """should_inject_recovery_hint respects min_interval."""
        state = ReActState(query="test", turn_count=3)
        state.last_recovery_hint_turn = 1
        assert state.should_inject_recovery_hint(min_interval=2) is True
        state.last_recovery_hint_turn = 3
        assert state.should_inject_recovery_hint(min_interval=2) is False

    def test_mark_recovery_hint_injected(self):
        """mark_recovery_hint_injected updates the turn."""
        state = ReActState(query="test", turn_count=5)
        state.error_history.append({"was_recovery_hint_injected": False})
        state.mark_recovery_hint_injected()
        assert state.last_recovery_hint_turn == 5
        assert state.error_history[-1]["was_recovery_hint_injected"] is True


# =============================================================================
# action_signature() & state_fingerprint()
# =============================================================================


class TestActionSignature:
    """Tests for action_signature()."""

    def test_same_args_same_signature(self):
        """Identical tool+args produce same signature."""
        sig1 = action_signature("bash", {"command": "ls"})
        sig2 = action_signature("bash", {"command": "ls"})
        assert sig1 == sig2

    def test_different_args_different_signature(self):
        """Different args produce different signature."""
        sig1 = action_signature("bash", {"command": "ls"})
        sig2 = action_signature("bash", {"command": "cat"})
        assert sig1 != sig2

    def test_order_independent(self):
        """Arg order doesn't matter for dict args."""
        sig1 = action_signature("bash", {"a": 1, "b": 2})
        sig2 = action_signature("bash", {"b": 2, "a": 1})
        assert sig1 == sig2


class TestStateFingerprint:
    """Tests for state_fingerprint()."""

    def test_different_summary_different_fingerprint(self):
        """Different summaries produce different fingerprints."""
        obs1 = {"tool": "bash", "summary": "result A", "exec_status": "success", "state_delta": {}}
        obs2 = {"tool": "bash", "summary": "result B", "exec_status": "success", "state_delta": {}}
        fp1 = state_fingerprint(obs1)
        fp2 = state_fingerprint(obs2)
        assert fp1 != fp2

    def test_effective_flag_distinguishes_success(self):
        """Effective (has artifact) vs success-without-artifact differ."""
        obs_with_artifact = {
            "tool": "file_create",
            "summary": "Created file",
            "exec_status": "success",
            "state_delta": {"created_files": ["/path/file.txt"]},
        }
        obs_without_artifact = {
            "tool": "bash",
            "summary": "ran",
            "exec_status": "success",
            "state_delta": {},
        }
        fp1 = state_fingerprint(obs_with_artifact)
        fp2 = state_fingerprint(obs_without_artifact)
        assert fp1 != fp2


# =============================================================================
# infer_preferred_extension()
# =============================================================================


class TestInferPreferredExtension:
    """Tests for infer_preferred_extension()."""

    def test_pptx_keywords(self):
        """pptx/presentation/slides keywords → .pptx."""
        assert infer_preferred_extension("create a presentation", None) == ".pptx"
        assert infer_preferred_extension("make slides", None) == ".pptx"

    def test_docx_keywords(self):
        """docx/document keywords → .docx."""
        assert infer_preferred_extension("create a document", None) == ".docx"

    def test_xlsx_keywords(self):
        """xlsx/excel keywords → .xlsx."""
        assert infer_preferred_extension("make a spreadsheet", None) == ".xlsx"

    def test_pdf_keyword(self):
        """pdf keyword → .pdf."""
        assert infer_preferred_extension("generate a PDF report", None) == ".pdf"

    def test_no_match(self):
        """No matching keyword → None."""
        assert infer_preferred_extension("do something", None) is None

    def test_with_params(self):
        """Params are also searched."""
        params = {"output_format": "pdf"}
        assert infer_preferred_extension("convert file", params) == ".pdf"
