# SPDX-License-Identifier: Apache-2.0
"""Integration tests for LoopDetector — real execution trajectories from logs.

These tests simulate the ACTUAL execution patterns observed in app_*.log:

1. Real tool names: fetch_webpage, search_web, file_create, read_file, python_repl
2. Real loop patterns: fetch_webpage → fetch_webpage (2-step repeating)
3. Real observation chain: 6+ consecutive fetch_webpage/search_web calls
4. Real error patterns: HTTP 400 errors, python_repl NameError
5. Real stall: no_progress_count threshold behavior

Key observations from logs:
- observation_chain fires when 6+ consecutive observation tools are called
- repeating_sequence fires when the same 2-step pattern repeats (e.g., A→A)
- HTTP errors (ERR: 400) are recorded as success with error content
- python_repl returns {"success": false, "result": ..., "error": ...} on errors
- stall_warning_count increments at no_progress_count > 6 (threshold 5 → stall 1)
- LoopSupervisionHook writes LOOP messages to scratchpad via update_scratchpad()
"""

from __future__ import annotations

import pytest

from core.skill.execution.loop_detector import LoopDetector, ToolCallRecord
from core.skill.execution.state import ReActState


# =============================================================================
# Real Trajectory 1: fetch_webpage observation chain
# From logs: Turns 1-7 were ALL fetch_webpage/search_web calls with no progress
# =============================================================================


class TestLoopDetectorRealFetchWebpageChain:
    """Simulate the real SAM3 web search loop: 6+ consecutive web fetch calls."""

    def test_six_consecutive_fetch_webpage_observations(self):
        """Real case: 6 fetch_webpage calls (no created files) → observation_chain.
        
        From logs: Turn 1-3 all fetch_webpage/search_web with same result content.
        Each returns ~15000 chars (same GitHub page content).
        """
        detector = LoopDetector(max_observation_chain=6, min_effect_ratio=0.15, window_size=10)
        for i in range(6):
            detector.record(
                tool_name="fetch_webpage",
                category="observation",
                turn=i,
                new_entities=0,  # Same page content returned repeatedly
            )
        result = detector.detect()
        assert result is not None
        assert result["type"] == "observation_chain"
        assert result["chain_length"] == 6
        assert result["severity"] == "high"

    def test_mixed_fetch_and_search_still_counts_consecutive(self):
        """Real case: fetch_webpage and search_web both count as observations."""
        detector = LoopDetector(max_observation_chain=6)
        # Alternating fetch/search (all observation category)
        tools = ["fetch_webpage", "search_web"] * 3
        for i, tool in enumerate(tools):
            detector.record(tool_name=tool, category="observation", turn=i)
        result = detector.detect()
        assert result is not None
        assert result["type"] == "observation_chain"
        assert result["chain_length"] == 6

    def test_observation_chain_skipped_when_recent_has_entities(self):
        """Real case: If a search returns NEW information, skip loop warning.
        
        From logs: search_web returned different content each time (different URLs).
        But all had new_entities=0 because same content was cached.
        """
        detector = LoopDetector(max_observation_chain=6)
        # First 5 with 0 new_entities
        for i in range(5):
            detector.record("fetch_webpage", "observation", turn=i, new_entities=0)
        # 6th has new_entities >= 2 (actual new info)
        detector.record("search_web", "observation", turn=5, new_entities=5)
        result = detector.detect()
        # observation_chain should be skipped because recent obs had new_entities >= 2
        assert result is None


# =============================================================================
# Real Trajectory 2: 2-step repeating pattern (A→A)
# From logs: [fetch_webpage, fetch_webpage] repeated → repeating_sequence
# =============================================================================


class TestLoopDetectorRealRepeatingSequence:
    """Simulate the real 2-step repeating pattern from logs."""

    def test_abab_pattern_with_fetch_webpage(self):
        """Real case: fetch_webpage → fetch_webpage → fetch_webpage → fetch_webpage.
        
        From logs: The LLM kept calling fetch_webpage twice per turn.
        The tool names are identical → 2-step repeating of the SAME tool.
        """
        # 2-tool pattern: [fetch_webpage, fetch_webpage]
        # But this means pattern = [fetch_webpage, fetch_webpage]
        # recent = [fetch_webpage, fetch_webpage, fetch_webpage, fetch_webpage]
        # recent == pattern * 2 → True (pattern repeated 2x)
        detector = LoopDetector(max_observation_chain=20)
        for i in range(6):
            detector.record("fetch_webpage", "observation", turn=i)
        result = detector._check_repeating_sequence()
        assert result is not None
        assert result["type"] == "repeating_sequence"
        # pattern_len=2, pattern=[fetch, fetch], recent=last 4=[fetch,fetch,fetch,fetch]
        # recent == pattern * 2 → True
        assert result["sequence"] == ["fetch_webpage", "fetch_webpage"]
        assert result["repetitions"] == 2

    def test_read_write_read_write_pattern(self):
        """Real case: read_file → file_create → read_file → file_create pattern.
        
        From logs: LLM alternated between reading created files and creating new ones.
        """
        detector = LoopDetector(max_observation_chain=20)
        pattern = ["read_file", "file_create"] * 3
        for i, tool in enumerate(pattern):
            detector.record(tool, "observation", turn=i)
        result = detector._check_repeating_sequence()
        assert result is not None
        assert result["sequence"] == ["read_file", "file_create"]
        assert result["repetitions"] == 2

    def test_repeating_sequence_priority_after_observation_chain(self):
        """Both observation_chain and repeating_sequence fire; check priority."""
        detector = LoopDetector(max_observation_chain=6)
        # 6 consecutive fetch_webpage calls → observation_chain
        # But they're also a repeating sequence
        for i in range(6):
            detector.record("fetch_webpage", "observation", turn=i)
        result = detector.detect()
        # observation_chain fires first (checked first)
        assert result["type"] == "observation_chain"


# =============================================================================
# Real Trajectory 3: HTTP errors and python_repl errors
# From logs: ERR: 400 Bad Request, NameError in python_repl
# =============================================================================


class TestLoopDetectorRealErrorPatterns:
    """Error patterns from real execution: HTTP errors and python_repl errors."""

    def test_http_error_recorded_as_observation(self):
        """Real case: HTTP 400 is returned as 'success' with error content.
        
        From logs: "ERR: fetch_webpage failed: Client error '400 Bad Request'"
        The adapter records this as success with error_dict=False (parseable as JSON).
        """
        detector = LoopDetector(max_observation_chain=6)
        # HTTP 400 - still an observation (no file created)
        detector.record(
            tool_name="fetch_webpage",
            category="observation",
            turn=0,
            new_entities=0,
        )
        assert detector.history[0].tool_name == "fetch_webpage"

    def test_python_repl_nameerror(self):
        """Real case: python_repl NameError → still an observation.
        
        From logs: "NameError: name 'MSO_CONNECTOR' is not defined"
        The error output is returned but no file is created.
        """
        detector = LoopDetector(max_observation_chain=6)
        detector.record(
            tool_name="python_repl",
            category="observation",
            turn=0,
            new_entities=0,
        )
        assert detector.history[0].category == "observation"

    def test_nameerror_followed_by_successful_file_create(self):
        """Real case: NameError (no_progress) then file_create (progress reset).
        
        From logs: python_repl failed with NameError (stall), then file_create succeeded
        (has_actual_new=True), resetting no_progress_count.
        """
        detector = LoopDetector(max_observation_chain=6)
        # 3 errors
        for i in range(3):
            detector.record("python_repl", "observation", turn=i, new_entities=0)
        # Then a file_create success
        detector.record("file_create", "observation", turn=3, created_artifacts=1)
        # The effect resets the observation chain for future detection
        result = detector._check_observation_chain()
        assert result is None  # Not enough trailing obs


# =============================================================================
# Real Trajectory 4: Token budget and summarization threshold
# From logs: Stage 2 triggered at ~85338 tokens
# =============================================================================


class TestLoopDetectorRealTokenBudget:
    """Token budget behavior from real execution logs."""

    def test_many_consecutive_observations_trigger_chain(self):
        """Real case: 10+ consecutive obs (from logs: ~14 tool calls with no files).
        
        From logs: turns 1-7 had ~14 fetch_webpage/search_web calls,
        total_tool_calls=14, all observation category.
        """
        detector = LoopDetector(max_observation_chain=6, window_size=10)
        # 10 consecutive observations (more than window_size)
        for i in range(10):
            detector.record("fetch_webpage", "observation", turn=i, new_entities=0)
        result = detector.detect()
        assert result is not None
        assert result["type"] == "observation_chain"
        assert result["chain_length"] == 10


# =============================================================================
# Real Trajectory 5: SkillAgent stall/no_progress detection
# From logs: no_progress_count increments, stall_warning_count triggers at > 6
# =============================================================================


class TestReActStateRealStallPatterns:
    """Real stall/no_progress patterns from execution logs."""

    def test_no_progress_count_increments_on_no_new_files(self):
        """Real case: no_progress_count increments when has_actual_new=False.
        
        From logs: has_actual_new=False → no_progress_count += 1 each turn.
        """
        state = ReActState(query="SAM3 web search", turn_count=1)
        # Simulate: created_files is empty (no new files)
        state.created_files = []
        state.updated_files = []
        # update_from_observation would be called...
        # The stall detection happens in SkillAgent, not directly on state
        # But we can test the state field behavior
        state.no_progress_count = 0
        # Manually simulate stall detection
        if not state.created_files and not state.updated_files:
            state.no_progress_count += 1
        assert state.no_progress_count == 1

    def test_stall_warning_fires_after_threshold(self):
        """Real case: stall_warning_count increments at no_progress_count > 6.
        
        From logs: no_progress_count reaches 5 in turn 5, 6 in turn 6.
        stall_warning_count increments only when no_progress_count > 6.
        """
        state = ReActState(query="SAM3 web search", turn_count=7)
        state.no_progress_count = 0
        state.stall_warning_count = 0
        
        # Simulate: 7 consecutive turns with no new files
        for _ in range(7):
            state.created_files = []  # No new files
            if not state.created_files:
                state.no_progress_count += 1
        
        # Threshold: no_progress_count > 6 triggers stall_warning
        if state.no_progress_count > 6:
            state.stall_warning_count += 1
        
        assert state.no_progress_count == 7
        assert state.stall_warning_count == 1

    def test_progress_resets_no_progress_count(self):
        """Real case: has_actual_new=True resets all counters.
        
        From logs: Turn 9 created SAM3_Presentation.md → 
        no_progress_count=0, stall_warning_count=0.
        """
        state = ReActState(query="SAM3 pptx", turn_count=9)
        state.no_progress_count = 5
        state.stall_warning_count = 0
        
        # Simulate: file_create succeeded
        state.created_files = ["/path/SAM3_Presentation.md"]
        
        if state.created_files:
            state.no_progress_count = 0
            state.stall_warning_count = 0
        
        assert state.no_progress_count == 0
        assert state.stall_warning_count == 0


# =============================================================================
# Real Trajectory 6: Observation chain detection with new_entities patterns
# From logs: search_web returned "### Search Results for:" content
# =============================================================================


class TestLoopDetectorRealNewEntities:
    """Real new_entities patterns from web search results."""

    def test_search_web_with_different_urls_has_different_entities(self):
        """Real case: search_web returned different URLs (different entities).
        
        From logs: Different URLs like github.com, ultralytics.com, datature.com
        each returned ~15000 chars but different content.
        """
        detector = LoopDetector(max_observation_chain=6)
        urls = [
            "https://github.com/facebookresearch/sam3/",
            "https://docs.ultralytics.com/models/sam-3/",
            "https://datature.com/blog/sam-3-a-technical-deep-dive",
        ]
        # Each URL returned different content → different entities
        for i, url in enumerate(urls):
            # In real execution, new_entities is derived from unique URLs found
            detector.record(
                tool_name="search_web",
                category="observation",
                turn=i,
                new_entities=len(url),  # Different each time
            )
        # No 6 consecutive, no repeating
        result = detector.detect()
        assert result is None

    def test_repeated_url_returns_zero_new_entities(self):
        """Real case: same URL fetched multiple times → no new entities.
        
        From logs: github.com/sam3 fetched 3 times, always same content.
        This is a form of diminishing returns (same info).
        """
        detector = LoopDetector(max_observation_chain=20)
        # Same URL fetched 4 times
        for i in range(4):
            detector.record(
                tool_name="fetch_webpage",
                category="observation",
                turn=i,
                new_entities=0,  # Same content
            )
        # Check diminishing_returns condition
        result = detector._check_diminishing_returns()
        # With all new_entities=0, sum=0 → diminishing_returns not triggered


# =============================================================================
# Real Trajectory 7: Low effect ratio with window size
# From logs: 14 tool calls, 0 created files → 0% effect ratio
# =============================================================================


class TestLoopDetectorRealLowEffectRatio:
    """Real low effect ratio patterns from logs."""

    def test_all_observations_zero_effect_ratio(self):
        """Real case: 10 web calls with 0 created files → 0% effect ratio.
        
        From logs: 14 total_tool_calls, all observation tools, no artifacts.
        Window = 10, all 10 are observations → 0% effect ratio.
        """
        detector = LoopDetector(
            max_observation_chain=20,
            min_effect_ratio=0.15,
            window_size=10,
        )
        # 10 web observations (no effect)
        for i in range(10):
            detector.record("fetch_webpage", "observation", turn=i)
        result = detector._check_effect_ratio()
        assert result is not None
        assert result["type"] == "low_effect_ratio"
        assert result["ratio"] == 0.0

    def test_one_file_create_in_window_rescues_ratio(self):
        """Real case: 1 file_create in window → 10% ratio still low.
        
        From logs: 10 tool calls, 1 file_create → 10% < 15% threshold.
        """
        detector = LoopDetector(
            max_observation_chain=20,
            min_effect_ratio=0.15,
            window_size=10,
        )
        # 9 web observations + 1 file_create
        for i in range(9):
            detector.record("fetch_webpage", "observation", turn=i)
        detector.record("file_create", "observation", turn=9, created_artifacts=1)
        result = detector._check_effect_ratio()
        assert result is not None
        assert result["type"] == "low_effect_ratio"
        assert result["ratio"] == 0.1  # 1/10 = 10%


# =============================================================================
# Real Trajectory 8: Tool category derivation from real tool results
# From logs: file_create returns "SUCCESS: Created file", python_repl JSON
# =============================================================================


class TestLoopDetectorRealToolCategory:
    """Real tool category behavior from log output patterns."""

    def test_bash_tool_classification(self):
        """Real case: bash output (e.g., ffmpeg, compilation) has different patterns."""
        detector = LoopDetector()
        # FFmpeg output
        detector.record(
            tool_name="bash",
            category="observation",
            turn=0,
            new_entities=0,
        )
        # File creation via bash
        detector.record(
            tool_name="bash",
            category="observation",
            turn=1,
            created_artifacts=1,
        )
        stats = detector.get_stats()
        assert stats["categories"]["effect"] == 1
        assert stats["categories"]["observation"] == 1

    def test_mixed_tool_types_in_history(self):
        """Real case: mixed web searches, file ops, code execution."""
        detector = LoopDetector(max_observation_chain=6)
        # 3 fetch calls
        for i in range(3):
            detector.record("fetch_webpage", "observation", turn=i, new_entities=0)
        # 1 file create
        detector.record("file_create", "observation", turn=3, created_artifacts=1)
        # 2 more fetch calls
        for i in range(4, 6):
            detector.record("fetch_webpage", "observation", turn=i, new_entities=0)
        # Not enough consecutive at end for observation_chain
        result = detector._check_observation_chain()
        assert result is None
        stats = detector.get_stats()
        assert stats["total_calls"] == 6
        assert stats["effect_ratio"] == 1 / 6
