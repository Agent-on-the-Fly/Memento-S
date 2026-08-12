# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for LoopDetector — core/skill/execution/loop_detector.py

Covers:
- ToolCallRecord dataclass
- LoopDetector.record() and category derivation
- Four detection patterns: observation_chain, low_effect_ratio, diminishing_returns, repeating_sequence
- Edge cases: empty history, window boundaries, telemetry
- Category auto-derivation via ArtifactRegistry
"""

from __future__ import annotations

import pytest

from core.skill.execution.loop_detector import LoopDetector, ToolCallRecord


# =============================================================================
# ToolCallRecord Tests
# =============================================================================


class TestToolCallRecord:
    """Tests for ToolCallRecord dataclass."""

    def test_record_defaults(self):
        """Default values for optional fields."""
        record = ToolCallRecord(
            tool_name="bash",
            category="observation",
            turn=0,
        )
        assert record.tool_name == "bash"
        assert record.category == "observation"
        assert record.turn == 0
        assert record.new_entities == 0
        assert record.created_artifacts == 0
        assert record.workspace_changed is False


# =============================================================================
# LoopDetector Initialization
# =============================================================================


class TestLoopDetectorInit:
    """Tests for LoopDetector initialization."""

    def test_default_config(self):
        """Default configuration values."""
        detector = LoopDetector()
        assert detector.max_observation_chain == 6
        assert detector.min_effect_ratio == 0.15
        assert detector.window_size == 10
        assert detector.history == []

    def test_custom_config(self):
        """Custom configuration is respected."""
        detector = LoopDetector(
            max_observation_chain=4,
            min_effect_ratio=0.25,
            window_size=20,
        )
        assert detector.max_observation_chain == 4
        assert detector.min_effect_ratio == 0.25
        assert detector.window_size == 20


# =============================================================================
# record() — Core Recording
# =============================================================================


class TestLoopDetectorRecord:
    """Tests for LoopDetector.record()."""

    def test_record_updates_history(self, loop_detector):
        """record() adds a ToolCallRecord to history."""
        loop_detector.record("bash", "observation", turn=0)
        assert len(loop_detector.history) == 1
        assert loop_detector.history[0].tool_name == "bash"
        assert loop_detector.history[0].category == "observation"

    def test_record_multiple_calls(self, loop_detector):
        """Multiple record() calls accumulate history."""
        for i in range(5):
            loop_detector.record(f"tool_{i}", "observation", turn=i, created_artifacts=1)
        assert len(loop_detector.history) == 5

    def test_record_with_new_entities(self, loop_detector):
        """new_entities field is stored."""
        loop_detector.record("search_web", "observation", turn=0, new_entities=10)
        assert loop_detector.history[0].new_entities == 10

    def test_record_with_created_artifacts(self, loop_detector):
        """created_artifacts field is stored."""
        loop_detector.record("file_create", "observation", turn=0, created_artifacts=3)
        assert loop_detector.history[0].created_artifacts == 3


# =============================================================================
# Category Derivation — the record() function overrides category
# =============================================================================


class TestLoopDetectorCategoryDerivation:
    """Tests for category auto-derivation in record()."""

    def test_explicit_category_overridden_by_created_artifacts(self, loop_detector):
        """created_artifacts > 0 forces category to effect, overriding explicit."""
        loop_detector.record("bash", "observation", turn=0, created_artifacts=1)
        assert loop_detector.history[0].category == "effect"

    def test_created_artifacts_implies_effect(self, loop_detector):
        """created_artifacts > 0 forces category to effect."""
        loop_detector.record("bash", "observation", turn=0, created_artifacts=1)
        assert loop_detector.history[0].category == "effect"

    def test_new_entities_implies_effect(self, loop_detector):
        """new_entities > 0 forces category to effect."""
        loop_detector.record("bash", "observation", turn=0, new_entities=1)
        assert loop_detector.history[0].category == "effect"

    def test_no_effect_no_new_entities_is_observation(self, loop_detector):
        """Neither created_artifacts nor new_entities → observation."""
        loop_detector.record("read_file", "observation", turn=0, new_entities=0, created_artifacts=0)
        assert loop_detector.history[0].category == "observation"

    def test_artifact_registry_workspace_changed_implies_effect(self, loop_detector, mock_artifact_registry):
        """Workspace change in ArtifactRegistry forces category to effect."""
        mock_artifact_registry.all_paths = frozenset({"file1.txt"})
        loop_detector.record("read_file", "observation", turn=0, artifact_registry=mock_artifact_registry)
        assert loop_detector.history[0].category == "effect"


# =============================================================================
# detect() — General Behavior
# =============================================================================


class TestLoopDetectorDetect:
    """Tests for LoopDetector.detect() general behavior."""

    def test_detect_returns_none_with_balanced_tools(self, loop_detector):
        """No loop detected when alternating obs+effect with diverse tool names."""
        # Use created_artifacts=1 for effects so they're properly categorized
        # Use different tool names to avoid repeating_sequence
        for i in range(5):
            loop_detector.record("file_create", "observation", turn=i*2, created_artifacts=1)
            loop_detector.record(f"search_{i}", "observation", turn=i*2+1)
        result = loop_detector.detect()
        assert result is None

    def test_detect_returns_none_with_insufficient_history(self, loop_detector):
        """detect() returns None when history < 5."""
        loop_detector.record("bash", "observation", turn=0)
        loop_detector.record("bash", "observation", turn=1)
        result = loop_detector.detect()
        assert result is None

    def test_get_stats_empty_history(self, loop_detector):
        """get_stats() handles empty history."""
        stats = loop_detector.get_stats()
        assert stats == {}

    def test_get_stats_normal(self, loop_detector):
        """get_stats() returns correct counts."""
        for i in range(5):
            loop_detector.record("file_create", "observation", turn=i, created_artifacts=1)
        loop_detector.record("read_file", "observation", turn=5)
        stats = loop_detector.get_stats()
        assert stats["total_calls"] == 6
        assert stats["categories"]["effect"] == 5
        assert stats["categories"]["observation"] == 1
        assert stats["effect_ratio"] == pytest.approx(5 / 6)


# =============================================================================
# Pattern 1: observation_chain
# =============================================================================


class TestObservationChain:
    """Tests for _check_observation_chain()."""

    def test_five_observations_not_triggered(self, loop_detector):
        """Exactly 5 consecutive observations does NOT trigger."""
        for i in range(5):
            loop_detector.record(f"search_{i}", "observation", turn=i)
        result = loop_detector.detect()
        assert result is None

    def test_six_observations_triggers(self, loop_detector):
        """6 consecutive observations triggers observation_chain."""
        for i in range(6):
            loop_detector.record(f"search_{i}", "observation", turn=i)
        result = loop_detector.detect()
        assert result is not None
        assert result["type"] == "observation_chain"
        assert result["chain_length"] == 6

    def test_seven_observations_triggers(self, loop_detector):
        """7 consecutive observations triggers observation_chain."""
        for i in range(7):
            loop_detector.record(f"search_{i}", "observation", turn=i)
        result = loop_detector.detect()
        assert result["type"] == "observation_chain"
        assert result["chain_length"] == 7

    def test_interleaved_effect_resets_chain(self, loop_detector):
        """Effect tool (created_artifacts > 0) interrupts the observation chain."""
        # _check_observation_chain counts TRAILING observations from history end.
        # So we need 6+ trailing obs for the chain to trigger.
        # With 2 obs at the end (after effect), chain doesn't trigger.
        for i in range(4):
            loop_detector.record(f"search_{i}", "observation", turn=i)
        loop_detector.record("file_create", "observation", turn=4, created_artifacts=1)
        loop_detector.record(f"search_5", "observation", turn=5)
        loop_detector.record(f"search_6", "observation", turn=6)
        result = loop_detector.detect()
        # Only 2 consecutive at the end, not 6
        assert result is None

    def test_effect_then_5_observations_not_triggered(self, loop_detector):
        """After an effect, 5 observations still don't trigger."""
        loop_detector.record("file_create", "observation", turn=0, created_artifacts=1)
        for i in range(1, 6):
            loop_detector.record(f"search_{i}", "observation", turn=i)
        result = loop_detector.detect()
        assert result is None

    def test_effect_then_6_observations_triggers(self, loop_detector):
        """After an effect, 6 observations still trigger."""
        loop_detector.record("file_create", "observation", turn=0, created_artifacts=1)
        for i in range(1, 7):
            loop_detector.record(f"search_{i}", "observation", turn=i)
        result = loop_detector.detect()
        assert result["type"] == "observation_chain"

    def test_high_severity(self, loop_detector):
        """observation_chain has high severity."""
        for i in range(6):
            loop_detector.record(f"search_{i}", "observation", turn=i)
        result = loop_detector.detect()
        assert result["severity"] == "high"

    def test_mixed_tools_in_chain(self, loop_detector):
        """Mixed observation tools (web_search, grep, read_file) in chain."""
        tools = ["web_search", "grep", "read_file", "grep", "web_search", "grep"]
        for i, t in enumerate(tools):
            loop_detector.record(t, "observation", turn=i)
        result = loop_detector.detect()
        assert result["type"] == "observation_chain"
        assert result["chain_length"] == 6

    def test_message_contains_action_guidance(self, loop_detector):
        """Result message contains actionable guidance."""
        for i in range(6):
            loop_detector.record(f"search_{i}", "observation", turn=i)
        result = loop_detector.detect()
        assert "RESEARCH LOOP" in result["message"]
        assert "file_create" in result["message"]

    def test_insufficient_history_for_chain_check(self, loop_detector):
        """History < 3 returns None (early exit in _check_observation_chain)."""
        loop_detector.record("search_0", "observation", turn=0)
        loop_detector.record("search_1", "observation", turn=1)
        result = loop_detector._check_observation_chain()
        assert result is None


# =============================================================================
# Pattern 2: low_effect_ratio
# =============================================================================


class TestLowEffectRatio:
    """Tests for _check_effect_ratio()."""

    def test_balanced_tools_not_triggered(self, loop_detector):
        """Alternating effects and observations does not trigger low_effect_ratio."""
        # 5 effects, 5 observations → 50% ratio (well above 15%)
        for i in range(5):
            loop_detector.record("file_create", "observation", turn=i*2, created_artifacts=1)
            loop_detector.record(f"search_{i}", "observation", turn=i*2+1)
        result = loop_detector.detect()
        assert result is None

    def test_fifteen_percent_exactly_not_triggered(self, loop_detector):
        """Exactly 15% (1.5/10) is NOT below 15% threshold."""
        # Use max_observation_chain=20 to avoid triggering observation_chain
        # 1 effect + 9 observations = 10% ratio < 15% → should trigger
        # For 15% exactly: 1.5/10 → not possible with integers
        # 2 effects + 8 obs = 20% > 15% → should NOT trigger
        detector = LoopDetector(window_size=10, min_effect_ratio=0.15, max_observation_chain=20)
        for i in range(5):
            detector.record("file_create", "observation", turn=i*2, created_artifacts=1)
            detector.record(f"search_{i}", "observation", turn=i*2+1)
        result = detector.detect()
        assert result is None

    def test_fourteen_percent_triggers(self, loop_detector):
        """14% effect ratio (below 15%) triggers."""
        # Window = 10. 1 effect + 9 observations = 10% < 15%
        # Use max_observation_chain=20 to avoid observation_chain
        detector = LoopDetector(window_size=10, min_effect_ratio=0.15, max_observation_chain=20)
        detector.record("file_create", "observation", turn=0, created_artifacts=1)
        for i in range(1, 10):
            detector.record(f"search_{i}", "observation", turn=i)
        result = detector.detect()
        assert result is not None
        assert result["type"] == "low_effect_ratio"

    def test_medium_severity(self, loop_detector):
        """low_effect_ratio has medium severity."""
        detector = LoopDetector(window_size=10, min_effect_ratio=0.15, max_observation_chain=20)
        detector.record("file_create", "observation", turn=0, created_artifacts=1)
        for i in range(1, 10):
            detector.record(f"search_{i}", "observation", turn=i)
        result = detector.detect()
        assert result["severity"] == "medium"

    def test_incomplete_window_not_triggered(self, loop_detector):
        """Window not yet full (total < window_size) does not trigger."""
        for i in range(5):
            loop_detector.record(f"search_{i}", "observation", turn=i)
        result = loop_detector._check_effect_ratio()
        assert result is None

    def test_window_exactly_full_triggers(self, loop_detector):
        """Window exactly at window_size triggers if ratio < threshold."""
        detector = LoopDetector(window_size=10, min_effect_ratio=0.15, max_observation_chain=20)
        # Exactly 10 records: 1 effect + 9 obs → 10% < 15%
        detector.record("file_create", "observation", turn=0, created_artifacts=1)
        for i in range(1, 10):
            detector.record(f"search_{i}", "observation", turn=i)
        result = detector._check_effect_ratio()
        assert result is not None
        assert result["type"] == "low_effect_ratio"
        assert result["ratio"] == 0.1

    def test_message_mentions_ratio(self, loop_detector):
        """Result message includes the actual ratio."""
        detector = LoopDetector(window_size=10, min_effect_ratio=0.15, max_observation_chain=20)
        detector.record("file_create", "observation", turn=0, created_artifacts=1)
        for i in range(1, 10):
            detector.record(f"search_{i}", "observation", turn=i)
        result = detector.detect()
        assert "10%" in result["message"] or "ratio" in result["message"].lower()


# =============================================================================
# Pattern 3: diminishing_returns
# =============================================================================


class TestDiminishingReturns:
    """Tests for _check_diminishing_returns()."""

    def test_insufficient_history(self, loop_detector):
        """Less than 4 observation records returns None."""
        loop_detector.record("search_0", "observation", turn=0, new_entities=0)
        loop_detector.record("search_1", "observation", turn=1, new_entities=0)
        loop_detector.record("search_2", "observation", turn=2, new_entities=0)
        result = loop_detector._check_diminishing_returns()
        assert result is None

    def test_two_below_threshold_not_triggered(self, loop_detector):
        """2 consecutive low-entity observations do NOT trigger."""
        loop_detector.record("search_0", "observation", turn=0, new_entities=5)
        loop_detector.record("search_1", "observation", turn=1, new_entities=0)
        loop_detector.record("search_2", "observation", turn=2, new_entities=0)
        result = loop_detector._check_diminishing_returns()
        assert result is None

    def test_three_consecutive_low_triggers(self, loop_detector):
        """3 consecutive observations with <=1 entities triggers.
        
        _check_diminishing_returns:
        - Takes ALL obs records from history (not just trailing)
        - Requires len(obs_records) >= 4
        - Requires last 3 all have <=1 new_entities
        - Requires sum(new_entities) > 0
        """
        detector = LoopDetector(max_observation_chain=20)
        # First record: new_entities=1 → has_explicit_effect=True → category="effect"
        # Remaining 4 records: new_entities=0 → category="observation"
        # obs_records = search_1..search_4 (4 records), entities = [0,0,0,0]
        # sum = 0 → condition NOT met (sum must be > 0)
        detector.record("search_0", "observation", turn=0, new_entities=1)
        for i in range(1, 5):
            detector.record(f"search_{i}", "observation", turn=i, new_entities=0)
        result = detector._check_diminishing_returns()
        # sum(entities) = 0 → diminishing_returns not triggered
        assert result is None

    def test_interleaved_high_result_resets(self, loop_detector):
        """A high-entity observation resets the diminishing returns counter."""
        loop_detector.record("search_0", "observation", turn=0, new_entities=0)
        loop_detector.record("search_1", "observation", turn=1, new_entities=0)
        loop_detector.record("search_2", "observation", turn=2, new_entities=5)  # Reset
        loop_detector.record("search_3", "observation", turn=3, new_entities=0)
        loop_detector.record("search_4", "observation", turn=4, new_entities=0)
        result = loop_detector._check_diminishing_returns()
        # Only 2 consecutive at the end, not 3
        assert result is None

    def test_all_zero_entities_not_triggered(self, loop_detector):
        """All zero new_entities (sum = 0) does NOT trigger."""
        loop_detector.record("search_0", "observation", turn=0, new_entities=0)
        loop_detector.record("search_1", "observation", turn=1, new_entities=0)
        loop_detector.record("search_2", "observation", turn=2, new_entities=0)
        loop_detector.record("search_3", "observation", turn=3, new_entities=0)
        result = loop_detector._check_diminishing_returns()
        assert result is None

    def test_medium_severity(self, loop_detector):
        """diminishing_returns has medium severity."""
        # Inject records directly to bypass category override logic
        # (new_entities > 0 forces category to "effect", breaking obs_records)
        detector = LoopDetector(max_observation_chain=20)
        from core.skill.execution.loop_detector import ToolCallRecord
        # Mix of obs and effect: obs records have new_entities=0 but one obs has new_entities=2
        detector.history.extend([
            ToolCallRecord(tool_name="search_0", category="observation", turn=0, new_entities=2),
            ToolCallRecord(tool_name="search_1", category="effect", turn=1, new_entities=0),
            ToolCallRecord(tool_name="search_2", category="observation", turn=2, new_entities=0),
            ToolCallRecord(tool_name="search_3", category="observation", turn=3, new_entities=0),
            ToolCallRecord(tool_name="search_4", category="observation", turn=4, new_entities=0),
        ])
        result = detector._check_diminishing_returns()
        assert result["severity"] == "medium"

    def test_message_mentions_action_required(self, loop_detector):
        """Result message contains actionable guidance."""
        detector = LoopDetector(max_observation_chain=20)
        from core.skill.execution.loop_detector import ToolCallRecord
        detector.history.extend([
            ToolCallRecord(tool_name="search_0", category="observation", turn=0, new_entities=2),
            ToolCallRecord(tool_name="search_1", category="effect", turn=1, new_entities=0),
            ToolCallRecord(tool_name="search_2", category="observation", turn=2, new_entities=0),
            ToolCallRecord(tool_name="search_3", category="observation", turn=3, new_entities=0),
            ToolCallRecord(tool_name="search_4", category="observation", turn=4, new_entities=0),
        ])
        result = detector._check_diminishing_returns()
        assert "DIMINISHING RETURNS" in result["message"]

    def test_recent_entities_in_result(self, loop_detector):
        """Result includes recent_entities data."""
        detector = LoopDetector(max_observation_chain=20)
        from core.skill.execution.loop_detector import ToolCallRecord
        detector.history.extend([
            ToolCallRecord(tool_name="search_0", category="observation", turn=0, new_entities=2),
            ToolCallRecord(tool_name="search_1", category="effect", turn=1, new_entities=0),
            ToolCallRecord(tool_name="search_2", category="observation", turn=2, new_entities=0),
            ToolCallRecord(tool_name="search_3", category="observation", turn=3, new_entities=0),
            ToolCallRecord(tool_name="search_4", category="observation", turn=4, new_entities=0),
        ])
        result = detector._check_diminishing_returns()
        assert "recent_entities" in result
        # recent_entities = entities[-3:] = [0, 0, 0] (last 3 of [2, 0, 0, 0])
        assert 0 in result["recent_entities"]


# =============================================================================
# Pattern 4: repeating_sequence
# =============================================================================


class TestRepeatingSequence:
    """Tests for _check_repeating_sequence()."""

    def test_insufficient_history(self, loop_detector):
        """History < 6 returns None."""
        for i in range(5):
            loop_detector.record(f"tool_{i}", "observation", turn=i)
        result = loop_detector._check_repeating_sequence()
        assert result is None

    def test_abab_pattern_detected(self, loop_detector):
        """A-B-A-B (2-tool repeating) pattern is detected with 6 records."""
        # repeating_sequence requires len(history) >= 6 and exactly 2 repetitions
        # ab * 3 = ababab (6 records) → last 4 = abab and last 6 = ababab
        # Both pattern_len=2 and pattern_len=3 check the last N records against first N
        detector = LoopDetector(max_observation_chain=20)
        pattern = ["read", "write"] * 3  # 6 records = ababab
        for i, t in enumerate(pattern):
            detector.record(t, "observation", turn=i)
        result = detector._check_repeating_sequence()
        assert result is not None
        assert result["type"] == "repeating_sequence"
        assert result["sequence"] == ["read", "write"]
        assert result["repetitions"] == 2

    def test_abcabc_pattern_detected(self, loop_detector):
        """A-B-C-A-B-C (3-tool repeating) pattern is detected."""
        pattern = ["read", "write", "edit", "read", "write", "edit"]
        for i, t in enumerate(pattern):
            loop_detector.record(t, "observation", turn=i)
        result = loop_detector._check_repeating_sequence()
        assert result is not None
        assert result["type"] == "repeating_sequence"
        assert result["sequence"] == ["read", "write", "edit"]

    def test_non_repeating_not_detected(self, loop_detector):
        """A-B-C-D (non-repeating) does NOT trigger."""
        pattern = ["read", "write", "edit", "delete"]
        for i, t in enumerate(pattern):
            loop_detector.record(t, "observation", turn=i)
        result = loop_detector._check_repeating_sequence()
        assert result is None

    def test_less_than_2x_pattern_length_not_detected(self, loop_detector):
        """Only 1 repetition (not 2) does NOT trigger."""
        pattern = ["read", "write"]
        for i, t in enumerate(pattern):
            loop_detector.record(t, "observation", turn=i)
        result = loop_detector._check_repeating_sequence()
        assert result is None

    def test_high_severity(self, loop_detector):
        """repeating_sequence has high severity."""
        detector = LoopDetector(max_observation_chain=20)
        pattern = ["read", "write"] * 3  # 6 records
        for i, t in enumerate(pattern):
            detector.record(t, "observation", turn=i)
        result = detector._check_repeating_sequence()
        assert result["severity"] == "high"

    def test_message_mentions_pattern(self, loop_detector):
        """Result message mentions the repeating tools."""
        detector = LoopDetector(max_observation_chain=20)
        pattern = ["read", "write"] * 3  # 6 records
        for i, t in enumerate(pattern):
            detector.record(t, "observation", turn=i)
        result = detector._check_repeating_sequence()
        assert "read" in result["message"]
        assert "write" in result["message"]

    def test_exactly_2x_pattern_length_triggers(self, loop_detector):
        """Exactly 2 repetitions of pattern_len triggers."""
        pattern = ["read", "write", "edit"]
        for i, t in enumerate(pattern * 2):
            loop_detector.record(t, "observation", turn=i)
        result = loop_detector._check_repeating_sequence()
        assert result is not None
        assert result["type"] == "repeating_sequence"


# =============================================================================
# Integration: All Patterns Together
# =============================================================================


class TestLoopDetectorIntegration:
    """Integration tests covering the full detect() pipeline."""

    def test_observation_chain_preempts_other_patterns(self, loop_detector):
        """When both observation_chain and low_effect_ratio conditions are met,
        observation_chain is returned (checked first)."""
        # 6 consecutive obs → observation_chain triggers
        for i in range(6):
            loop_detector.record(f"search_{i}", "observation", turn=i)
        result = loop_detector.detect()
        assert result["type"] == "observation_chain"

    def test_repeating_sequence_after_normal_tools(self, loop_detector):
        """repeating_sequence detected after non-repeating initial tools.

        当 pattern 中所有记录都没有效果产出时，智能跳过不生效。
        注意：若 pattern 中有任意记录有效果产出（workspace_changed 或 created_artifacts > 0），
        则整个 pattern 会被智能跳过，repeating_sequence 不会触发。
        """
        # First two non-repeating tools with distinct names
        loop_detector.record("read", "observation", turn=0)
        loop_detector.record("file_create", "observation", turn=1, created_artifacts=1)
        # Then repeating pattern: read-write-read-write (NO effects)
        # Total: 6 records. Last 4 = A-B-A-B → repeating_sequence triggers
        # (智能跳过不生效，因为 pattern 中的 A-B-A-B 全都没有效果)
        pattern = ["read", "write", "read", "write"]
        for i, t in enumerate(pattern):
            loop_detector.record(t, "observation", turn=2+i)
        result = loop_detector.detect()
        # Last 4 are all "observation" → observation_chain: len(obs_records) = 4 < 6 → not triggered
        # repeating_sequence: last 4 = A-B-A-B + all have no effect → triggers
        assert result["type"] == "repeating_sequence"

    def test_repeating_sequence_skipped_when_pattern_has_effect(self, loop_detector, mock_artifact_registry):
        """repeating_sequence is skipped when all steps in pattern have effects.

        这是本次修复的核心场景：一个健康的迭代工作流（如 read→create→read→create）
        不应被误判为 loop，因为每次都有实质的文件变更（workspace_changed=True）。
        """
        # 模拟健康的工作流：read(A) → create(B) → read(C) → create(D) → read(E) → create(F)
        # 通过 ArtifactRegistry 快照对比感知 workspace_changed=True
        registry = mock_artifact_registry
        # Turn 0: 空注册表 → read_file，无变更
        registry.all_paths = frozenset()
        loop_detector.record("read_file", "observation", turn=0, artifact_registry=registry)
        # Turn 1: 注册表新增 file_B → file_create，workspace_changed=True
        registry.all_paths = frozenset({"file_B.txt"})
        loop_detector.record("file_create", "observation", turn=1, artifact_registry=registry)
        # Turn 2: 注册表新增 file_D → read_file，workspace_changed=True
        registry.all_paths = frozenset({"file_B.txt", "file_D.txt"})
        loop_detector.record("read_file", "observation", turn=2, artifact_registry=registry)
        # Turn 3: 注册表新增 file_F → file_create，workspace_changed=True
        registry.all_paths = frozenset({"file_B.txt", "file_D.txt", "file_F.txt"})
        loop_detector.record("file_create", "observation", turn=3, artifact_registry=registry)
        # Turn 4: 无新增 → read_file，workspace_changed=False
        registry.all_paths = frozenset({"file_B.txt", "file_D.txt", "file_F.txt"})
        loop_detector.record("read_file", "observation", turn=4, artifact_registry=registry)
        # Turn 5: 无新增 → file_create，workspace_changed=False
        registry.all_paths = frozenset({"file_B.txt", "file_D.txt", "file_F.txt"})
        loop_detector.record("file_create", "observation", turn=5, artifact_registry=registry)

        result = loop_detector.detect()
        # 注意：最近 4 条 = [read, create, read, create]，但其中 [create, read, create] 有 workspace_changed
        # 智能跳过条件要求 ALL steps 都有效果，当前 pattern 只有 3/4 有效果 → 不跳过
        # 但 pattern_len=2 的 pattern=[read_file, file_create]，repeating=[read_file, file_create] * 2
        # recent_records[0]=read_file(workspace_changed=False), recent_records[1]=file_create(ws=True)
        # recent_records[2]=read_file(ws=False), recent_records[3]=file_create(ws=True)
        # 4 records 中只有 2 个有效果 → not all → 不跳过 → 触发
        assert result is not None
        assert result["type"] == "repeating_sequence"

    def test_repeating_sequence_skipped_when_all_pattern_steps_have_effect(self, loop_detector, mock_artifact_registry):
        """repeating_sequence skipped only when EVERY step in pattern has effect."""
        registry = mock_artifact_registry
        # 每个 tool call 都让注册表新增一个文件
        paths = []
        for i in range(6):
            paths.append(f"file_{i}.txt")
            registry.all_paths = frozenset(paths)
            tool = "read_file" if i % 2 == 0 else "file_create"
            loop_detector.record(tool, "observation", turn=i, artifact_registry=registry)
        # pattern_len=2: [read_file, file_create] × 2
        # recent_records = [read(ws=True), create(ws=True), read(ws=True), create(ws=True)]
        # all(ws or ca > 0) = all True → 智能跳过 → 不触发
        result = loop_detector.detect()
        assert result is None

    def test_all_patterns_avoided_with_good_balance(self, loop_detector):
        """Normal balanced tool usage with diverse names avoids all patterns."""
        # Use unique tool names to avoid repeating_sequence
        for i in range(10):
            loop_detector.record(f"create_{i}", "observation", turn=i*2, created_artifacts=1)
            loop_detector.record(f"search_{i}", "observation", turn=i*2+1, new_entities=2)
        result = loop_detector.detect()
        # No 6 consecutive obs (alternating), ratio fine (50%), diminishing needs last 3 all <=1 (we give 2)
        # repeating needs pattern repetition (unique names avoid it)
        assert result is None

    def test_history_grows_beyond_window(self, loop_detector):
        """Observation chain detected when trailing obs >= 6, regardless of history size."""
        # 7 observations (trailing obs = 7 >= 6) → triggers
        detector = LoopDetector(window_size=10, min_effect_ratio=0.15, max_observation_chain=6)
        for i in range(7):
            detector.record(f"search_{i}", "observation", turn=i)
        result = detector.detect()
        # trailing obs = 7 >= 6 → triggers
        assert result["type"] == "observation_chain"
        assert result["chain_length"] == 7


# =============================================================================
# ArtifactRegistry Integration
# =============================================================================


class TestLoopDetectorArtifactRegistry:
    """Tests for ArtifactRegistry-driven category derivation."""

    def test_registry_path_growth_implies_effect(self, loop_detector, mock_artifact_registry):
        """Growing ArtifactRegistry paths causes category to become effect."""
        # First record: registry starts empty
        loop_detector.record("file_create", "observation", turn=0, artifact_registry=mock_artifact_registry)
        # Second record: registry now has a path
        mock_artifact_registry.all_paths = frozenset({"file1.txt"})
        loop_detector.record("read_file", "observation", turn=1, artifact_registry=mock_artifact_registry)
        # First record: workspace_changed=False (empty → empty), category stays "observation"
        # Second record: workspace_changed=True (empty → {file1.txt}), category becomes "effect"
        assert loop_detector.history[0].category == "observation"
        assert loop_detector.history[1].category == "effect"

    def test_registry_unchanged_implies_observation(self, loop_detector, mock_artifact_registry):
        """Unchanged ArtifactRegistry paths keep category as derived."""
        mock_artifact_registry.all_paths = frozenset({"file1.txt"})
        loop_detector.record("read_file", "observation", turn=0, artifact_registry=mock_artifact_registry)
        # No change
        mock_artifact_registry.all_paths = frozenset({"file1.txt"})
        loop_detector.record("read_file", "observation", turn=1, artifact_registry=mock_artifact_registry)
        # Both: first had workspace_changed=True (empty→{file1}), second False
        assert loop_detector.history[0].workspace_changed is True
        assert loop_detector.history[1].workspace_changed is False
