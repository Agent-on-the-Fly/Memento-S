# SPDX-License-Identifier: Apache-2.0
"""Conftest for core/skill/tests/execution — execution layer test fixtures."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from core.skill.execution.loop_detector import LoopDetector


@pytest.fixture
def loop_detector():
    """LoopDetector with default config."""
    return LoopDetector(
        max_observation_chain=6,
        min_effect_ratio=0.15,
        window_size=10,
    )


@pytest.fixture
def loop_detector_small_window():
    """LoopDetector with small window for edge case testing."""
    return LoopDetector(
        max_observation_chain=4,
        min_effect_ratio=0.2,
        window_size=5,
    )


@pytest.fixture
def mock_artifact_registry():
    """Mock ArtifactRegistry for loop detection tests."""
    registry = MagicMock()
    registry.all_paths = []
    registry.artifacts = []
    return registry


def make_record(tool_name: str, category: str, turn: int = 0,
                new_entities: int = 0, created_artifacts: int = 0) -> dict:
    """Helper to create a record dict for testing."""
    return {
        "tool_name": tool_name,
        "category": category,
        "turn": turn,
        "new_entities": new_entities,
        "created_artifacts": created_artifacts,
    }