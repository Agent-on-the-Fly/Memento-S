# SPDX-License-Identifier: Apache-2.0
"""Tests for infra/context/factory.py — ContextFactoryConfig, create_context."""
from __future__ import annotations

from pathlib import Path

import pytest

from infra.context.base import ContextProvider
from infra.context.factory import ContextFactoryConfig, create_context


class TestContextFactoryConfig:
    """Test ContextFactoryConfig dataclass."""

    def test_defaults(self):
        cfg = ContextFactoryConfig()
        assert cfg.session_id == ""
        assert cfg.history_load_limit == 20
        assert cfg.recent_rounds_keep == 3
        assert cfg.history_budget_ratio == 0.5
        assert cfg.summary_ratio == 0.15
        assert cfg.persist_ratio == 0.15
        assert cfg.extract_ratio == 0.05
        assert cfg.preview_ratio == 0.005
        assert cfg.slim_ratio == 0.003
        assert cfg.microcompact_keep_recent == 5
        assert cfg.emergency_keep_tail == 6
        assert cfg.max_compact_failures == 3
        assert cfg.sm_compact_min_ratio == 0.02
        assert cfg.sm_compact_max_ratio == 0.08
        assert cfg.breaker_cooldown_s == 60.0
        assert cfg.sm_llm_update_interval == 5

    def test_custom_values(self):
        cfg = ContextFactoryConfig(
            session_id="session-001",
            sm_llm_update_interval=3,
        )
        assert cfg.session_id == "session-001"
        assert cfg.sm_llm_update_interval == 3

    def test_microcompact_tools_default(self):
        cfg = ContextFactoryConfig()
        assert "execute_skill" in cfg.microcompact_compactable_tools
        assert "search_skill" in cfg.microcompact_compactable_tools


class TestCreateContext:
    """Test create_context factory function."""

    def test_requires_session_id(self):
        cfg = ContextFactoryConfig()
        with pytest.raises(ValueError, match="session_id is required"):
            create_context(cfg)

    def test_creates_file_context_provider(
        self,
        tmp_path: Path,
        mock_history_loader,
    ):
        cfg = ContextFactoryConfig(
            session_id="test-session",
            session_dir=tmp_path / "session",
            data_dir=tmp_path,
        )
        provider = create_context(cfg)
        assert isinstance(provider, ContextProvider)

    def test_file_provider_with_memory(
        self,
        tmp_path: Path,
        mock_history_loader,
    ):
        cfg = ContextFactoryConfig(
            session_id="test-session-mem",
            session_dir=tmp_path / "session",
            data_dir=tmp_path,
        )
        provider = create_context(cfg)
        assert isinstance(provider, ContextProvider)
