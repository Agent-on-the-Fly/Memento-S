"""Fixtures for context module integration tests.

All fixtures use real objects — no unittest.mock.
ContextManager depends on g_config.paths.context_dir; we load real config
and override context_dir to a temp directory.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from core.context.schemas import ContextConfig
from core.context.scratchpad import Scratchpad
from core.context.manager import ContextManager
from middleware.config import g_config


def _ensure_config_loaded() -> None:
    """Load g_config if not already loaded."""
    if g_config._config is None:
        g_config.load()


@pytest.fixture
def ctx_cfg() -> ContextConfig:
    return ContextConfig()


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def date_dir(tmp_dir: Path) -> Path:
    today_str = datetime.now().strftime("%Y-%m-%d")
    d = tmp_dir / "context" / today_str
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def ctx_dir(tmp_dir: Path) -> Path:
    d = tmp_dir / "context"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def scratchpad(date_dir: Path) -> Scratchpad:
    return Scratchpad("test-session", date_dir)


@pytest.fixture
def context_manager(tmp_dir: Path, ctx_cfg: ContextConfig) -> ContextManager:
    """Create a real ContextManager pointing at a temp directory."""
    _ensure_config_loaded()
    original = g_config.paths.context_dir
    g_config.paths.context_dir = tmp_dir / "context"
    try:
        cm = ContextManager(session_id="test-session", config=ctx_cfg)
        yield cm
    finally:
        g_config.paths.context_dir = original
