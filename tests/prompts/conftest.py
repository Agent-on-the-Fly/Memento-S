# SPDX-License-Identifier: Apache-2.0
"""Pytest 配置文件 — tests/prompts/"""

import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="session", autouse=True)
def _load_config():
    from middleware.config import g_config
    if not g_config.is_loaded():
        g_config.load()
    yield


@pytest.fixture(scope="session")
def _sandbox_temp_base():
    base = os.path.join(_ROOT, ".pytest_temp")
    os.makedirs(base, exist_ok=True)
    return base


@pytest.fixture
def tmp_session_dir(_sandbox_temp_base):
    import uuid
    d = os.path.join(_sandbox_temp_base, f"session_{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture
def tmp_data_dir(_sandbox_temp_base):
    import uuid
    d = os.path.join(_sandbox_temp_base, f"data_{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture
def mock_g_config_paths(tmp_session_dir, tmp_data_dir):
    mock_paths = MagicMock()
    mock_paths.context_dir = Path(tmp_session_dir)
    mock_paths.memory_dir = Path(tmp_data_dir)
    mock_paths.workspace_dir = Path(_ROOT) / "workspace"
    mock_llm_profile = MagicMock()
    mock_llm_profile.model = "gpt-4o"
    mock_llm_profile.input_budget = 100000
    mock_llm_profile.max_tokens = 4096
    mock_llm = MagicMock()
    mock_llm.current_profile = mock_llm_profile
    with patch("core.context.context_manager.g_config") as mock_g:
        mock_g.paths = mock_paths
        mock_g.llm = mock_llm
        yield mock_g


@pytest.fixture
def mock_session_ctx(tmp_session_dir):
    """Create a SessionContext for tests in this directory."""
    from core.context.session_context import SessionContext
    return SessionContext.create("test-session", base_dir=Path(tmp_session_dir))


@pytest.fixture(autouse=True)
def _auto_enable_verbose():
    os.environ.setdefault("MEMENTO_S_LOG_VERBOSE", "1")
    yield
