# SPDX-License-Identifier: Apache-2.0
"""conftest.py — Fixtures for infra/memory tests.

使用真实路径 ~/memento_s/context 进行测试，验证实际文件写入。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from middleware.config import g_config


def pytest_configure(config):
    """Ensure asyncio mode is set for async tests."""
    pass


@pytest.fixture(scope="session", autouse=True)
def _ensure_config():
    """Session-scoped fixture: load g_config once for all integration tests."""
    if not g_config.is_loaded():
        g_config.load()


@pytest.fixture(scope="session")
def real_memory_root() -> Path:
    """Session-scoped real memory root for tests.
    
    从 g_config.paths.context_dir 获取基础路径，再在其下创建测试子目录。
    """
    context_base = Path(g_config.paths.context_dir)
    context_base.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    root = context_base / ".test_context" / "test_runs" / today
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def tmp_memory_dir(tmp_path: Path) -> Path:
    """Temp directory for memory storage."""
    d = tmp_path / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def real_memory_dir(real_memory_root: Path, request) -> Path:
    """Real directory for memory tests using real paths."""
    test_name = request.node.name.replace("[", "_").replace("]", "_").replace("/", "_")
    d = real_memory_root / f"memory_{test_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tmp_session_dir(tmp_path: Path) -> Path:
    """Temp directory for session storage."""
    d = tmp_path / "session"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def real_session_dir(real_memory_root: Path, request) -> Path:
    """Real directory for session storage using real paths."""
    test_name = request.node.name.replace("[", "_").replace("]", "_").replace("/", "_")
    d = real_memory_root / f"session_{test_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d
