# SPDX-License-Identifier: Apache-2.0
"""conftest.py — Fixtures for infra/context tests.

使用真实路径 ~/memento_s/context 进行测试，验证实际文件写入。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from infra.context.factory import ContextFactoryConfig
from middleware.config import g_config


@pytest.fixture(scope="session", autouse=True)
def _ensure_config():
    """Session-scoped fixture: load g_config once for all integration tests."""
    if not g_config.is_loaded():
        g_config.load()


@pytest.fixture(scope="session")
def real_ctx_root() -> Path:
    """Session-scoped real context root for tests.
    
    从 g_config.paths.context_dir 获取基础路径，再在其下创建测试子目录。
    """
    context_base = Path(g_config.paths.context_dir)
    context_base.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    root = context_base / ".test_context" / "test_runs" / today
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def ctx_cfg() -> ContextFactoryConfig:
    """Default context config for all tests."""
    return ContextFactoryConfig()


@pytest.fixture
def tmp_ctx_dir(tmp_path: Path) -> Path:
    """Temp directory for context storage."""
    d = tmp_path / "context"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def real_ctx_dir(real_ctx_root: Path, request) -> Path:
    """Real directory for context tests using real paths."""
    test_name = request.node.name.replace("[", "_").replace("]", "_").replace("/", "_")
    d = real_ctx_root / f"ctx_{test_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def mock_history_loader():
    """Mock history loader returning sample messages."""
    async def loader(current_message: str | None = None):
        return [
            {
                "role": "user",
                "content": "Hello, can you help me with coding?",
                "_seq": 1,
            },
            {
                "role": "assistant",
                "content": "Of course! I'd be happy to help with your coding task.",
                "_seq": 2,
            },
        ]

    return AsyncMock(side_effect=loader)


@pytest.fixture
def mock_embedding_client():
    """Mock embedding client."""
    async def embed(query: str):
        return [0.1] * 384

    client = AsyncMock()
    client.embed_query = AsyncMock(side_effect=embed)
    return client


@pytest.fixture
def mock_vector_storage():
    """Mock vector storage."""
    storage = AsyncMock()
    storage.search = AsyncMock(return_value=[])
    return storage
