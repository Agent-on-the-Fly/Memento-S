"""embedding 测试共享 fixtures

提供 EmbeddingGenerator 测试所需的 fixtures。
"""

from __future__ import annotations

import pytest

from middleware.config import ConfigManager, g_config
from middleware.llm.embedding_client import EmbeddingClient


@pytest.fixture(scope="session")
def test_config():
    """加载测试配置并确保 g_config 已加载"""
    config_manager = ConfigManager()
    config_manager.load()
    # 确保 g_config 也被加载
    if not g_config._config:
        g_config._config = config_manager._config
    return config_manager


@pytest.fixture
def embedding_client(test_config):
    """EmbeddingClient 实例（可选，依赖配置）"""
    cfg = test_config.skills.retrieval

    if not cfg.embedding_base_url:
        pytest.skip("embedding_base_url not configured")

    # 使用 from_config 创建 client
    client = EmbeddingClient.from_config()

    yield client


@pytest.fixture
def generator(skill_config):
    """EmbeddingGenerator 实例"""
    from core.skill.embedding import EmbeddingGenerator

    gen = EmbeddingGenerator.from_config(skill_config)
    yield gen
