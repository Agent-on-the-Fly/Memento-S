# SPDX-License-Identifier: Apache-2.0
"""retrieval 测试共享 fixtures

提供 RecallTestConfig 和所有 recall 策略的 fixtures。
"""

from __future__ import annotations

import asyncio

import pytest
from pathlib import Path

from middleware.config import ConfigManager


@pytest.fixture(scope="session")
def test_config():
    """加载测试配置"""
    config_manager = ConfigManager()
    config_manager.load()
    return config_manager


@pytest.fixture
def skills_dir(test_config):
    """技能目录路径"""
    return test_config.get_skills_path()


@pytest.fixture
def db_path(test_config):
    """数据库路径"""
    return test_config.get_db_path()


@pytest.fixture
def cloud_url(test_config):
    """云端服务 URL"""
    return test_config.skills.cloud_catalog_url


@pytest.fixture
def local_recall(skills_dir):
    """LocalRecall 实例"""
    from core.skill.retrieval import LocalRecall

    return LocalRecall(skills_dir)


@pytest.fixture
def remote_recall(cloud_url):
    """RemoteRecall 实例（可选，依赖网络）"""
    if not cloud_url:
        pytest.skip("cloud_catalog_url not configured")

    from core.skill.retrieval import RemoteRecall

    recall = RemoteRecall(cloud_url)
    asyncio.run(recall._health_check())

    if not recall.is_available():
        pytest.skip("Remote service not available")

    yield recall
    asyncio.run(recall.close())


@pytest.fixture
def multi_recall(skills_dir, cloud_url):
    """MultiRecall 实例（组合所有可用策略）"""
    from core.skill.retrieval import MultiRecall, LocalRecall, RemoteRecall

    recalls = []

    # LocalRecall
    local = LocalRecall(skills_dir)
    if local.is_available():
        recalls.append(local)

    # RemoteRecall（可选）
    if cloud_url:
        remote = RemoteRecall(cloud_url)
        asyncio.run(remote._health_check())
        if remote.is_available():
            recalls.append(remote)

    multi = MultiRecall(recalls)
    yield multi

    # 清理
    asyncio.run(multi.close())

