"""retrieval 测试共享 fixtures

提供 RecallTestConfig 和所有 recall 策略的 fixtures。
"""

from __future__ import annotations

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
def local_file_recall(skills_dir):
    """LocalFileRecall 实例"""
    from core.skill.retrieval import LocalFileRecall

    return LocalFileRecall(skills_dir)


@pytest.fixture
def local_db_recall(db_path):
    """LocalDbRecall 实例（可选，依赖 sqlite-vec）"""
    try:
        from core.skill.retrieval import LocalDbRecall

        return LocalDbRecall(db_path, embedding_client=None)
    except ImportError:
        pytest.skip("sqlite-vec not installed")


@pytest.fixture
def remote_recall(cloud_url):
    """RemoteRecall 实例（可选，依赖网络）"""
    if not cloud_url:
        pytest.skip("cloud_catalog_url not configured")

    from core.skill.retrieval import RemoteRecall

    recall = RemoteRecall(cloud_url)

    if not recall.is_available():
        pytest.skip("Remote service not available")

    yield recall
    recall.close()


@pytest.fixture
def multi_recall(skills_dir, db_path, cloud_url):
    """MultiRecall 实例（组合所有可用策略）"""
    from core.skill.retrieval import MultiRecall, LocalFileRecall, RemoteRecall

    recalls = []

    # LocalFileRecall
    file_recall = LocalFileRecall(skills_dir)
    if file_recall.is_available():
        recalls.append(file_recall)

    # LocalDbRecall（可选）
    try:
        from core.skill.retrieval import LocalDbRecall

        db_recall = LocalDbRecall(db_path, embedding_client=None)
        if db_recall.is_available():
            recalls.append(db_recall)
    except ImportError:
        pass

    # RemoteRecall（可选）
    if cloud_url:
        remote_recall = RemoteRecall(cloud_url)
        if remote_recall.is_available():
            recalls.append(remote_recall)

    multi = MultiRecall(recalls)
    yield multi

    # 清理
    import asyncio

    asyncio.run(multi.close())
