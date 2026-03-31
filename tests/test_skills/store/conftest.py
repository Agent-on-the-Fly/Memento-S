"""store 测试共享 fixtures

提供所有存储类的 fixtures，使用真实配置创建。
所有路径都从 g_config 读取，不使用临时目录。
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from pathlib import Path

from middleware.config import ConfigManager
from core.skill.config import SkillConfig


@pytest.fixture(scope="session")
def test_config():
    """加载测试配置"""
    config_manager = ConfigManager()
    config_manager.load()
    return config_manager


@pytest.fixture(scope="session")
def skill_config(test_config):
    """SkillConfig 实例"""
    from middleware.config import g_config

    # 确保 g_config 已加载
    if not g_config._config:
        g_config._config = test_config._config
    return SkillConfig.from_global_config()


@pytest.fixture(scope="session")
def skills_dir(skill_config):
    """从 g_config 获取 skills 目录"""
    return Path(skill_config.skills_dir)


@pytest.fixture(scope="session")
def db_dir(skill_config):
    """从 g_config 获取数据库目录"""
    return skill_config.db_path.parent


@pytest_asyncio.fixture
async def file_storage(skills_dir):
    """FileStorage 实例 - 使用真实路径"""
    from core.skill.store import FileStorage

    # 确保目录存在
    skills_dir.mkdir(parents=True, exist_ok=True)

    storage = FileStorage(skills_dir)
    await storage.init()
    yield storage
    await storage.close()


@pytest_asyncio.fixture
async def skill_service(test_config):
    """SkillService 实例（可选，依赖数据库）"""
    try:
        from middleware.storage import SkillService
        from middleware.storage.core.engine import get_db_manager
        from middleware.config import g_config

        # 确保 g_config 已加载
        if not g_config._config:
            g_config._config = test_config._config

        # 获取数据库路径
        db_path = g_config.get_db_path()
        if not db_path:
            pytest.skip("Database path not configured")

        # 确保目录存在
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        db_manager = get_db_manager()
        db_url = f"sqlite+aiosqlite:///{db_path}"

        await db_manager.init(db_url)

        # 创建表
        from middleware.storage.models import Base

        async with db_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # 创建服务实例
        service = SkillService()
        yield service

        # 清理
        await db_manager.dispose()

    except Exception as e:
        print(f"Database setup failed: {e}")
        pytest.skip(f"Database not available: {e}")


@pytest_asyncio.fixture
async def db_storage(skill_service):
    """DBStorage 实例"""
    from core.skill.store import DBStorage

    if not skill_service:
        pytest.skip("Skill service not available")

    storage = DBStorage(skill_service)
    await storage.init()

    yield storage
    await storage.close()


@pytest.fixture
def embedding_client():
    """EmbeddingClient 实例（可选，依赖配置）"""
    from middleware.llm.embedding_client import EmbeddingClient
    from middleware.config import g_config

    cfg = g_config.skills.retrieval
    if not cfg.embedding_base_url:
        pytest.skip("embedding_base_url not configured")

    client = EmbeddingClient.from_config()
    yield client


@pytest_asyncio.fixture
async def vector_storage(db_dir, embedding_client):
    """VectorStorage 实例 - 使用真实数据库目录"""
    from core.skill.store import VectorStorage

    if not embedding_client:
        pytest.skip("Embedding client not available")

    # 确保目录存在
    db_dir.mkdir(parents=True, exist_ok=True)

    db_path = db_dir / "skill_embeddings.db"
    storage = VectorStorage(db_path=db_path, dimension=embedding_client.dimension)
    await storage.init()
    yield storage
    await storage.close()


@pytest_asyncio.fixture
async def skill_store(skills_dir, db_dir, skill_service, embedding_client):
    """完整的 SkillStore 实例 - 使用真实路径"""
    from core.skill.store import (
        SkillStore,
        FileStorage,
        DBStorage,
        VectorStorage,
    )

    # 确保依赖可用
    if not skill_service:
        pytest.skip("Skill service not available")
    if not embedding_client:
        pytest.skip("Embedding client not available")

    # 确保目录存在
    skills_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    # 文件存储
    file_storage = FileStorage(skills_dir)
    await file_storage.init()

    # DB 存储
    db_storage = DBStorage(skill_service)
    await db_storage.init()

    # 向量存储
    db_path = db_dir / "skill_embeddings.db"
    vector_storage = VectorStorage(
        db_path=db_path, dimension=embedding_client.dimension
    )
    await vector_storage.init()

    store = SkillStore(file_storage, db_storage, vector_storage)

    yield store

    await store.close()


@pytest.fixture
def sample_skill():
    """示例 Skill 对象 - 包含完整的 SKILL.md frontmatter"""
    from core.skill.schema import Skill

    content = """---
name: test_skill
description: A test skill for store testing
metadata:
  function_name: test_skill
  dependencies:
    - pytest
---

# Test Skill

This is a test skill content for testing store functionality.
"""

    return Skill(
        name="test_skill",
        description="A test skill for store testing",
        content=content,
        dependencies=["pytest"],
        files={"test.py": "print('hello')"},
    )


@pytest.fixture
def sample_skill2():
    """第二个示例 Skill 对象 - 包含完整的 SKILL.md frontmatter"""
    from core.skill.schema import Skill

    content = """---
name: another_test_skill
description: Another test skill
metadata:
  function_name: another_test_skill
---

# Another Test

This is another test skill.
"""

    return Skill(
        name="another_test_skill",
        description="Another test skill",
        content=content,
        dependencies=[],
        files={},
    )


# 清理 fixture - 测试完成后清理测试数据
@pytest.fixture(autouse=True)
def cleanup_test_skills(skills_dir, request):
    """测试完成后清理测试创建的 skills"""
    yield
    # 测试完成后可以在这里清理
    # 但保留数据有助于调试，所以暂时不自动清理
    pass
