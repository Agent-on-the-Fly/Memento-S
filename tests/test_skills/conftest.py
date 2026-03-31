"""test_skills 根 conftest - 所有 skill 测试共享的 fixtures

提供真实配置和常用 fixtures，不使用 mock。
所有路径都从 g_config 读取，确保测试在真实环境中运行。
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from pathlib import Path

from middleware.config import ConfigManager, g_config
from core.skill.config import SkillConfig


@pytest.fixture(scope="session")
def test_config():
    """加载测试配置（全局单例）"""
    config_manager = ConfigManager()
    config_manager.load()
    # 确保 g_config 已加载
    if not g_config._config:
        g_config._config = config_manager._config
    return config_manager


@pytest.fixture(scope="session")
def skill_config(test_config) -> SkillConfig:
    """SkillConfig 实例（从全局配置创建）"""
    return SkillConfig.from_global_config()


@pytest.fixture(scope="session")
def skills_dir(skill_config) -> Path:
    """Skills 目录路径"""
    return Path(skill_config.skills_dir)


@pytest.fixture(scope="session")
def builtin_skills_dir(skill_config) -> Path:
    """Builtin skills 目录路径"""
    return Path(skill_config.builtin_skills_dir)


@pytest.fixture(scope="session")
def workspace_dir(skill_config) -> Path:
    """Workspace 目录路径"""
    return Path(skill_config.workspace_dir)


@pytest.fixture(scope="session")
def db_path(skill_config) -> Path:
    """数据库文件路径"""
    return skill_config.db_path


@pytest.fixture
def sample_skill_data():
    """示例 skill 数据（用于测试 Skill 模型）"""
    return {
        "name": "test_skill",
        "description": "A test skill for unit testing",
        "content": """---
name: test_skill
description: A test skill for unit testing
metadata:
  function_name: test_skill
  dependencies:
    - pytest
---

# Test Skill

This is a test skill content.
""",
        "dependencies": ["pytest"],
    }


@pytest_asyncio.fixture
async def db_manager(test_config):
    """数据库管理器实例"""
    from middleware.storage.core.engine import get_db_manager
    from middleware.storage.models import Base

    db_path = g_config.get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db_manager = get_db_manager()
    db_url = f"sqlite+aiosqlite:///{db_path}"

    await db_manager.init(db_url)

    # 创建表
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield db_manager

    await db_manager.dispose()
