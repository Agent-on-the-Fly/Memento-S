# SPDX-License-Identifier: Apache-2.0
"""store 测试共享 fixtures

提供所有存储类的 fixtures，使用真实配置创建。
所有路径都从 g_config 读取，不使用临时目录。
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from pathlib import Path

from middleware.config import ConfigManager
from shared.schema import SkillConfig


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
    if not g_config._runtime_config:
        g_config._runtime_config = test_config._runtime_config
    return SkillConfig.from_global_config()


@pytest.fixture(scope="session")
def skills_dir(skill_config):
    """从 g_config 获取 skills 目录"""
    return Path(skill_config.skills_dir)


@pytest_asyncio.fixture
async def skill_storage(skills_dir):
    """SkillStorage 实例 - 使用真实路径"""
    from core.skill.store import SkillStorage
    from core.skill.registry import SkillRegistry

    skills_dir.mkdir(parents=True, exist_ok=True)
    store = SkillStorage(skills_dir, SkillRegistry())
    await store.init()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def file_storage(skills_dir):
    """FileStorage 别名（兼容旧测试）- 实际指向 SkillStorage"""
    return (await pytest.importorskip("conftest").skill_storage(skills_dir))


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
