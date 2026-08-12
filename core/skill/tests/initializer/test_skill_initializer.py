# SPDX-License-Identifier: Apache-2.0
"""test_skill_initializer.py - SkillInitializer 测试

测试 SkillInitializer 初始化 skill 系统的功能。
"""

import pytest
from pathlib import Path

from core.skill.initializer import SkillInitializer


class TestSkillInitializer:
    """SkillInitializer 测试类"""

    @pytest.fixture
    def initializer(self, skill_config):
        """创建 SkillInitializer 实例"""
        return SkillInitializer(skill_config)

    @pytest.mark.asyncio
    async def test_initializer_initialization(self, initializer, skill_config):
        """测试初始化器初始化"""
        assert initializer is not None
        assert initializer._config == skill_config

    @pytest.mark.asyncio
    async def test_sync_builtin_skills(self, initializer):
        """测试同步 builtin skills"""
        synced = initializer.sync_builtin_skills()
        assert isinstance(synced, list)

    @pytest.mark.asyncio
    async def test_initialize_full_flow(self, initializer, skill_config):
        """测试完整初始化流程"""
        from core.skill.store import SkillStorage
        from core.skill.registry import SkillRegistry

        skills_dir = Path(skill_config.skills_dir)
        skills_dir.mkdir(parents=True, exist_ok=True)
        store = SkillStorage(skills_dir, SkillRegistry())
        await store.init()

        try:
            result = await initializer.initialize(sync_builtin=True)

            assert "builtin_synced" in result
            assert isinstance(result["builtin_synced"], list)

        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_initialize_without_sync(self, initializer, skill_config):
        """测试不执行同步的初始化"""
        from core.skill.store import SkillStorage
        from core.skill.registry import SkillRegistry

        skills_dir = Path(skill_config.skills_dir)
        skills_dir.mkdir(parents=True, exist_ok=True)
        store = SkillStorage(skills_dir, SkillRegistry())
        await store.init()

        try:
            result = await initializer.initialize(sync_builtin=False)
            assert result["builtin_synced"] == []

        finally:
            await store.close()
