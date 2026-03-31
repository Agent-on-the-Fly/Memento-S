"""test_skill_initializer.py - SkillInitializer 测试

测试 SkillInitializer 初始化 skill 系统的功能。
"""

import pytest
from pathlib import Path

from core.skill.initializer import SkillInitializer
from core.skill.store import SkillStore


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
        # 注意：这个测试可能需要实际 builtin skills 目录存在
        synced = initializer.sync_builtin_skills()

        # 返回应该是列表
        assert isinstance(synced, list)

    @pytest.mark.asyncio
    async def test_sync_workspace_skills(self, initializer):
        """测试同步 workspace skills"""
        synced = initializer.sync_workspace_skills()

        # 返回应该是列表
        assert isinstance(synced, list)

    @pytest.mark.asyncio
    async def test_initialize_full_flow(self, initializer, skill_config):
        """测试完整初始化流程"""
        # 创建 SkillStore
        store = await SkillStore.from_config(skill_config)

        try:
            # 执行初始化
            result = await initializer.initialize(
                store, sync_builtin=True, sync_workspace=True
            )

            # 验证返回结构
            assert "builtin_synced" in result
            assert "workspace_synced" in result
            assert "refreshed" in result
            assert "db_synced" in result
            assert "orphans_cleaned" in result

            # 验证类型
            assert isinstance(result["builtin_synced"], list)
            assert isinstance(result["workspace_synced"], list)
            assert isinstance(result["refreshed"], int)
            assert isinstance(result["db_synced"], int)
            assert isinstance(result["orphans_cleaned"], list)

        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_initialize_without_sync(self, initializer, skill_config):
        """测试不执行同步的初始化"""
        store = await SkillStore.from_config(skill_config)

        try:
            result = await initializer.initialize(
                store, sync_builtin=False, sync_workspace=False
            )

            # 不应该同步任何东西
            assert result["builtin_synced"] == []
            assert result["workspace_synced"] == []

        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_resolve_workspace_skills_root(self, initializer):
        """测试解析 workspace skills 根目录"""
        # 这个方法可能会返回 None（如果没有 workspace/skills 目录）
        root = initializer._resolve_workspace_skills_root()

        # 返回应该是 Path 或 None
        assert root is None or isinstance(root, Path)
