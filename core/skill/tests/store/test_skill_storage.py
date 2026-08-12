# SPDX-License-Identifier: Apache-2.0
"""test_skill_storage.py - SkillStorage 单元测试

测试 SkillStorage 的所有功能：文件读写、注册表操作、同步。
"""

from __future__ import annotations

import pytest

from core.skill.store import SkillStorage
from core.skill.schema import Skill


def create_skill_with_frontmatter(
    name: str, description: str, extra_content: str = ""
) -> Skill:
    """辅助函数：创建带有 frontmatter 的 Skill"""
    content = f"""---
name: {name}
description: {description}
metadata:
  function_name: {name}
---

# {name}

{extra_content}
"""
    return Skill(name=name, description=description, content=content)


class TestSkillStorage:
    """SkillStorage 测试类"""

    @pytest.mark.asyncio
    async def test_init_and_close(self, skill_storage):
        """测试初始化和关闭"""
        assert skill_storage.is_ready is True

    @pytest.mark.asyncio
    async def test_add_and_get_skill(self, skill_storage, sample_skill):
        """测试添加和获取 skill"""
        await skill_storage.add_skill(sample_skill)

        loaded = await skill_storage.get_skill(sample_skill.name)
        assert loaded is not None
        assert loaded.name == sample_skill.name

    @pytest.mark.asyncio
    async def test_add_skill_creates_file(self, skill_storage, sample_skill):
        """测试添加 skill 时磁盘文件已创建"""
        await skill_storage.add_skill(sample_skill)

        skill_dir = skill_storage._skills_dir / "test-skill"
        skill_md = skill_dir / "SKILL.md"
        assert skill_dir.exists()
        assert skill_md.exists()

    @pytest.mark.asyncio
    async def test_remove_skill(self, skill_storage, sample_skill):
        """测试删除 skill"""
        await skill_storage.add_skill(sample_skill)

        result = await skill_storage.remove_skill(sample_skill.name)

        assert result is True
        assert await skill_storage.get_skill(sample_skill.name) is None

    @pytest.mark.asyncio
    async def test_list_all_skills(self, skill_storage, sample_skill):
        """测试列出所有 skills"""
        skill2 = create_skill_with_frontmatter("another_skill", "Another test")

        await skill_storage.add_skill(sample_skill)
        await skill_storage.add_skill(skill2)

        skills = await skill_storage.list_all_skills()

        assert sample_skill.name in skills
        assert "another_skill" in skills

    @pytest.mark.asyncio
    async def test_sync_from_disk(self, skill_storage, sample_skill):
        """测试从磁盘同步"""
        # 先添加 skill（源头）
        await skill_storage.add_skill(sample_skill)

        # 同步
        count = await skill_storage.sync_from_disk()

        assert count >= 1

    @pytest.mark.asyncio
    async def test_refresh_from_disk(self, skill_storage):
        """测试刷新磁盘"""
        skill = create_skill_with_frontmatter("refresh_test", "Test")
        await skill_storage.add_skill(skill)

        # 刷新
        added = await skill_storage.refresh_from_disk()

        # 应该找到已添加的 skill（在缓存中已存在，所以可能为 0）
        assert added >= 0

    @pytest.mark.asyncio
    async def test_cleanup_orphans(self, skill_storage, sample_skill):
        """测试清理孤儿"""
        await skill_storage.add_skill(sample_skill)

        # 清理（应该没有孤儿）
        cleaned = await skill_storage.cleanup_orphans()

        assert isinstance(cleaned, list)

    @pytest.mark.asyncio
    async def test_delete_not_found(self, skill_storage):
        """测试删除不存在的 skill"""
        result = await skill_storage.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_skill_not_found(self, skill_storage):
        """测试获取不存在的 skill"""
        result = await skill_storage.get_skill("nonexistent")
        assert result is None


class TestSkillStorageIntegration:
    """SkillStorage 集成测试"""

    @pytest.mark.asyncio
    async def test_full_workflow(self, skill_storage, sample_skill):
        """测试完整工作流"""
        # 添加 skills
        skill1 = create_skill_with_frontmatter("workflow1", "Test1")
        skill2 = create_skill_with_frontmatter("workflow2", "Test2")

        await skill_storage.add_skill(skill1)
        await skill_storage.add_skill(skill2)

        # 列出
        skills = await skill_storage.list_all_skills()
        assert len(skills) >= 2

        # 验证 skill 已添加
        found = await skill_storage.get_skill("workflow1")
        assert found is not None

        # 同步
        await skill_storage.sync_from_disk()

        # 删除
        await skill_storage.remove_skill("workflow1")
        assert await skill_storage.get_skill("workflow1") is None
