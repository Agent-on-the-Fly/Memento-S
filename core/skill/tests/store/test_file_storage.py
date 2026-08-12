# SPDX-License-Identifier: Apache-2.0
"""test_file_storage.py - FileStorage 磁盘读写测试

测试 SkillStorage 的磁盘文件读写功能（save/load/delete/list_names）。
"""

from __future__ import annotations

import pytest

from core.skill.schema import Skill
from core.skill.store import SkillStorage


class TestSkillStorageDisk:
    """SkillStorage 磁盘读写测试类"""

    @pytest.mark.asyncio
    async def test_save_and_load(self, skill_storage, sample_skill):
        """测试保存和加载 skill"""
        # 保存
        await skill_storage.save(sample_skill.name, sample_skill)

        # 加载
        loaded = await skill_storage.load(sample_skill.name)

        assert loaded is not None
        assert loaded.name == sample_skill.name
        assert loaded.description == sample_skill.description

    @pytest.mark.asyncio
    async def test_save_creates_skill_dir(self, skill_storage, sample_skill):
        """测试保存 skill 创建目录"""
        await skill_storage.save(sample_skill.name, sample_skill)

        # 检查 skill 目录
        skill_dir = skill_storage._skills_dir / "test-skill"
        skill_md = skill_dir / "SKILL.md"

        assert skill_dir.exists()
        assert skill_md.exists()
        # 验证内容包含 skill 名称
        content = skill_md.read_text()
        assert "test_skill" in content or "test-skill" in content

    @pytest.mark.asyncio
    async def test_load_not_found(self, skill_storage):
        """测试加载不存在的 skill"""
        result = await skill_storage.load("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, skill_storage, sample_skill):
        """测试删除 skill"""
        # 先保存
        await skill_storage.save(sample_skill.name, sample_skill)

        # 删除
        result = await skill_storage.delete(sample_skill.name)

        assert result is True
        assert await skill_storage.load(sample_skill.name) is None

    @pytest.mark.asyncio
    async def test_delete_not_found(self, skill_storage):
        """测试删除不存在的 skill"""
        result = await skill_storage.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_names(self, skill_storage, sample_skill, sample_skill2):
        """测试列出所有 skill 名称"""
        # 保存两个 skills
        await skill_storage.save(sample_skill.name, sample_skill)
        await skill_storage.save(sample_skill2.name, sample_skill2)

        # 列出
        names = await skill_storage.list_names()

        # 使用真实目录，可能已有其他 skills，只验证保存的 skills 存在
        assert sample_skill.name in names
        assert sample_skill2.name in names
        assert len(names) >= 2

    @pytest.mark.asyncio
    async def test_list_names_returns_set(self, skill_storage):
        """测试目录列出返回类型"""
        names = await skill_storage.list_names()
        # 使用真实目录，返回的至少是集合类型
        assert isinstance(names, set)

    @pytest.mark.asyncio
    async def test_kebab_case_conversion(self, skill_storage):
        """测试 kebab-case 转换"""
        content = """---
name: Test_Skill_Name
description: Test
metadata:
  function_name: Test_Skill_Name
---

# Test
"""

        skill = Skill(
            name="Test_Skill_Name",
            description="Test",
            content=content,
        )

        await skill_storage.save(skill.name, skill)

        # 检查目录名是 kebab-case
        expected_dir = skill_storage._skills_dir / "test-skill-name"
        assert expected_dir.exists()

    @pytest.mark.asyncio
    async def test_skill_with_references(self, skill_storage):
        """测试带 references 的 skill"""
        content = """---
name: ref_test
description: Test with refs
metadata:
  function_name: ref_test
---

# Test
"""

        skill = Skill(
            name="ref_test",
            description="Test with refs",
            content=content,
            references={"ref.md": "Reference content"},
        )

        await skill_storage.save(skill.name, skill)

        # 加载并检查 references
        loaded = await skill_storage.load("ref_test")
        assert loaded is not None
        assert "ref.md" in loaded.references
        assert loaded.references["ref.md"] == "Reference content"

    @pytest.mark.asyncio
    async def test_close(self, skill_storage):
        """测试关闭 - 不应抛出异常"""
        await skill_storage.close()
