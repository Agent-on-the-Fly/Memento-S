"""file_storage.py 测试 - FileStorage"""

from __future__ import annotations

import pytest
from pathlib import Path

from core.skill.schema import Skill
from core.skill.store import FileStorage


class TestFileStorage:
    """FileStorage 测试类"""

    @pytest.mark.asyncio
    async def test_init(self, skills_dir):
        """测试初始化 - 使用 g_config 路径"""
        storage = FileStorage(skills_dir)
        await storage.init()

        assert storage.is_ready is True
        assert skills_dir.exists()

        await storage.close()

    @pytest.mark.asyncio
    async def test_save_and_load(self, file_storage, sample_skill):
        """测试保存和加载 skill"""
        # 保存
        await file_storage.save(sample_skill.name, sample_skill)

        # 加载
        loaded = await file_storage.load(sample_skill.name)

        assert loaded is not None
        assert loaded.name == sample_skill.name
        assert loaded.description == sample_skill.description

    @pytest.mark.asyncio
    async def test_save_creates_skill_dir(self, file_storage, sample_skill):
        """测试保存 skill 创建目录"""
        await file_storage.save(sample_skill.name, sample_skill)

        # 检查 skill 目录
        skill_dir = file_storage._skills_dir / "test-skill"
        skill_md = skill_dir / "SKILL.md"

        assert skill_dir.exists()
        assert skill_md.exists()
        # 验证内容包含 skill 名称
        content = skill_md.read_text()
        assert "test_skill" in content or "test-skill" in content

    @pytest.mark.asyncio
    async def test_load_not_found(self, file_storage):
        """测试加载不存在的 skill"""
        result = await file_storage.load("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, file_storage, sample_skill):
        """测试删除 skill"""
        # 先保存
        await file_storage.save(sample_skill.name, sample_skill)

        # 删除
        result = await file_storage.delete(sample_skill.name)

        assert result is True
        assert await file_storage.load(sample_skill.name) is None

    @pytest.mark.asyncio
    async def test_delete_not_found(self, file_storage):
        """测试删除不存在的 skill"""
        result = await file_storage.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_names(self, file_storage, sample_skill, sample_skill2):
        """测试列出所有 skill 名称 - 使用真实目录"""
        # 保存两个 skills
        await file_storage.save(sample_skill.name, sample_skill)
        await file_storage.save(sample_skill2.name, sample_skill2)

        # 列出
        names = await file_storage.list_names()

        # 使用真实目录，可能已有其他 skills，只验证保存的 skills 存在
        assert sample_skill.name in names
        assert sample_skill2.name in names
        assert len(names) >= 2  # 至少有 2 个

    @pytest.mark.asyncio
    async def test_list_names_returns_set(self, file_storage):
        """测试目录列出返回类型 - 使用真实目录"""
        names = await file_storage.list_names()
        # 使用真实目录，返回的至少是集合类型
        assert isinstance(names, set)

    @pytest.mark.asyncio
    async def test_list_names_with_skills(
        self, file_storage, sample_skill, sample_skill2
    ):
        """测试列出所有 skill 名称 - 使用真实目录"""
        # 保存两个 skills
        await file_storage.save(sample_skill.name, sample_skill)
        await file_storage.save(sample_skill2.name, sample_skill2)

        # 列出所有 names
        names = await file_storage.list_names()

        # 使用真实目录，可能已有其他 skills，只验证保存的 skills 存在
        assert sample_skill.name in names
        assert sample_skill2.name in names
        assert len(names) >= 2  # 至少有 2 个

    @pytest.mark.asyncio
    async def test_kebab_case_conversion(self, file_storage):
        """测试 kebab-case 转换"""
        from core.skill.schema import Skill

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

        await file_storage.save(skill.name, skill)

        # 检查目录名是 kebab-case
        expected_dir = file_storage._skills_dir / "test-skill-name"
        assert expected_dir.exists()

    @pytest.mark.asyncio
    async def test_skill_with_references(self, file_storage):
        """测试带 references 的 skill"""
        from core.skill.schema import Skill

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

        await file_storage.save(skill.name, skill)

        # 加载并检查 references
        loaded = await file_storage.load("ref_test")
        assert loaded is not None
        assert "ref.md" in loaded.references
        assert loaded.references["ref.md"] == "Reference content"

    @pytest.mark.asyncio
    async def test_close(self, skills_dir):
        """测试关闭 - 使用 g_config 路径"""
        storage = FileStorage(skills_dir)
        await storage.init()
        await storage.close()
        # 不应抛出异常
