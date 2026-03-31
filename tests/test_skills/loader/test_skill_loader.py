"""test_skill_loader.py - SkillLoader 加载器测试

测试 SkillLoader 从磁盘加载 skill 的功能。
"""

import pytest
from pathlib import Path

from core.skill.loader import SkillLoader
from core.skill.schema import Skill


class TestSkillLoader:
    """SkillLoader 测试类"""

    @pytest.fixture
    def loader(self, skills_dir):
        """创建 SkillLoader 实例"""
        return SkillLoader(skills_dir)

    @pytest.mark.asyncio
    async def test_loader_initialization(self, loader, skills_dir):
        """测试加载器初始化"""
        assert loader._skills_dir == Path(skills_dir)

    @pytest.mark.asyncio
    async def test_load_from_directory(self, loader, skills_dir):
        """测试从目录加载 skill"""
        # 先创建一个测试 skill 目录
        test_skill_dir = skills_dir / "test-loader-skill"
        test_skill_dir.mkdir(parents=True, exist_ok=True)

        # 创建 SKILL.md
        skill_md = test_skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: test_loader_skill
description: Test skill for loader
---

# Test Loader Skill

This is a test skill.
""")

        try:
            skill = loader.load_from_dir(test_skill_dir)

            assert skill is not None
            assert skill.name == "test_loader_skill"
            assert skill.description == "Test skill for loader"
        finally:
            # 清理
            import shutil

            if test_skill_dir.exists():
                shutil.rmtree(test_skill_dir)

    @pytest.mark.asyncio
    async def test_load_by_name(self, loader, skills_dir):
        """测试按名称加载 skill"""
        # 创建一个测试 skill
        test_skill_dir = skills_dir / "test-by-name"
        test_skill_dir.mkdir(parents=True, exist_ok=True)

        skill_md = test_skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: test_by_name
description: Test by name
---

# Test
""")

        try:
            skill = loader.load("test_by_name")

            assert skill is not None
            assert skill.name == "test_by_name"
        finally:
            import shutil

            if test_skill_dir.exists():
                shutil.rmtree(test_skill_dir)

    @pytest.mark.asyncio
    async def test_load_not_found(self, loader):
        """测试加载不存在的 skill"""
        skill = loader.load("non_existent_skill_xyz")
        assert skill is None

    @pytest.mark.asyncio
    async def test_load_with_scripts(self, loader, skills_dir):
        """测试加载带 scripts 的 skill"""
        test_skill_dir = skills_dir / "test-with-scripts"
        test_skill_dir.mkdir(parents=True, exist_ok=True)

        # 创建 SKILL.md
        skill_md = test_skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: test_with_scripts
description: Test with scripts
---

# Test
""")

        # 创建 scripts 目录和文件
        scripts_dir = test_skill_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        main_py = scripts_dir / "main.py"
        main_py.write_text("print('hello')")

        try:
            skill = loader.load_from_dir(test_skill_dir, full=True)

            assert skill is not None
            # files 字典只包含文件名，不包含路径前缀
            assert "main.py" in skill.files
        finally:
            import shutil

            if test_skill_dir.exists():
                shutil.rmtree(test_skill_dir)

    @pytest.mark.asyncio
    async def test_name_normalization(self, loader, skills_dir):
        """测试名称规范化（snake_case 转 kebab-case）"""
        test_skill_dir = skills_dir / "name-normalization"
        test_skill_dir.mkdir(parents=True, exist_ok=True)

        skill_md = test_skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: name_normalization
description: Test name normalization
---

# Test
""")

        try:
            # 使用 snake_case 名称加载
            skill = loader.load("name_normalization")
            assert skill is not None

            # 使用 kebab-case 名称加载
            skill2 = loader.load("name-normalization")
            assert skill2 is not None
        finally:
            import shutil

            if test_skill_dir.exists():
                shutil.rmtree(test_skill_dir)
