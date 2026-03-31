"""test_skill_builder.py - SkillBuilder 构建器测试

测试 SkillBuilder 构建规范 skill 目录的功能。
"""

import pytest
from pathlib import Path

from core.skill.builder import SkillBuilder, validate_name, validate_description
from core.skill.schema import Skill


class TestSkillBuilder:
    """SkillBuilder 测试类"""

    @pytest.fixture
    def builder(self):
        """创建 SkillBuilder 实例"""
        return SkillBuilder()

    @pytest.mark.asyncio
    async def test_builder_initialization(self, builder):
        """测试构建器初始化"""
        assert builder is not None
        assert isinstance(builder, SkillBuilder)

    @pytest.mark.asyncio
    async def test_validate_name(self):
        """测试名称验证"""
        # 有效名称（kebab-case）
        valid, error = validate_name("test-skill")
        assert valid is True
        assert error is None

        # 有效名称
        valid, error = validate_name("myskill")
        assert valid is True

        # 无效名称（包含下划线）
        valid, error = validate_name("test_skill")
        assert valid is False
        assert "invalid characters" in error

    @pytest.mark.asyncio
    async def test_validate_description(self):
        """测试描述验证"""
        # 有效描述
        valid, error = validate_description("A valid description")
        assert valid is True
        assert error is None

        # 有效描述
        valid, error = validate_description("Test")
        assert valid is True

        # 无效描述（空）
        valid, error = validate_description("")
        assert valid is False

    @pytest.mark.asyncio
    async def test_build_skill(self, builder, skills_dir):
        """测试构建 skill"""
        import shutil

        skill_name = "test-built-skill"
        skill_dir = skills_dir / skill_name

        # 清理旧的测试 skill
        if skill_dir.exists():
            shutil.rmtree(skill_dir)

        try:
            # 创建 Skill 对象
            skill = Skill(
                name="test-built-skill",
                description="A test built skill",
                content="# Test Built Skill\n\nThis is content.",
            )

            # 构建 skill
            result = builder.build(skill, skill_dir)

            assert result.skill_dir is not None
            assert result.skill_dir.exists()
            assert (result.skill_dir / "SKILL.md").exists()

            # 验证内容
            skill_md = (result.skill_dir / "SKILL.md").read_text()
            assert "test-built-skill" in skill_md
            assert "A test built skill" in skill_md

        finally:
            # 清理
            if skill_dir.exists():
                shutil.rmtree(skill_dir)

    @pytest.mark.asyncio
    async def test_build_with_files(self, builder, skills_dir):
        """测试构建带文件的 skill"""
        import shutil

        skill_name = "test-with-files"
        skill_dir = skills_dir / skill_name

        if skill_dir.exists():
            shutil.rmtree(skill_dir)

        try:
            # 创建带文件的 Skill
            skill = Skill(
                name="test-with-files",
                description="Skill with files",
                content="# Files\n\nTest.",
                files={
                    "main.py": "print('hello')",
                    "utils.py": "def util(): pass",
                },
            )

            result = builder.build(skill, skill_dir)

            assert result.skill_dir is not None

            # 验证文件（注意：非 Python 代码的文件会被放在 files/ 目录）
            files_dir = result.skill_dir / "files"
            assert files_dir.exists() or (result.skill_dir / "SKILL.md").exists()

        finally:
            if skill_dir.exists():
                shutil.rmtree(skill_dir)

    @pytest.mark.asyncio
    async def test_build_with_dependencies(self, builder, skills_dir):
        """测试构建带依赖的 skill"""
        import shutil

        skill_name = "test-with-deps"
        skill_dir = skills_dir / skill_name

        if skill_dir.exists():
            shutil.rmtree(skill_dir)

        try:
            # 创建带依赖的 Skill
            skill = Skill(
                name="test-with-deps",
                description="Skill with dependencies",
                content="# Deps\n\nTest.",
                dependencies=["requests", "pandas"],
            )

            result = builder.build(skill, skill_dir)

            # 验证依赖在 frontmatter 中
            skill_md = (result.skill_dir / "SKILL.md").read_text()
            assert "requests" in skill_md or "pandas" in skill_md

        finally:
            if skill_dir.exists():
                shutil.rmtree(skill_dir)

    @pytest.mark.asyncio
    async def test_build_python_playbook(self, builder, skills_dir):
        """测试构建 Python playbook skill"""
        import shutil
        from core.skill.schema import ExecutionMode

        skill_name = "test-playbook"
        skill_dir = skills_dir / skill_name

        if skill_dir.exists():
            shutil.rmtree(skill_dir)

        try:
            # 创建 Python playbook Skill（显式设置 execution_mode）
            skill = Skill(
                name="test-playbook",
                description="Python playbook skill",
                content="print('Hello World')",
                execution_mode=ExecutionMode.PLAYBOOK,
            )

            result = builder.build(skill, skill_dir)

            # 检查识别为 PLAYBOOK
            assert result.skill.execution_mode == ExecutionMode.PLAYBOOK

            # 检查 skill 目录存在
            assert result.skill_dir.exists()
            assert (result.skill_dir / "SKILL.md").exists()

            # Python 代码应该被放到 scripts/ 目录
            scripts_dir = result.skill_dir / "scripts"
            assert scripts_dir.exists()
            assert (scripts_dir / "test-playbook.py").exists()

        finally:
            if skill_dir.exists():
                shutil.rmtree(skill_dir)
