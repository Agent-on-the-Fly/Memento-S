"""test_skill_model.py - Skill 模型测试

测试 Skill 领域模型的创建、验证和属性。
"""

import pytest
from pathlib import Path

from core.skill.schema import Skill, ExecutionMode, DEFAULT_SKILL_PARAMS


class TestSkillModel:
    """Skill 模型测试类"""

    @pytest.mark.asyncio
    async def test_skill_creation(self):
        """测试创建基本 Skill 对象"""
        skill = Skill(
            name="test_skill",
            description="A test skill",
            content="# Test\n\nThis is test content.",
        )

        assert skill.name == "test_skill"
        assert skill.description == "A test skill"
        assert skill.content == "# Test\n\nThis is test content."
        assert skill.dependencies == []
        assert skill.version == 0

    @pytest.mark.asyncio
    async def test_skill_with_frontmatter(self):
        """测试从 frontmatter 解析 skill"""
        content = """---
name: data_analysis
description: Analyze data with pandas
metadata:
  function_name: analyze_data
  dependencies:
    - pandas
    - numpy
---

# Data Analysis

This skill analyzes data.
"""
        skill = Skill(
            name="data_analysis",
            description="Analyze data with pandas",
            content=content,
            dependencies=["pandas", "numpy"],
        )

        assert skill.name == "data_analysis"
        assert "pandas" in skill.dependencies
        assert "numpy" in skill.dependencies

    @pytest.mark.asyncio
    async def test_skill_execution_mode(self):
        """测试 Skill 执行模式"""
        knowledge_skill = Skill(
            name="knowledge_skill",
            description="Knowledge mode",
            content="# Knowledge",
            execution_mode=ExecutionMode.KNOWLEDGE,
        )

        playbook_skill = Skill(
            name="playbook_skill",
            description="Playbook mode",
            content="# Playbook",
            execution_mode=ExecutionMode.PLAYBOOK,
        )

        assert knowledge_skill.execution_mode == ExecutionMode.KNOWLEDGE
        assert playbook_skill.execution_mode == ExecutionMode.PLAYBOOK

    @pytest.mark.asyncio
    async def test_skill_with_files(self):
        """测试带文件的 Skill"""
        skill = Skill(
            name="file_skill",
            description="Skill with files",
            content="# Skill with files",
            files={
                "main.py": "print('hello')",
                "utils.py": "def helper(): pass",
            },
        )

        assert "main.py" in skill.files
        assert "utils.py" in skill.files
        assert skill.files["main.py"] == "print('hello')"

    @pytest.mark.asyncio
    async def test_skill_with_references(self):
        """测试带 references 的 Skill"""
        skill = Skill(
            name="ref_skill",
            description="Skill with references",
            content="# References",
            references={
                "doc.md": "# Documentation",
                "example.py": "# Example code",
            },
        )

        assert "doc.md" in skill.references
        assert skill.references["doc.md"] == "# Documentation"

    @pytest.mark.asyncio
    async def test_skill_default_params(self):
        """测试默认参数 schema"""
        assert "type" in DEFAULT_SKILL_PARAMS
        assert DEFAULT_SKILL_PARAMS["type"] == "object"
        assert "properties" in DEFAULT_SKILL_PARAMS
        assert "request" in DEFAULT_SKILL_PARAMS["properties"]

    @pytest.mark.asyncio
    async def test_skill_with_source_dir(self):
        """测试带 source_dir 的 Skill"""
        skill = Skill(
            name="dir_skill",
            description="Skill with directory",
            content="# Directory",
            source_dir="/path/to/skills/dir-skill",
        )

        assert skill.source_dir == "/path/to/skills/dir-skill"

    @pytest.mark.asyncio
    async def test_skill_with_parameters(self):
        """测试带自定义参数的 Skill"""
        custom_params = {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["filename"],
        }

        skill = Skill(
            name="param_skill",
            description="Skill with parameters",
            content="# Parameters",
            parameters=custom_params,
        )

        assert skill.parameters == custom_params
        assert "filename" in skill.parameters["properties"]
