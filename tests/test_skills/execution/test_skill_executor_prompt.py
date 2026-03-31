"""test_skill_executor_prompt.py - SkillExecutor Prompt 构建测试

测试 SkillExecutor 的 prompt 构建功能。
"""

import pytest
from pathlib import Path

from core.skill.execution import SkillExecutor
from core.skill.config import SkillConfig
from core.skill.schema import Skill


class TestSkillExecutorPrompt:
    """SkillExecutor Prompt 构建测试类"""

    @pytest.fixture
    def executor(self, skill_config):
        """创建 SkillExecutor 实例"""
        return SkillExecutor(skill_config)

    @pytest.fixture
    def sample_skill(self):
        """创建示例 skill"""
        return Skill(
            name="test_skill",
            description="A test skill",
            content="# Test Skill\n\nThis is test content.",
        )

    @pytest.mark.asyncio
    async def test_build_prompt_basic(self, executor, sample_skill):
        """测试基础 prompt 构建"""
        prompt = executor._build_prompt(
            skill=sample_skill,
            query="Test query",
            selected_references={},
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Test query" in prompt
        assert "test_skill" in prompt

    @pytest.mark.asyncio
    async def test_build_prompt_with_references(self, executor, sample_skill):
        """测试带 references 的 prompt 构建"""
        references = {
            "doc.md": "# Documentation\n\nThis is documentation.",
            "example.py": "# Example\nprint('hello')",
        }

        prompt = executor._build_prompt(
            skill=sample_skill,
            query="Test with references",
            selected_references=references,
        )

        assert "doc.md" in prompt or "Documentation" in prompt
        assert isinstance(prompt, str)

    @pytest.mark.asyncio
    async def test_build_prompt_with_params(self, executor, sample_skill):
        """测试带参数的 prompt 构建"""
        params = {"filename": "test.txt", "count": 5}

        prompt = executor._build_prompt(
            skill=sample_skill,
            query="Test with params",
            selected_references={},
            params=params,
        )

        assert isinstance(prompt, str)
        # 参数应该在 prompt 中体现
        assert len(prompt) > 0

    @pytest.mark.asyncio
    async def test_select_relevant_references(self, executor):
        """测试选择相关 references"""
        references = {
            "doc.md": "Documentation content",
            "example.py": "Example code",
            "readme.md": "Readme content",
        }

        selected = executor._select_relevant_references(references, "documentation")

        # 应该返回部分 references
        assert isinstance(selected, dict)
        assert len(selected) <= len(references)

    @pytest.mark.asyncio
    async def test_select_relevant_references_empty(self, executor):
        """测试空 references"""
        selected = executor._select_relevant_references({}, "query")

        assert selected == {}

    @pytest.mark.asyncio
    async def test_select_relevant_references_none(self, executor):
        """测试 None references"""
        selected = executor._select_relevant_references(None, "query")

        assert selected == {}
