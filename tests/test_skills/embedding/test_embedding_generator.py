"""test_embedding_generator.py - EmbeddingGenerator 测试

测试 EmbeddingGenerator 生成向量的功能。
"""

import pytest
from pathlib import Path

from core.skill.embedding import EmbeddingGenerator
from core.skill.schema import Skill


class TestEmbeddingGenerator:
    """EmbeddingGenerator 测试类"""

    @pytest.mark.asyncio
    async def test_generator_initialization(self, skill_config):
        """测试生成器初始化"""
        generator = EmbeddingGenerator.from_config(skill_config)

        assert generator is not None
        assert generator.is_ready is True

    @pytest.mark.asyncio
    async def test_generator_not_ready_without_client(self):
        """测试没有 client 时生成器未就绪"""
        generator = EmbeddingGenerator(None)

        assert generator.is_ready is False

    @pytest.mark.asyncio
    async def test_generate_for_skill(self, skill_config):
        """测试为 skill 生成向量"""
        generator = EmbeddingGenerator.from_config(skill_config)

        skill = Skill(
            name="test-embedding-skill",
            description="Test skill for embedding",
            content="# Test\n\nThis is a test skill for embedding generation.",
        )

        vector = await generator.generate_for_skill(skill)

        assert vector is not None
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.asyncio
    async def test_generate_for_text(self, skill_config):
        """测试为文本生成向量"""
        generator = EmbeddingGenerator.from_config(skill_config)

        text = "This is a test text for embedding generation."
        vector = await generator.generate(text)

        assert vector is not None
        assert isinstance(vector, list)
        assert len(vector) > 0

    @pytest.mark.asyncio
    async def test_generate_not_ready(self):
        """测试生成器未就绪时返回 None"""
        generator = EmbeddingGenerator(None)

        skill = Skill(
            name="test-skill",
            description="Test",
            content="# Test",
        )

        vector = await generator.generate_for_skill(skill)

        assert vector is None
