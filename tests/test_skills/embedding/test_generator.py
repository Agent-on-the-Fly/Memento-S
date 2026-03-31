"""EmbeddingGenerator 真实调用测试

使用 .venv 环境运行真实测试：
    .venv/bin/python -m pytest tests/test_skills/embedding/test_generator.py -v

测试使用真实配置，如果未配置 embedding 服务会自动跳过。
"""

from __future__ import annotations

import pytest

from core.skill.embedding import EmbeddingGenerator
from core.skill.schema import Skill


class TestEmbeddingGeneratorReal:
    """EmbeddingGenerator 真实调用测试"""

    @pytest.mark.asyncio
    async def test_is_ready(self, generator, embedding_client):
        """测试 is_ready 属性"""
        assert generator.is_ready is True
        # dimension 在首次 embed 后才有值
        assert generator.dimension is None or isinstance(generator.dimension, int)

    @pytest.mark.asyncio
    async def test_generate(self, generator):
        """测试单文本生成"""
        result = await generator.generate("这是一个测试文本")

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

        # 验证维度缓存
        assert generator.dimension == len(result)

    @pytest.mark.asyncio
    async def test_generate_consistency(self, generator):
        """测试多次生成维度一致"""
        text1 = "测试文本1"
        text2 = "测试文本2"

        result1 = await generator.generate(text1)
        result2 = await generator.generate(text2)

        # 维度应该一致
        assert len(result1) == len(result2)
        assert generator.dimension == len(result1)

    @pytest.mark.asyncio
    async def test_generate_chinese(self, generator):
        """测试中文文本"""
        result = await generator.generate("这是一个中文测试句子")

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_english(self, generator):
        """测试英文文本"""
        result = await generator.generate("This is an English test sentence")

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_for_skill(self, generator):
        """测试为 skill 生成 embedding"""
        skill = Skill(
            name="test_skill",
            description="A test skill for embedding generation",
            content="# Test Skill\n\nThis is a test skill content.",
        )

        result = await generator.generate_for_skill(skill)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_empty_string(self, generator):
        """测试空字符串"""
        result = await generator.generate("")

        # 空字符串应该也能生成（取决于服务实现）
        if result is not None:
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_from_config(self, test_config):
        """测试 from_config 工厂方法"""
        from core.skill.config import SkillConfig

        config = SkillConfig.from_global_config()
        gen = EmbeddingGenerator.from_config(config)

        assert gen.is_ready is True

        # 测试生成
        result = await gen.generate("测试 from_config")
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0


class TestEmbeddingGeneratorWithoutClient:
    """无 client 情况下的测试"""

    def test_init_none_client(self):
        """测试传入 None client"""
        gen = EmbeddingGenerator(None)

        assert gen.is_ready is False
        assert gen.dimension is None

    @pytest.mark.asyncio
    async def test_generate_not_ready(self):
        """测试未初始化时返回 None"""
        gen = EmbeddingGenerator(None)

        result = await gen.generate("test")

        assert result is None

    def test_close_none_client(self):
        """测试关闭 None client 不报错"""
        gen = EmbeddingGenerator(None)

        # 不应抛出异常
        gen.close()

    def test_close_without_close_method(self):
        """测试 client 没有 close 方法"""

        class FakeClient:
            pass

        gen = EmbeddingGenerator(FakeClient())

        # 不应抛出异常
        gen.close()


@pytest.mark.stress
class TestEmbeddingGeneratorStress:
    """压力测试"""

    @pytest.mark.asyncio
    async def test_generate_long_text(self, generator):
        """测试长文本"""
        long_text = "这是一个测试。" * 100

        result = await generator.generate(long_text)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_special_chars(self, generator):
        """测试特殊字符"""
        special_text = "Test with special chars: !@#$%^&*() 中文 🎉"

        result = await generator.generate(special_text)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
