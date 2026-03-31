"""Embedding 生成器 - 为 Skill 模块提供向量生成能力

这是一个通用工具类，供 store 和 retrieval 模块使用。
职责：封装 embedding 生成逻辑，提供统一的生成接口。
"""

from __future__ import annotations

from typing import List, Any

from core.skill.config import SkillConfig
from core.skill.schema import Skill
from middleware.llm.embedding_client import EmbeddingClient
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Embedding 生成器

    职责：
    1. 为单个文本/skill 生成 embedding
    2. 管理 embedding client 的生命周期

    使用：
        from core.skill.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator(embedding_client)
        vector = await generator.generate("查询文本")
        vector = await generator.generate_for_skill(skill)
    """

    def __init__(self, embedding_client: Any) -> None:
        """
        构造函数：接收外部 client（用于测试/自定义）

        Args:
            embedding_client: 符合 OpenAI 兼容接口的 embedding client
        """
        self._client = embedding_client

    @classmethod
    def from_config(cls, config: "SkillConfig") -> "EmbeddingGenerator":
        """
        工厂方法：从配置创建（用于生产环境）

        Args:
            config: SkillConfig 配置

        Returns:
            配置好的 EmbeddingGenerator 实例
        """
        client = EmbeddingClient.from_config()
        return cls(client)

    @property
    def is_ready(self) -> bool:
        """检查 client 是否可用"""
        return self._client is not None

    @property
    def dimension(self) -> int | None:
        """返回 embedding 维度"""
        if not self._client:
            return None
        return getattr(self._client, "dimension", None)

    async def generate(self, text: str) -> List[float] | None:
        """
        为单个文本生成 embedding

        Args:
            text: 输入文本

        Returns:
            embedding 向量，失败返回 None
        """
        if not self._client:
            return None

        try:
            vectors = await self._client.embed([text])
            if vectors and len(vectors) > 0:
                return vectors[0]
        except Exception as e:
            logger.warning("Failed to generate embedding: {}", e)

        return None

    async def generate_for_skill(self, skill: Skill) -> List[float] | None:
        """
        为 skill 生成 embedding

        Args:
            skill: Skill 对象

        Returns:
            embedding 向量，失败返回 None
        """
        # 使用 skill 的 embedding 文本表示
        text = skill.to_embedding_text()
        return await self.generate(text)

    def close(self) -> None:
        """关闭资源（如果需要）"""
        if self._client and hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception as e:
                logger.warning("Failed to close embedding client: {}", e)
