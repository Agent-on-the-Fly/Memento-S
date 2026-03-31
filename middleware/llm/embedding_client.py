"""OpenAI 兼容 Embedding API 客户端 — 轻量级即用即走模式

参考 LLMClient 设计，无需初始化，按需创建使用。

示例:
    from middleware.llm import EmbeddingClient, EmbeddingClientConfig

    # 简单使用
    async with EmbeddingClient() as client:
        embeddings = await client.embed(["text1", "text2"])
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Any

import httpx

from middleware.config import g_config
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingError(Exception):
    """Embedding API 调用异常基类"""

    pass


class EmbeddingAPIError(EmbeddingError):
    """Embedding API 返回错误"""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: dict | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body or {}


@dataclass
class EmbeddingClientConfig:
    """Embedding 客户端配置

    Attributes:
        base_url: Embedding API 基础 URL，默认从 g_config 读取
        api_key: API 密钥（可选）
        model: 模型名称
        timeout: 请求超时时间（秒）
        batch_size: 批量处理大小
    """

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: float = 60.0
    batch_size: int = 64

    @classmethod
    def from_config(cls) -> "EmbeddingClientConfig":
        """从全局配置创建配置实例"""
        cfg = g_config.skills.retrieval
        return cls(
            base_url=cfg.embedding_base_url or "",
            api_key=cfg.embedding_api_key or "",
            model=cfg.embedding_model or "",
            timeout=60.0,
            batch_size=64,
        )


class EmbeddingClient:
    """轻量级 Embedding 客户端

    无需初始化，即用即走。每次请求独立创建 HTTP 连接，
    使用上下文管理器确保资源释放。
    """

    def __init__(self, config: EmbeddingClientConfig | None = None):
        """初始化 Embedding 客户端

        Args:
            config: 配置对象，为 None 时从 g_config 自动读取
        """
        self._config = config or EmbeddingClientConfig.from_config()
        self._dim: int | None = None

    @classmethod
    def from_config(cls) -> "EmbeddingClient":
        """从全局配置创建客户端实例

        Returns:
            配置好的 EmbeddingClient 实例
        """
        return cls(EmbeddingClientConfig.from_config())

    @property
    def dimension(self) -> int | None:
        """embedding 维度（首次调用 embed 后缓存）"""
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量获取 embedding 向量

        Args:
            texts: 要嵌入的文本列表

        Returns:
            embedding 向量列表
        """
        if not texts:
            return []

        if not self._config.base_url:
            raise EmbeddingError("Embedding service not configured")

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._config.batch_size):
            batch = texts[i : i + self._config.batch_size]
            embeddings = await self._request(batch)
            all_embeddings.extend(embeddings)

        if all_embeddings and self._dim is None:
            self._dim = len(all_embeddings[0])

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """嵌入单条查询

        Args:
            query: 查询文本

        Returns:
            embedding 向量
        """
        results = await self.embed([query])
        if not results:
            raise EmbeddingAPIError("Embedding API returned empty result")
        return results[0]

    async def _request(self, texts: list[str]) -> list[list[float]]:
        """发送 embedding 请求"""
        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            resp = await client.post(
                f"{self._config.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._config.api_key or 'no-key-required'}",
                    "Content-Type": "application/json",
                },
                json={"input": texts, "model": self._config.model},
            )
            resp.raise_for_status()
            body = resp.json()
            data = body.get("data")

            if not data:
                raise EmbeddingAPIError(
                    f"Embedding API returned no data: {body}",
                    status_code=resp.status_code,
                    response_body=body,
                )

            data.sort(key=lambda x: x["index"])
            return [d["embedding"] for d in data]

    async def __aenter__(self) -> "EmbeddingClient":
        """上下文管理器入口"""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """上下文管理器出口 — 无需清理资源"""
        pass
