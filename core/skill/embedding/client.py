"""OpenAI 兼容 Embedding API 客户端 — 用于本地 skill 向量化

通过 base_url + api_key + model 调用远程 Embedding 服务，
替代之前本地加载 SentenceTransformer / Qwen3 模型。

示例:
    # 方式 1: 上下文管理器 (推荐)
    async with EmbeddingClient(base_url="http://localhost:8000") as client:
        embeddings = await client.embed(["text1", "text2"])

    # 方式 2: 手动管理
    client = EmbeddingClient(base_url="http://localhost:8000")
    try:
        embeddings = await client.embed(["text1", "text2"])
    finally:
        await client.close()
"""

from __future__ import annotations

import atexit
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from utils.logger import get_logger

if TYPE_CHECKING:
    from types import TracebackType

logger = get_logger(__name__)

# 追踪所有活动的 EmbeddingClient 实例，用于程序退出时清理
_active_clients: weakref.WeakSet = weakref.WeakSet()


def _cleanup_all_clients():
    """程序退出时清理所有活动的 EmbeddingClient"""
    clients = list(_active_clients)
    if clients:
        logger.debug("Cleaning up {} active EmbeddingClient(s)", len(clients))
        for client in clients:
            if hasattr(client, "_closed") and not client._closed:
                try:
                    # 尝试同步关闭
                    import asyncio

                    try:
                        loop = asyncio.get_event_loop()
                        if not loop.is_running():
                            loop.run_until_complete(client.close())
                    except RuntimeError:
                        pass
                except Exception:
                    pass


# 注册退出清理函数
atexit.register(_cleanup_all_clients)


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


class EmbeddingConfigError(EmbeddingError):
    """Embedding 配置错误"""

    pass


@dataclass
class EmbeddingClientConfig:
    """Embedding 客户端配置

    Attributes:
        base_url: Embedding API 基础 URL
        api_key: API 密钥（可选）
        model: 模型名称
        timeout: 请求超时时间（秒）
        batch_size: 批量处理大小
        max_retries: 最大重试次数
    """

    base_url: str
    api_key: str = ""
    model: str = ""
    timeout: float = 60.0
    batch_size: int = 64
    max_retries: int = 2

    def __post_init__(self):
        """验证并规范化配置"""
        if not self.base_url or not isinstance(self.base_url, str):
            raise EmbeddingConfigError("base_url is required and must be a string")

        # 自动修正：用户可能填了完整的 /v1/embeddings
        url = self.base_url.rstrip("/")
        if url.endswith("/embeddings"):
            url = url.rsplit("/embeddings", 1)[0]

        # 验证 URL 格式
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise EmbeddingConfigError(f"Invalid base_url: {self.base_url}")

        self.base_url = url

        if self.batch_size <= 0:
            raise EmbeddingConfigError("batch_size must be positive")

        if self.timeout <= 0:
            raise EmbeddingConfigError("timeout must be positive")

        if self.max_retries < 0:
            raise EmbeddingConfigError("max_retries must be non-negative")


@dataclass
class EmbeddingStats:
    """Embedding 客户端使用统计"""

    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    total_retries: int = 0

    def reset(self) -> None:
        """重置统计信息"""
        self.total_requests = 0
        self.total_tokens = 0
        self.total_errors = 0
        self.total_retries = 0


class EmbeddingClient:
    """通过 OpenAI 兼容接口获取 embedding 向量

    支持上下文管理器，自动释放资源。
    """

    def __init__(self, config: EmbeddingClientConfig | None = None, **kwargs):
        """初始化 Embedding 客户端

        Args:
            config: 配置对象（优先级高）
            **kwargs: 如果未提供 config，则从这些参数创建配置
                - base_url: str (required)
                - api_key: str = ""
                - model: str = "qwen-4b"
                - timeout: float = 60.0
                - batch_size: int = 64
                - max_retries: int = 2

        Raises:
            EmbeddingConfigError: 配置无效
        """
        if config is not None and kwargs:
            raise EmbeddingConfigError("Cannot provide both config and kwargs")

        if config is not None:
            self._config = config
        else:
            self._config = EmbeddingClientConfig(**kwargs)

        self._client: httpx.AsyncClient | None = None
        self._dim: int | None = None
        self._stats = EmbeddingStats()
        self._closed = False

        # 注册到活动客户端集合，用于程序退出时清理
        _active_clients.add(self)
        self._loop_id: int | None = None  # 追踪创建客户端的事件循环

    def _get_client(self) -> httpx.AsyncClient:
        """获取 httpx 客户端，每次都创建新的实例避免跨事件循环问题

        注意：由于存在 "Event loop is closed" 问题（当在多个 asyncio.run() 调用中
        复用 AsyncClient 时会发生），我们选择每次都创建新实例而非缓存。
        这会带来一些性能开销，但能保证正确性。
        """
        return httpx.AsyncClient(timeout=self._config.timeout)

    @property
    def dimension(self) -> int | None:
        """embedding 维度（首次调用 embed 后缓存）"""
        return self._dim

    @property
    def stats(self) -> EmbeddingStats:
        """获取使用统计"""
        return self._stats

    @property
    def is_closed(self) -> bool:
        """检查客户端是否已关闭"""
        return self._closed

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量获取 embedding 向量

        Args:
            texts: 要嵌入的文本列表

        Returns:
            embedding 向量列表

        Raises:
            EmbeddingAPIError: API 调用失败
            EmbeddingError: 其他错误
        """
        if self._closed:
            raise EmbeddingError("Client is closed")

        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_chars = sum(len(t) for t in texts)

        logger.debug("Embedding {} texts ({} chars total)", len(texts), total_chars)

        for i in range(0, len(texts), self._config.batch_size):
            batch = texts[i : i + self._config.batch_size]
            batch_num = i // self._config.batch_size + 1
            total_batches = (
                len(texts) + self._config.batch_size - 1
            ) // self._config.batch_size

            try:
                embeddings = await self._request(batch, batch_num, total_batches)
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error("Failed to embed batch {}/{}: {}", batch_num, total_batches, e)
                raise

        if all_embeddings and self._dim is None:
            self._dim = len(all_embeddings[0])
            logger.info("Detected embedding dimension: {}", self._dim)

        self._stats.total_tokens += total_chars
        logger.debug("Successfully embedded {} texts", len(texts))

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """嵌入单条查询

        Args:
            query: 查询文本

        Returns:
            embedding 向量

        Raises:
            EmbeddingAPIError: API 调用失败
        """
        if not query or not isinstance(query, str):
            raise EmbeddingError("Query must be a non-empty string")

        results = await self.embed([query])
        if not results:
            raise EmbeddingAPIError("Embedding API returned empty result")
        return results[0]

    async def _request(
        self, texts: list[str], batch_num: int = 1, total_batches: int = 1
    ) -> list[list[float]]:
        """发送 embedding 请求（带重试）

        Args:
            texts: 文本批次
            batch_num: 当前批次号
            total_batches: 总批次数

        Returns:
            embedding 向量列表

        Raises:
            EmbeddingAPIError: 请求失败
        """
        last_err: Exception | None = None
        client = self._get_client()

        try:
            for attempt in range(self._config.max_retries + 1):
                try:
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
                    embeddings = [d["embedding"] for d in data]

                    self._stats.total_requests += 1
                    logger.debug(
                        f"Batch {batch_num}/{total_batches} succeeded "
                        f"(attempt {attempt + 1}/{self._config.max_retries + 1})"
                    )

                    return embeddings

                except httpx.HTTPStatusError as e:
                    last_err = e
                    self._stats.total_errors += 1
                    if attempt < self._config.max_retries:
                        self._stats.total_retries += 1
                        logger.warning(
                            f"Embedding API batch {batch_num}/{total_batches} failed "
                            f"(attempt {attempt + 1}/{self._config.max_retries + 1}): "
                            f"HTTP {e.response.status_code}, retrying..."
                        )
                    else:
                        raise EmbeddingAPIError(
                            f"Embedding API error after {self._config.max_retries + 1} attempts: {e}",
                            status_code=e.response.status_code,
                            response_body=e.response.json()
                            if e.response.text
                            else None,
                        ) from e

                except (httpx.HTTPError, KeyError) as e:
                    last_err = e
                    self._stats.total_errors += 1
                    if attempt < self._config.max_retries:
                        self._stats.total_retries += 1
                        logger.warning(
                            f"Embedding API batch {batch_num}/{total_batches} failed "
                            f"(attempt {attempt + 1}/{self._config.max_retries + 1}): {e}, retrying..."
                        )
                    else:
                        raise EmbeddingAPIError(
                            f"Embedding API error after {self._config.max_retries + 1} attempts: {e}"
                        ) from e

            # 不应该执行到这里
            raise EmbeddingAPIError(f"Unexpected error: {last_err}")
        finally:
            # 确保 client 被关闭，避免资源泄漏
            await client.aclose()

    async def close(self) -> None:
        """关闭客户端，释放资源

        使用完客户端后应调用此方法，或使用上下文管理器。
        """
        self._closed = True
        if self._client is not None:
            try:
                logger.debug("Closing EmbeddingClient")
                await self._client.aclose()
            except RuntimeError as e:
                # 事件循环可能已关闭，静默处理
                if "Event loop is closed" in str(e):
                    logger.debug("Event loop already closed, skipping client cleanup")
                else:
                    raise
            finally:
                self._client = None

    async def __aenter__(self) -> EmbeddingClient:
        """上下文管理器入口"""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """上下文管理器出口"""
        await self.close()

    def __del__(self):
        """析构函数 - 完全放弃异步关闭，静默处理避免错误"""
        # 不尝试异步关闭，避免 "Event loop is closed" 错误
        # 资源将在进程退出时由系统回收
        self._closed = True
        self._client = None
