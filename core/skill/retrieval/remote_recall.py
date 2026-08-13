# SPDX-License-Identifier: Apache-2.0
"""remote_recall — 远程 Skill Retrieval API 客户端

通过 HTTP 调用独立部署的 skill_retrieval_api 微服务，
提供 search（检索）接口。
"""

from __future__ import annotations

import certifi
import httpx
import ssl

from utils.logger import get_logger
from .base import BaseRecall
from shared.schema import SkillSearchResult

logger = get_logger(__name__)


class RemoteRecall(BaseRecall):
    """远程 Skill Retrieval API 客户端

    提供接口：
    - POST /api/v1/search    — 检索 skill，返回 top-k 的 name + description
    """

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._embedding_ready: bool = False
        self._size: int = 0
        self._available: bool = False

    def _make_client(self) -> httpx.AsyncClient:
        """Create a loop-local client.

        RemoteRecall is used by sync fixtures and async application code. A
        persistent AsyncClient cannot safely cross those event-loop boundaries.
        """
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        return httpx.AsyncClient(
            timeout=self._timeout,
            verify=ssl_context,
            trust_env=False,
        )

    async def _health_check(self) -> None:
        """启动时健康检查，填充元数据并设置可用性标记。"""
        import asyncio
        try:
            async with self._make_client() as client:
                resp = await asyncio.wait_for(
                    client.get(f"{self._base_url}/health"),
                    timeout=self._timeout,
                )
            if resp.status_code == 200:
                data = resp.json()
                self._embedding_ready = data.get("embedding_ready", False)
                self._size = data.get("catalog_size", 0) or data.get("total_skills", 0)
                self._available = True
                logger.info(
                    "RemoteRecall connected: {} (skills={}, embedding={})",
                    self._base_url,
                    self._size,
                    self._embedding_ready,
                )
            else:
                self._available = False
                logger.warning("RemoteRecall health check returned HTTP {}", resp.status_code)
        except Exception as e:
            self._available = False
            logger.warning("RemoteRecall health check failed: {}", e)

    @classmethod
    async def from_config(cls, config: "SkillConfig") -> "RemoteRecall | None":
        """从配置异步创建 RemoteRecall 实例。

        Args:
            config: SkillConfig 配置

        Returns:
            RemoteRecall 实例（如果配置了 cloud_catalog_url），否则返回 None
        """
        if config.cloud_catalog_url:
            recall = cls(config.cloud_catalog_url, timeout=5.0)
            await recall._health_check()
            return recall
        return None

    @property
    def name(self) -> str:
        """召回策略名称"""
        return "remote"

    def is_available(self) -> bool:
        """检查远程服务是否可用"""
        return self._available

    @property
    def embedding_ready(self) -> bool:
        return self._embedding_ready

    @property
    def size(self) -> int:
        return self._size

    async def search(self, query: str, k: int = 5, **kwargs) -> list["SkillSearchResult"]:
        """搜索 skill，返回 SkillSearchResult 列表。"""
        try:
            async with self._make_client() as client:
                resp = await client.post(
                    f"{self._base_url}/api/v1/search",
                    json={"query": query, "top_k": k},
                )
            if resp.status_code != 200:
                logger.warning("Remote search failed: HTTP {}", resp.status_code)
                return []

            results = resp.json().get("results", [])
            candidates: list[SkillSearchResult] = []
            for r in results:
                candidates.append(
                    SkillSearchResult(
                        name=r["name"],
                        description=r.get("description", ""),
                        source="cloud",  # "remote" → "cloud"
                        score=r.get("score", 0.0),
                        match_type="remote",
                    )
                )
            return candidates
        except Exception as e:
            logger.warning("Remote search error: {}", e)
            return []

    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = super().get_stats()
        stats.update(
            {
                "base_url": self._base_url,
                "embedding_ready": self._embedding_ready,
                "catalog_size": self._size,
            }
        )
        return stats

    async def close(self) -> None:
        """Compatibility no-op; request-local clients close themselves."""
        return None
