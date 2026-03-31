"""remote_recall — 远程 Skill Retrieval API 客户端

通过 HTTP 调用独立部署的 skill_retrieval_api 微服务，
提供 search（检索）接口。
"""

from __future__ import annotations

import certifi
import httpx

from utils.logger import get_logger
from .base import BaseRecall
from .schema import RecallCandidate

logger = get_logger(__name__)


class RemoteRecall(BaseRecall):
    """远程 Skill Retrieval API 客户端

    提供接口：
    - POST /api/v1/search    — 检索 skill，返回 top-k 的 name + description
    """

    def __init__(self, base_url: str, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout,verify=certifi.where(),trust_env=False)
        self._embedding_ready: bool = False
        self._size: int = 0

        # 健康检查
        try:
            resp = self._client.get(f"{self._base_url}/health")
            if resp.status_code == 200:
                data = resp.json()
                self._embedding_ready = data.get("embedding_ready", False)
                self._size = data.get("catalog_size", 0) or data.get("total_skills", 0)
                logger.info(
                    "RemoteRecall connected: {} (skills={}, embedding={})",
                    self._base_url,
                    self._size,
                    self._embedding_ready,
                )
        except Exception as e:
            logger.warning("RemoteRecall health check failed: {}", e)

    @classmethod
    def from_config(cls, config: "SkillConfig") -> "RemoteRecall | None":
        """从配置创建 RemoteRecall 实例

        Args:
            config: SkillConfig 配置

        Returns:
            RemoteRecall 实例（如果配置了 cloud_catalog_url），否则返回 None
        """
        if config.cloud_catalog_url:
            return cls(config.cloud_catalog_url)
        return None

    @property
    def name(self) -> str:
        """召回策略名称"""
        return "remote"

    def is_available(self) -> bool:
        """检查远程服务是否可用"""
        try:
            resp = self._client.get(f"{self._base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    @property
    def embedding_ready(self) -> bool:
        return self._embedding_ready

    @property
    def size(self) -> int:
        return self._size

    async def search(self, query: str, k: int = 5, **kwargs) -> list["RecallCandidate"]:
        """搜索 skill，返回 RecallCandidate 列表。"""
        try:
            resp = self._client.post(
                f"{self._base_url}/api/v1/search",
                json={"query": query, "top_k": k},
            )
            if resp.status_code != 200:
                logger.warning("Remote search failed: HTTP {}", resp.status_code)
                return []

            results = resp.json().get("results", [])
            candidates: list[RecallCandidate] = []
            for r in results:
                candidates.append(
                    RecallCandidate(
                        name=r["name"],
                        description=r.get("description", ""),
                        source="remote",
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

    def close(self) -> None:
        """关闭 HTTP 客户端连接"""
        self._client.close()
