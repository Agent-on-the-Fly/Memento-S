"""远程 Skill Retrieval API 客户端

通过 HTTP 调用独立部署的 skill_retrieval_api 微服务，
提供 search（检索）和 download（下载）接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from utils.logger import get_logger
from core.skill.importers.factory import create_default_download_manager

logger = get_logger(__name__)


@dataclass
class RemoteSkillInfo:
    """远程检索返回的 skill 信息"""

    name: str
    description: str
    score: float = 0.5


class RemoteCloudCatalog:
    """远程 Skill Retrieval API 客户端

    提供 2 个接口：
    - POST /api/v1/search    — 检索 skill，返回 top-k 的 name + description
    - POST /api/v1/download  — 按 skill_name 下载 skill 到本地
    """

    def __init__(self, base_url: str, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._embedding_ready: bool = False
        self._size: int = 0

        try:
            resp = self._client.get(f"{self._base_url}/health")
            if resp.status_code == 200:
                data = resp.json()
                self._embedding_ready = data.get("embedding_ready", False)
                self._size = data.get("catalog_size", 0) or data.get("total_skills", 0)
                logger.info(
                    "RemoteCloudCatalog connected: {} (skills={}, embedding={})",
                    self._base_url, self._size, self._embedding_ready,
                )
        except Exception as e:
            logger.warning("RemoteCloudCatalog health check failed: {}", e)

    @property
    def embedding_ready(self) -> bool:
        return self._embedding_ready

    @property
    def size(self) -> int:
        return self._size

    def search(self, query: str, k: int = 5) -> list[RemoteSkillInfo]:
        """搜索 skill，返回 top_k 的 name + description"""
        try:
            resp = self._client.post(
                f"{self._base_url}/api/v1/search",
                json={"query": query, "top_k": k},
            )
            if resp.status_code != 200:
                logger.warning("Remote search failed: HTTP {}", resp.status_code)
                return []

            results = resp.json().get("results", [])
            return [
                RemoteSkillInfo(
                    name=r["name"],
                    description=r.get("description", ""),
                    score=r.get("score", 0.0),
                )
                for r in results
            ]
        except Exception as e:
            logger.warning("Remote search error: {}", e)
            return []

    def download(self, skill_name: str, target_dir: Path) -> Path | None:
        """从云端下载 skill 到本地目录，返回本地路径，失败返回 None。"""
        try:
            resp = self._client.post(
                f"{self._base_url}/api/v1/download",
                json={"skill_name": skill_name},
            )
            if resp.status_code != 200:
                logger.warning(
                    "Remote download failed for '{}': HTTP {}", skill_name, resp.status_code,
                )
                return None

            github_url = resp.json().get("github_url", "")
            if not github_url:
                logger.warning("Remote download: no github_url for '{}'", skill_name)
                return None

            download_manager = create_default_download_manager()
            return download_manager.download(github_url, target_dir, skill_name)

        except Exception as e:
            logger.warning("Remote download error for '{}': {}", skill_name, e)
            return None
