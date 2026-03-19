"""multi_recall — 本地全量 + 云端检索 → 去重合并

本地：直接返回全部 local_cache 中的 skill
云端：RemoteCloudCatalog — HTTP API 检索

合并后按 name 去重（local 优先），返回候选列表供 LLM 选择。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from utils.logger import get_logger

from .remote_catalog import RemoteCloudCatalog

logger = get_logger(__name__)


@dataclass
class RecallCandidate:
    """召回候选"""

    name: str
    description: str
    source: Literal["local", "cloud"]
    score: float = 0.0
    match_type: str = ""
    skill: Any = None


class MultiRecall:
    """本地全量返回 + 云端检索 → 去重合并"""

    def __init__(self, cloud_catalog: RemoteCloudCatalog | None = None):
        self._cloud_catalog = cloud_catalog

    async def recall(
        self,
        query: str,
        local_cache: dict[str, Any],
        cloud_k: int = 5,
    ) -> list[RecallCandidate]:
        """本地全量 + 云端检索 → 去重合并（local 优先）。"""
        seen: dict[str, RecallCandidate] = {}

        for name, skill in local_cache.items():
            seen[name] = RecallCandidate(
                name=name,
                description=getattr(skill, "description", "") or "",
                source="local",
                score=1.0,
                match_type="local",
                skill=skill,
            )
        logger.debug("[RECALL] Local: {} skills returned", len(seen))

        if self._cloud_catalog:
            try:
                cloud_results = self._cloud_catalog.search(query, k=cloud_k)
                for info in cloud_results:
                    if info.name not in seen:
                        seen[info.name] = RecallCandidate(
                            name=info.name,
                            description=info.description,
                            source="cloud",
                            score=info.score,
                            match_type="cloud",
                            skill=None,
                        )
            except Exception as e:
                logger.warning("[RECALL] Cloud recall failed: {}", e)

        candidates = list(seen.values())
        local_count = sum(1 for c in candidates if c.source == "local")
        cloud_count = len(candidates) - local_count
        logger.info(
            "[RECALL] MultiRecall: query={} → {} candidates (local={}, cloud={})",
            query, len(candidates), local_count, cloud_count,
        )

        return candidates
