"""SkillIndexer — embedding 索引管理

集中处理 embedding 生成、索引更新和向量字段写入。
"""

from __future__ import annotations

from typing import Iterable

from utils.logger import get_logger
from core.skill.schema import Skill
from core.skill.retrieval.embedding_recall import EmbeddingRecall

logger = get_logger(__name__)


class SkillIndexer:
    """Embedding 索引管理器

    负责：
    - embedding 生成与 sqlite-vec upsert
    """

    def __init__(self, embedding_recall: EmbeddingRecall | None = None):
        self._embedding_recall = embedding_recall

    @property
    def is_ready(self) -> bool:
        return bool(self._embedding_recall and self._embedding_recall.is_ready)

    async def ensure_ready(self) -> bool:
        if not self._embedding_recall:
            return False
        return await self._embedding_recall.ensure_ready_async()

    async def index(self, skill: Skill) -> bool:
        """为单个技能建立索引。"""
        if not self._embedding_recall or not self._embedding_recall.is_ready:
            return False

        try:
            text = skill.to_embedding_text()
            ok = await self._embedding_recall.upsert_async(skill.name, text)
            return ok
        except Exception as e:
            logger.warning("Skill index failed for '{}': {}", skill.name, e)
            return False

    async def index_batch(self, skills: Iterable[Skill]) -> int:
        """批量索引技能。"""
        if not self._embedding_recall or not self._embedding_recall.is_ready:
            return 0

        skills_list = list(skills)
        if not skills_list:
            return 0

        try:
            texts = [s.to_embedding_text() for s in skills_list]
            names = [s.name for s in skills_list]
            items = list(zip(names, texts))
            count = await self._embedding_recall.upsert_batch_async(items)
            return count
        except Exception as e:
            logger.warning("Batch skill index failed: {}", e)
            return 0

    def delete(self, skill_name: str) -> None:
        if not self._embedding_recall or not self._embedding_recall.is_ready:
            return
        self._embedding_recall.delete([skill_name])

    def cleanup_orphans(self, valid_names: set[str]) -> int:
        if not self._embedding_recall:
            return 0
        return self._embedding_recall.cleanup_orphans(valid_names)

