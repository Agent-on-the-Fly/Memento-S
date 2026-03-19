"""SkillStore — 技能库持久化封装

磁盘 SKILL.md 持久化 + local_cache + DB 元数据同步。

    Agent  ←→  Provider  ←→  SkillStore
                                ├── persistence (磁盘 SKILL.md I/O)
                                └── SkillService (DB 元数据)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from middleware.config import g_config
from utils.logger import get_logger
from core.skill.schema import Skill
from core.skill.store.persistence import (
    load_all_skills,
    load_skill_from_dir,
    save_skill_to_disk,
)

logger = get_logger(__name__)


class SkillStore:
    """技能库：磁盘持久化 + local_cache + DB 元数据

    目录结构:
        workspace/
        ├── skills/
        │   ├── get-weather-mock/
        │   │   ├── SKILL.md
        │   │   └── scripts/
        │   │       └── get_weather_mock.py
        │   └── ...
        ├── data/memento_s.db
    """

    def __init__(
        self,
        skill_service: Any | None = None,
        embedding_client=None,
    ):
        self._skill_service = skill_service  # middleware SkillService (async)
        self._embedding_client = embedding_client

        self.skills_directory = g_config.get_skills_path()
        self.local_cache: dict[str, Skill] = load_all_skills(self.skills_directory)
        logger.info(
            f"SkillStore init: skills_dir={self.skills_directory}, "
            f"local_cache={len(self.local_cache)}, "
            f"names={sorted(self.local_cache.keys())}"
        )

    # ── Public API ────────────────────────────────────────────────

    async def add_skill(self, skill: Skill) -> None:
        """注册技能到磁盘 + 内存缓存 + DB

        Args:
            skill: 技能对象
        """
        embedding = await self._embed_skill(skill)
        save_skill_to_disk(skill, self.skills_directory)
        self.local_cache[skill.name] = skill
        await self._upsert_to_db(skill, embedding=embedding)
        logger.info("Skill stored: {}", skill.name)

    async def remove_skill(self, skill_name: str) -> bool:
        """从技能库完全删除一个 skill

        删除内容:
            1. 文件系统: skills/<name>/ 目录
            2. 内存缓存: local_cache 中移除
            3. DB: skills 表记录
        """
        if skill_name not in self.local_cache:
            return False

        for dirname in [skill_name, skill_name.replace("_", "-")]:
            skill_dir = self.skills_directory / dirname
            if skill_dir.exists():
                shutil.rmtree(skill_dir)
                logger.info("Removed skill directory: {}", skill_dir)
                break

        del self.local_cache[skill_name]

        await self._delete_from_db(skill_name)

        logger.info("Skill '{}' removed", skill_name)
        return True

    async def refresh_from_disk(self) -> int:
        """增量扫描 skills/ 目录，将磁盘上新增的 skill 加载到 local_cache + DB。

        仅加载 local_cache 中尚不存在的目录。
        返回新增 skill 数量。
        """
        added = 0
        if not self.skills_directory.exists():
            return added

        for skill_dir in sorted(self.skills_directory.iterdir()):
            if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
                continue
            try:
                skill = load_skill_from_dir(skill_dir)
                if skill.name not in self.local_cache:
                    self.local_cache[skill.name] = skill
                    await self._upsert_to_db(skill)
                    added += 1
                    logger.info("Hot-loaded new skill from disk: {}", skill.name)
            except Exception as e:
                logger.debug("refresh_from_disk: skip '{}': {}", skill_dir.name, e)

        if added:
            logger.info("refresh_from_disk: {} new skill(s) added", added)

        return added

    async def cleanup_orphaned_skills(self, indexer=None) -> list[str]:
        """清理磁盘已删除但 DB/向量库残留的孤儿 skill。"""
        if not self._skill_service:
            logger.debug("No skill service available, skipping orphan cleanup")
            return []

        try:
            db_skill_names = await self._skill_service.list_all_names()
            orphans = db_skill_names - set(self.local_cache.keys())

            if not orphans:
                logger.debug("No orphaned skills found in DB")
                return []

            cleaned = []
            for skill_name in orphans:
                try:
                    if indexer and indexer.is_ready:
                        indexer.delete(skill_name)
                        logger.debug("Deleted embedding for skill: {}", skill_name)
                    await self._delete_from_db(skill_name)
                    cleaned.append(skill_name)
                    logger.info("Cleaned up orphaned skill: {}", skill_name)
                except Exception as e:
                    logger.warning(
                        "Failed to clean up orphaned skill '{}': {}", skill_name, e
                    )

            if cleaned:
                logger.info(
                    "Cleanup complete: {} orphaned skill(s) removed", len(cleaned)
                )
            return cleaned

        except Exception as e:
            logger.warning("Failed to cleanup orphaned skills: {}", e)
            return []

    async def sync_all_to_db(self):
        """启动时将所有 local_cache 中的 skill 同步到 DB

        幂等操作：已存在的更新描述，不存在的创建。
        """
        if not self._skill_service or not self.local_cache:
            return

        synced = 0
        for skill in self.local_cache.values():
            try:
                await self._upsert_to_db(skill)
                synced += 1
            except Exception as e:
                logger.debug("sync_all_to_db: skip '{}': {}", skill.name, e)

        if synced:
            logger.info("Synced {} skill(s) to DB", synced)

    def find_by_name(self, name: str) -> Skill | None:
        """按名称查找 skill，支持多种命名格式。

        支持 snake_case、kebab-case、以及带连字符的变体。

        Args:
            name: skill 名称（支持多种格式）

        Returns:
            Skill 对象或 None
        """
        from core.skill.store.persistence import to_kebab_case

        # 标准化：统一为 snake_case
        normalized = name.replace("-", "_")
        if normalized in self.local_cache:
            return self.local_cache[normalized]

        # 尝试 kebab-case 变体
        alt = to_kebab_case(name).replace("-", "_")
        if alt in self.local_cache:
            return self.local_cache[alt]

        return None

    async def load_from_path(self, path: Path) -> Skill:
        """从指定路径加载 skill 并添加到缓存。

        用于云端下载后加载 skill 到本地存储。

        Args:
            path: skill 目录路径

        Returns:
            加载的 Skill 对象

        Raises:
            FileNotFoundError: 如果 SKILL.md 不存在
            ValueError: 如果解析失败
        """
        from core.skill.store.persistence import load_skill_from_dir

        skill = load_skill_from_dir(path)
        self.local_cache[skill.name] = skill
        await self._upsert_to_db(skill)
        logger.info("Skill loaded from path and added to cache: {}", skill.name)
        return skill

    # ── Internal: DB ─────────────────────────────────────────────

    async def _upsert_to_db(self, skill: Skill, embedding: bytes | None = None):
        """写入或更新 DB 元数据"""
        if not self._skill_service:
            return

        try:
            from middleware.storage.schemas import SkillCreate, SkillUpdate

            existing = await self._skill_service.get_by_name(skill.name)
            if existing:
                await self._skill_service.update(
                    existing.id,
                    SkillUpdate(description=skill.description),
                )
            else:
                await self._skill_service.create(
                    SkillCreate(
                        name=skill.name,
                        description=skill.description,
                        source_type="local",
                        local_path=str(skill.source_dir or ""),
                        embedding=embedding,
                    )
                )
        except Exception as e:
            logger.warning("DB upsert failed for '{}': {}", skill.name, e)

    async def _delete_from_db(self, skill_name: str):
        """从 DB 删除元数据"""
        if not self._skill_service:
            return

        try:
            existing = await self._skill_service.get_by_name(skill_name)
            if existing:
                await self._skill_service.delete(existing.id)
        except Exception as e:
            logger.warning("DB delete failed for '{}': {}", skill_name, e)

    async def _embed_skill(self, skill: Skill) -> bytes | None:
        if not self._embedding_client:
            return None
        try:
            vecs = await self._embedding_client.embed([skill.to_embedding_text()])
            if not vecs:
                return None
            from core.skill.embedding.utils import serialize_f32

            return serialize_f32(vecs[0])
        except Exception as e:
            logger.warning("Skill embedding failed for '{}': {}", skill.name, e)
            return None
