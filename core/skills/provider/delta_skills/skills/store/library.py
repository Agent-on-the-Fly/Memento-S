
from __future__ import annotations

import shutil
from typing import List

from core.config import g_settings
from core.config.logging import get_logger
from ...retrieval import BM25Index, CrossEncoderReranker, HybridRetriever, ScoredSkill
from ...schema import Skill
from .embedding import EmbeddingStore
from .persistence import (
    get_skill_mtime,
    load_all_skills,
    load_skill_from_dir,
    save_skill_to_disk,
    to_kebab_case,
)

logger = get_logger(__name__)


class SkillLibrary:

    def __init__(self, version_manager=None):
        self.workspace = g_settings.workspace_path
        logger.warning("workspace = %s", self.workspace)

        self.skills_directory = self.workspace / "skills"
        self.skills_directory.mkdir(parents=True, exist_ok=True)

        self.version_manager = version_manager

        self.local_cache: dict[str, Skill] = {}
        self.bm25_index = BM25Index()
        self._on_change_callbacks: list = []

        self._embedding = EmbeddingStore(self.workspace)

        self.local_cache = load_all_skills(self.skills_directory, self.version_manager)

        if self.local_cache:
            self.bm25_index.build(self.local_cache)


    def add_skill(self, skill: Skill, change_type: str = "create", change_note: str = ""):
        is_update = skill.name in self.local_cache
        effective_type = change_type if change_type != "create" else ("update" if is_update else "create")

        save_skill_to_disk(skill, self.skills_directory)
        self.local_cache[skill.name] = skill

        self._embedding.ensure(self.local_cache, self.skills_directory)
        self._embedding.upsert(skill)
        self.bm25_index.add(skill)
        HybridRetriever.invalidate_cache()

        if self.version_manager:
            try:
                sv = self.version_manager.save_version(skill, effective_type, change_note)
                skill.version = sv.version
                logger.info("Skill stored: %s (v%d, %s)", skill.name, sv.version, effective_type)
            except Exception as e:
                logger.warning("Version creation failed for '%s': %s", skill.name, e)
        else:
            logger.info("Skill stored: %s", skill.name)

        self._notify_change()

    def remove_skill(self, skill_name: str) -> bool:
        if skill_name not in self.local_cache:
            return False

        for dirname in [skill_name, skill_name.replace("_", "-")]:
            skill_dir = self.skills_directory / dirname
            if skill_dir.exists():
                shutil.rmtree(skill_dir)
                logger.info("Removed skill directory: %s", skill_dir)
                break

        try:
            self._embedding.ensure(self.local_cache, self.skills_directory)
            self._embedding.delete([skill_name])
            logger.info("Removed from ChromaDB: %s", skill_name)
        except Exception as e:
            logger.debug("ChromaDB cleanup skipped: %s", e)

        if skill_name in self._embedding.mtime_cache:
            del self._embedding.mtime_cache[skill_name]
            self._embedding.save_mtime_cache()

        del self.local_cache[skill_name]
        self.bm25_index.build(self.local_cache)
        from ..retrieval.hybrid import HybridRetriever
        HybridRetriever.invalidate_cache()
        self._notify_change()

        logger.info("Skill '%s' removed successfully", skill_name)
        return True

    def retrieve_skills(
        self,
        query: str,
        k: int = 5,
        min_score: float = None,
        extra_queries: List[str] = None,
    ) -> List[ScoredSkill]:
        self._embedding.ensure(self.local_cache, self.skills_directory)

        if self._embedding.collection is None:
            if self.bm25_index.is_built:
                results = self.bm25_index.search(query, k=k)
                effective_min = min_score if min_score is not None else g_settings.retrieval_min_score
                scored = []
                for name, score in results:
                    if score < effective_min:
                        continue
                    if name in self.local_cache:
                        scored.append(ScoredSkill(
                            skill=self.local_cache[name],
                            score=score,
                            bm25_score=score,
                        ))
                return scored[:k]
            return []

        if self.bm25_index.is_built:
            reranker = None
            if g_settings.reranker_enabled:
                try:
                    reranker = CrossEncoderReranker.get_instance()
                except Exception as e:
                    logger.debug("Reranker unavailable, proceeding without: %s", e)

            retriever = HybridRetriever(
                bm25_index=self.bm25_index,
                chroma_collection=self._embedding.collection,
                skill_cache=self.local_cache,
                reranker=reranker,
            )
            effective_min = min_score if min_score is not None else g_settings.retrieval_min_score
            return retriever.retrieve(
                query, k=k, min_score=effective_min,
                extra_queries=extra_queries,
            )

        logger.debug("BM25 index not ready, falling back to ChromaDB only")
        return self._embedding.query(query, k=k, skill_cache=self.local_cache)

    def sync_all(self, force: bool = False):
        logger.info("Starting full database sync (force=%s)...", force)

        if force:
            self.local_cache.clear()
            self.local_cache = load_all_skills(self.skills_directory, self.version_manager)

        self._embedding.ensure(self.local_cache, self.skills_directory)

        for name, skill in self.local_cache.items():
            try:
                if self.version_manager:
                    latest = self.version_manager.get_latest_version(skill.name)
                    if not latest:
                        logger.info("Syncing '%s' to SQLite (missing)", name)
                        self.version_manager.save_version(
                            skill,
                            change_type="syncdb",
                            change_note="Restored via syncdb command",
                        )

                if force:
                    self._embedding.upsert(skill)
                    skill_dir = self.skills_directory / to_kebab_case(name)
                    self._embedding.mtime_cache[name] = get_skill_mtime(skill_dir)

            except Exception as e:
                logger.error("Failed to sync skill '%s': %s", name, e)

        if force:
            self._embedding.save_mtime_cache()

        logger.info("Sync completed. Processed %d skills.", len(self.local_cache))

    def refresh_from_disk(self) -> int:
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
                    self._embedding.upsert(skill)
                    added += 1
                    logger.info("Hot-loaded new skill from disk: %s", skill.name)
            except Exception as e:
                logger.debug("refresh_from_disk: skip '%s': %s", skill_dir.name, e)

        if added:
            self.bm25_index.build(self.local_cache)
            HybridRetriever.invalidate_cache()
            self._notify_change()
            logger.info("refresh_from_disk: %d new skill(s) added", added)

        return added

    def on_change(self, callback):
        self._on_change_callbacks.append(callback)

    def destroy_chroma(self):
        self._embedding.destroy()


    def _notify_change(self):
        for cb in self._on_change_callbacks:
            try:
                cb(self.local_cache)
            except Exception as e:
                logger.debug("on_change callback failed: %s", e)
