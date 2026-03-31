"""Skill 存储协调层 - 组合三种存储后端"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from core.skill.config import SkillConfig
from core.skill.embedding import EmbeddingGenerator
from core.skill.schema import Skill
from core.skill.store.base import SkillStorage
from core.skill.store.db_storage import DBStorage
from core.skill.store.file_storage import FileStorage
from core.skill.store.vector_storage import VectorStorage
from core.utils.text import to_kebab_case, to_snake_case
from utils.logger import get_logger

logger = get_logger(__name__)


class SkillStore:
    """
    Skill 存储协调层

    组合三种存储后端：
    - file: 磁盘文件存储
    - db: 数据库元数据存储
    - vector: 向量索引存储

    使用方式：
        # 方式1：从配置异步构建（推荐用于生产）
        # 内部自动创建 DBStorage、EmbeddingGenerator 等依赖
        store = await SkillStore.from_config(config)

        # 方式2：显式传入（用于测试/自定义）
        store = SkillStore(file_storage, db_storage, vector_storage, embedding_generator)
    """

    def __init__(
        self,
        file_storage: SkillStorage,
        db_storage: SkillStorage,
        vector_storage: SkillStorage,
        embedding_generator: EmbeddingGenerator,
    ) -> None:
        # 三个存储都是必须的参数
        self._file = file_storage
        self._db = db_storage
        self._vector = vector_storage
        self._embedding_generator = embedding_generator

        # 从 file_storage 获取 loader
        if isinstance(file_storage, FileStorage):
            self._loader = file_storage._loader
        else:
            # 对于非 FileStorage 情况，不创建 loader
            self._loader = None

    @classmethod
    async def from_config(
        cls,
        config: SkillConfig,
    ) -> SkillStore:
        """
        从配置构建 SkillStore（工厂方法）

        内部自动处理所有依赖（DB、embedding 等）的创建，无需外部传入。

        Args:
            config: SkillConfig 配置

        Returns:
            初始化好的 SkillStore 实例
        """
        # 1. 文件存储
        file_storage = FileStorage(config.skills_dir)
        await file_storage.init()

        # 2. DB 存储（内部自动创建）
        db_storage = await DBStorage.from_config(config)
        await db_storage.init()

        # 3. 向量存储
        vector_storage = await VectorStorage.from_config(config)
        await vector_storage.init()

        # 4. 创建 embedding generator
        embedding_generator = EmbeddingGenerator.from_config(config)
        logger.info("Embedding generator created from config")

        store = cls(file_storage, db_storage, vector_storage, embedding_generator)
        logger.info("SkillStore created from config")
        return store

    async def close(self) -> None:
        """关闭所有存储"""
        await self._file.close()
        await self._db.close()
        await self._vector.close()

    # ========== 核心操作 ==========

    async def add_skill(self, skill: Skill) -> None:
        """
        添加 skill 到所有存储

        SkillStore 内部自动处理 embedding 生成（如果已配置）。

        Args:
            skill: Skill 对象
        """
        # 内部统一 snake_case
        internal_name = to_snake_case(skill.name)

        # 保存到 DB（元数据）
        await self._db.save(internal_name, skill)

        # 生成并保存向量（统一使用 snake_case 作为 key）
        vector = await self._embedding_generator.generate_for_skill(skill)
        if vector:
            await self._vector.save(internal_name, vector)

        logger.info("Added skill: {} (id: {})", internal_name, internal_name)

    async def remove_skill(self, name: str) -> bool:
        """从所有存储删除 skill"""

        # 首先尝试获取 skill，优先按内部名加载
        skill = self._loader.load(name) if self._loader else None

        # 如果从文件加载失败，尝试从 DB 查找（处理 catalog name 和实际名称不一致的情况）
        if not skill and self._loader:
            db_skill = await self._db.load(to_snake_case(name))
            if db_skill and db_skill.source_dir:
                # 使用 source_dir 的目录名重新加载
                alt_name = Path(db_skill.source_dir).name
                skill = self._loader.load(alt_name)

        internal_name = to_snake_case(skill.name) if skill else to_snake_case(name)
        storage_name = to_kebab_case(internal_name)

        file_deleted = await self._file.delete(storage_name)
        await self._db.delete(internal_name)
        await self._vector.delete(internal_name)

        if file_deleted:
            logger.info(
                "Removed skill: {} (internal: {}, storage: {})",
                name,
                internal_name,
                storage_name,
            )

        return file_deleted

    async def get_skill(self, name: str) -> Skill | None:
        """获取 skill（从文件加载完整数据）"""
        return self._loader.load(name) if self._loader else None

    async def list_all_skills(self) -> Dict[str, Skill]:
        """加载所有 skills"""
        # Fallback
        skills: Dict[str, Skill] = {}
        names = await self._file.list_names()
        for name in names:
            try:
                skill = self._loader.load(name) if self._loader else None
                if skill:
                    skills[name] = skill
            except ValueError as e:
                logger.warning("Failed to load skill '{}': {}", name, e)
                continue
        return skills

    async def load_from_path(self, path: Path) -> Skill:
        """从指定路径加载 skill 并添加到所有存储（DB + Vector）

        用于云端下载后加载 skill 到本地存储。

        Args:
            path: skill 目录路径

        Returns:
            加载的 Skill 对象
        """
        # 从文件加载（目录名 kebab -> 内部名 snake）
        skill = await self._file.load(to_snake_case(path.name))

        if not skill:
            raise FileNotFoundError(f"Cannot load skill from {path}")

        # 使用 add_skill 完成 DB + Vector 同步
        await self.add_skill(skill)

        return skill

    # ========== 同步操作 ==========

    async def refresh_from_disk(self) -> int:
        """
        增量扫描 skills/ 目录，将新增的 skill 加载到所有存储（DB + Vector）

        Returns:
            新增 skill 数量
        """
        added = 0

        # 获取文件存储目录
        skills_dir = getattr(self._file, "_skills_dir", None)
        if not skills_dir or not skills_dir.exists():
            return added

        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
                continue
            try:
                skill = await self._file.load(to_snake_case(skill_dir.name))
                if skill:
                    # 使用 add_skill 完成 DB + Vector 同步
                    await self.add_skill(skill)
                    added += 1
                    logger.info("Hot-loaded new skill from disk: {}", skill.name)
            except Exception as e:
                logger.warning("refresh_from_disk: skip '{}': {}", skill_dir.name, e)

        if added:
            logger.info("refresh_from_disk: {} new skill(s) added", added)

        return added

    async def sync_from_disk(self) -> int:
        """
        从磁盘同步到 DB（新版方法名）

        Returns:
            同步的 skill 数量
        """
        skills = await self.list_all_skills()

        for skill in skills.values():
            await self._db.save(skill.name, skill)

        logger.info("Synced {} skills to DB", len(skills))
        return len(skills)

    async def sync_vectors(self, vectors: Dict[str, List[float]]) -> int:
        """
        批量同步向量（由 Indexer 调用）

        Args:
            vectors: {skill_name: vector} 字典

        Returns:
            同步的向量数量
        """
        count = 0
        for name, vector in vectors.items():
            await self._vector.save(name, vector)
            count += 1

        logger.info("Synced {} vectors", count)
        return count

    async def cleanup_orphans(self) -> List[str]:
        """
        清理孤儿记录（DB/向量中存在但文件不存在的 skill）

        Returns:
            清理的 skill 名称列表
        """
        file_names = await self._file.list_names()
        db_names = await self._db.list_names()
        vector_names = await self._vector.list_names()

        cleaned: List[str] = []

        # 清理 DB 中的孤儿
        for name in db_names - file_names:
            await self._db.delete(name)
            cleaned.append(name)

        # 清理向量库中的孤儿（统一 snake 比较）
        file_names_snake = {to_snake_case(n) for n in file_names}
        vector_names_snake = {to_snake_case(n) for n in vector_names}
        for name in vector_names_snake - file_names_snake:
            await self._vector.delete(name)
            if name not in cleaned:
                cleaned.append(name)

        if cleaned:
            logger.info("Cleaned up {} orphaned skill(s)", len(cleaned))

        return cleaned

    # ========== 属性访问 ==========

    @property
    def file_storage(self) -> SkillStorage:
        return self._file

    @property
    def db_storage(self) -> SkillStorage:
        return self._db

    @property
    def vector_storage(self) -> SkillStorage:
        return self._vector

    @property
    def embedding_generator(self):
        return self._embedding_generator

    @property
    def skill_loader(self):
        return self._loader
