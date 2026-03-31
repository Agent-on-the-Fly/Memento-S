"""DB 元数据存储 - 管理 skills 表的元数据"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Set

from core.skill.config import SkillConfig
from core.skill.schema import Skill
from core.skill.store.base import SkillStorage
from middleware.storage.core.engine import DatabaseManager
from middleware.storage.models import Base
from middleware.storage.schemas import SkillCreate, SkillUpdate
from middleware.storage.services import SkillService
from core.utils.text import to_kebab_case, to_snake_case
from utils.logger import get_logger

logger = get_logger(__name__)


class DBStorage(SkillStorage):
    """DB 元数据存储 - 通过 SkillService 操作数据库"""

    def __init__(self, skill_service: Any) -> None:
        super().__init__()
        self._service = skill_service

    @classmethod
    async def from_config(cls, config: "SkillConfig") -> "DBStorage":
        """从配置创建 DBStorage 实例

        自动初始化数据库连接和 SkillService。

        Args:
            config: SkillConfig 配置

        Returns:
            初始化好的 DBStorage 实例
        """
        # 使用配置中的 db_path
        db_file = Path(config.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite+aiosqlite:///{db_file}"

        # 初始化数据库管理器（使用 from_config 确保单例被正确初始化）
        db_manager = await DatabaseManager.from_config(db_url=db_url, echo=False)

        # 创建表
        async with db_manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # 创建 SkillService（传入 db_manager，它会自动使用 session_factory）
        skill_service = SkillService(db_manager)

        storage = cls(skill_service)
        await storage.init()
        logger.info("DBStorage created from config: {}", db_file)
        return storage

    async def init(self) -> None:
        """检查 service 是否可用"""
        self._initialized = self._service is not None
        if self._initialized:
            logger.info("DBStorage initialized")
        else:
            logger.warning("DBStorage: no skill_service provided")

    async def close(self) -> None:
        """DB 连接由 service 管理"""
        pass

    async def save(
        self, name: str, skill: Skill, embedding: bytes | None = None
    ) -> None:
        """保存或更新 skill 元数据"""
        if not self._service:
            return

        try:
            db_name = to_snake_case(name or skill.name)
            if skill.source_dir:
                display_name = to_kebab_case(Path(skill.source_dir).name)
            else:
                display_name = to_kebab_case(skill.name)

            existing = await self._service.get_by_name(db_name)
            if existing:
                await self._service.update(
                    existing.id, SkillUpdate(description=skill.description)
                )
                logger.debug("Updated skill in DB: {}", db_name)
            else:
                await self._service.create(
                    SkillCreate(
                        name=db_name,
                        display_name=display_name,
                        description=skill.description,
                        source_type="local",
                        local_path=str(skill.source_dir or ""),
                        embedding=embedding,
                    )
                )
                logger.debug(
                    "Created skill in DB: {} (display_name: {})",
                    db_name,
                    display_name,
                )
        except Exception as e:
            logger.warning("Failed to save skill '{}' to DB: {}", name, e)

    async def load(self, name: str) -> Skill | None:
        """从 DB 加载元数据"""
        if not self._service:
            return None

        try:
            record = await self._service.get_by_name(name)
            if record:
                return Skill(
                    name=record.name,
                    description=record.description,
                    content="",  # DB 不存储完整内容
                    source_dir=record.local_path,
                )
        except Exception as e:
            logger.warning("Failed to load skill '{}' from DB: {}", name, e)

        return None

    async def delete(self, name: str) -> bool:
        """从 DB 删除"""
        if not self._service:
            return False

        try:
            existing = await self._service.get_by_name(name)
            if existing:
                await self._service.delete(existing.id)
                logger.debug("Deleted skill from DB: {}", name)
                return True
        except Exception as e:
            logger.warning("Failed to delete skill '{}' from DB: {}", name, e)

        return False

    async def list_names(self) -> Set[str]:
        """返回 DB 中所有 skill 名称"""
        if not self._service:
            return set()

        try:
            names = await self._service.list_all_names()
            return set(names)
        except Exception as e:
            logger.warning("Failed to list skill names from DB: {}", e)
            return set()
