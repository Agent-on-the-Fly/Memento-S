"""本地文件存储 - 管理磁盘上的 SKILL.md 和 scripts"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Set

from core.skill.schema import Skill
from core.skill.store.base import SkillStorage
from core.skill.loader import SkillLoader
from core.skill.builder import SkillBuilder
from core.utils.text import to_kebab_case
from utils.logger import get_logger

logger = get_logger(__name__)


class FileStorage(SkillStorage):
    """本地文件存储 - 管理磁盘上的 SKILL.md 和 scripts 目录"""

    def __init__(self, skills_dir: Path) -> None:
        super().__init__()
        self._skills_dir = Path(skills_dir)
        self._loader = SkillLoader(self._skills_dir)

    async def init(self) -> None:
        """创建 skills 目录"""
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        logger.info("FileStorage initialized: {}", self._skills_dir)

    async def close(self) -> None:
        """文件存储无需关闭"""
        pass

    async def save(self, name: str, skill: Skill) -> None:
        """保存 skill 到磁盘

        使用 SkillBuilder 构建规范的目录结构，然后保存。
        """
        # 优先使用已存在的 source_dir，否则使用 name 的 kebab-case
        if skill.source_dir and Path(skill.source_dir).exists():
            skill_dir = Path(skill.source_dir)
        else:
            kebab_name = to_kebab_case(name)
            skill_dir = self._skills_dir / kebab_name

        # 使用 SkillBuilder 构建
        builder = SkillBuilder()
        built = builder.build(skill, skill_dir)

        logger.debug("Saved skill to disk: {}", built.skill_dir)

    async def load(self, name: str) -> Skill | None:
        """从磁盘加载单个 skill

        使用 SkillLoader 加载 skill。
        """
        try:
            return self._loader.load_from_dir(self._skills_dir / to_kebab_case(name))
        except FileNotFoundError:
            return None

    async def delete(self, name: str) -> bool:
        """删除 skill 目录"""
        kebab_name = to_kebab_case(name)
        skill_dir = self._skills_dir / kebab_name

        if skill_dir.exists():
            shutil.rmtree(skill_dir)
            logger.info("Deleted skill directory: {}", skill_dir)
            return True
        return False

    async def list_names(self) -> Set[str]:
        """扫描目录返回所有 skill 名称"""
        names: Set[str] = set()
        if not self._skills_dir.exists():
            return names

        for skill_dir in self._skills_dir.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                try:
                    skill = self._loader.load_from_dir(skill_dir)
                    names.add(skill.name)
                except Exception as e:
                    logger.warning(
                        "Failed to load skill from '{}': {}", skill_dir.name, e
                    )

        return names
