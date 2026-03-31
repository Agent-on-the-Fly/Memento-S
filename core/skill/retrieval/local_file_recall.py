"""local_file_recall — 本地文件遍历召回

直接扫描 skills 目录，解析 SKILL.md 文件，无需依赖 SkillStore。
使用文件系统元数据缓存实现快速变更检测。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.skill.schema import Skill
from core.skill.loader import SkillLoader
from utils.logger import get_logger
from .base import BaseRecall
from .schema import RecallCandidate

logger = get_logger(__name__)


@dataclass
class _SkillCacheEntry:
    """单个技能的缓存条目"""

    skill: "Skill"
    mtime: float  # SKILL.md 的修改时间
    size: int  # SKILL.md 的文件大小


@dataclass
class _ScanState:
    """扫描状态缓存"""

    skills: dict[str, _SkillCacheEntry] = field(default_factory=dict)
    dir_signature: str = ""  # 目录状态签名
    last_scan_time: float = 0.0


class LocalFileRecall(BaseRecall):
    """本地文件遍历召回

    直接扫描 skills 目录，解析 SKILL.md 文件，无需依赖 SkillStore。
    使用文件系统元数据缓存实现快速变更检测。

    Args:
        skills_dir: skills 目录路径
    """

    @staticmethod
    async def load_full_skill(skills_dir: Path | str, name: str) -> Optional["Skill"]:
        """按名称加载完整 skill（含 scripts / references）。

        作为检索层的一部分，集中承载 load 逻辑。
        """
        from core.skill.loader import SkillLoader
        from core.utils.text import to_kebab_case

        loader = SkillLoader(Path(skills_dir))

        normalized = name.replace("-", "_")
        alt = to_kebab_case(name).replace("-", "_")
        return loader.load(normalized) or loader.load(alt)

    def __init__(self, skills_dir: Path | str):
        self._skills_dir = Path(skills_dir)
        self._loader = SkillLoader(self._skills_dir)  # 注入 SkillLoader
        self._state = _ScanState()
        self._lock = False  # 简单锁，防止并发扫描

    @classmethod
    def from_config(cls, config: "SkillConfig") -> "LocalFileRecall":
        """从配置创建 LocalFileRecall 实例

        Args:
            config: SkillConfig 配置

        Returns:
            LocalFileRecall 实例
        """
        return cls(config.skills_dir)

    @property
    def name(self) -> str:
        """召回策略名称"""
        return "local_file"

    def is_available(self) -> bool:
        """检查目录是否存在"""
        return self._skills_dir.exists() and self._skills_dir.is_dir()

    def _compute_dir_signature(self) -> str:
        """计算目录状态签名，用于快速检测变更。

        基于以下因素：
        1. 子目录数量（每个 skill 一个目录）
        2. 每个子目录下 SKILL.md 的修改时间和大小

        Returns:
            目录状态签名（MD5 哈希）
        """
        sig_parts = []

        root_dir = self._skills_dir
        if not root_dir or not root_dir.exists():
            return hashlib.md5(b"").hexdigest()

        # 获取所有包含 SKILL.md 的子目录
        skill_dirs = []
        for item in sorted(root_dir.iterdir()):
            if item.is_dir():
                skill_md = item / "SKILL.md"
                if skill_md.exists():
                    try:
                        stat = skill_md.stat()
                        # 使用目录名+修改时间+大小作为签名的一部分
                        skill_dirs.append(
                            f"{item.name}:{stat.st_mtime:.6f}:{stat.st_size}"
                        )
                    except (OSError, IOError):
                        continue

        sig_parts.extend(sorted(skill_dirs))

        # 计算整体签名
        content = "|".join(sig_parts)
        return hashlib.md5(content.encode()).hexdigest()

    def _has_changes(self) -> bool:
        """检查 skills 目录是否有变更。

        Returns:
            True 如果有变更，False 如果没有变更
        """
        current_sig = self._compute_dir_signature()
        return current_sig != self._dir_signature

    @property
    def _dir_signature(self) -> str:
        """获取当前缓存的目录签名"""
        return self._state.dir_signature

    def _load_skill_from_dir(self, skill_dir: Path) -> Optional["Skill"]:
        """从单个 skill 目录加载 Skill 对象（轻量级版本）。

        使用 SkillLoader 轻量级模式，只解析元数据，不加载 scripts/references。

        Args:
            skill_dir: skill 目录路径

        Returns:
            Skill 对象，解析失败返回 None
        """
        try:
            # 使用 SkillLoader 的轻量级模式（full=False）
            return self._loader.load_from_dir(skill_dir, full=False)
        except Exception as e:
            logger.warning("Failed to load skill from '{}': {}", skill_dir, e)
            return None

    def _scan_directory(self, root_dir: Path | None) -> dict[str, _SkillCacheEntry]:
        """扫描单个目录下的所有 skills。

        Args:
            root_dir: 要扫描的目录

        Returns:
            skill 名称到缓存条目的映射
        """
        result = {}
        if not root_dir or not root_dir.exists():
            return result

        for item in sorted(root_dir.iterdir()):
            if not item.is_dir():
                continue

            skill_md = item / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                stat = skill_md.stat()
                mtime = stat.st_mtime
                size = stat.st_size

                # 检查缓存中是否存在且未变更
                skill_name = item.name.replace("-", "_")

                # 尝试使用缓存
                if skill_name in self._state.skills:
                    entry = self._state.skills[skill_name]
                    if entry.mtime == mtime and entry.size == size:
                        # 未变更，使用缓存
                        result[skill_name] = entry
                        continue

                # 需要重新加载
                skill = self._load_skill_from_dir(item)
                if skill:
                    # 使用实际解析的 skill name
                    skill_name = skill.name
                    result[skill_name] = _SkillCacheEntry(
                        skill=skill, mtime=mtime, size=size
                    )

            except (OSError, IOError) as e:
                logger.warning("Failed to stat '{}': {}", skill_md, e)
                continue

        return result

    def _refresh_cache(self) -> None:
        """刷新缓存，重新扫描所有 skills 目录。"""
        if self._lock:
            logger.debug("Scan already in progress, skipping")
            return

        self._lock = True
        try:
            start_time = time.time()

            # 扫描 skills 目录
            new_skills = {}
            new_skills.update(self._scan_directory(self._skills_dir))

            # 更新状态
            self._state.skills = new_skills
            self._state.dir_signature = self._compute_dir_signature()
            self._state.last_scan_time = time.time()

            elapsed = (time.time() - start_time) * 1000
            logger.debug(
                "[LOCAL_FILE_RECALL] Cache refreshed: {} skills in {:.1f}ms",
                len(new_skills),
                elapsed,
            )

        finally:
            self._lock = False

    async def search(self, query: str, k: int = 0, **kwargs) -> list["RecallCandidate"]:
        """搜索本地 skills。

        注意：对于本地文件召回，query 参数被忽略（全量返回所有 skills）。
        k 参数也被忽略（返回所有 skills），除非未来实现过滤功能。

        Args:
            query: 搜索查询（当前版本忽略）
            k: 返回结果数限制（当前版本忽略，返回所有）
            **kwargs: 额外参数

        Returns:
            RecallCandidate 列表
        """
        # 检查是否需要刷新缓存
        if self._has_changes():
            self._refresh_cache()

        # 从缓存构建召回结果
        candidates: list[RecallCandidate] = []
        for name, entry in self._state.skills.items():
            skill = entry.skill
            candidates.append(
                RecallCandidate(
                    name=name,
                    description=skill.description or "",
                    source="local",
                    score=1.0,  # 本地全量返回，默认最高分数
                    match_type="local_file",
                    skill=skill,
                )
            )

        logger.debug("[LOCAL_FILE_RECALL] Returned {} local skills", len(candidates))
        return candidates

    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = super().get_stats()
        stats.update(
            {
                "skills_dir": str(self._skills_dir),
                "cached_skills": len(self._state.skills),
                "last_scan_time": self._state.last_scan_time,
            }
        )
        return stats
