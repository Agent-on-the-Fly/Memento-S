# SPDX-License-Identifier: Apache-2.0
"""Skill 系统初始化器 - 内置技能同步。"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from shared.schema import SkillConfig

logger = get_logger(__name__)


class SkillInitializer:
    """Skill 系统初始化器。

    负责：
    1. 同步内置 skills 到工作目录

    注册表同步统一由 SkillStorage.from_config() 完成后，
    此初始化器只负责文件系统层面的 skills 同步。

    不再依赖 DB 或 Vector 存储。

    Usage:
        initializer = SkillInitializer(config)
        result = await initializer.initialize()
    """

    def __init__(self, config: SkillConfig):
        self._config = config
        self._builtin_root = config.builtin_skills_dir

    def _sha256_file(self, path: Path) -> str:
        """计算文件 SHA256。"""
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _build_skill_manifest(self, skill_dir: Path) -> dict[str, tuple[int, str]]:
        """构建 skill 指纹清单：relative_path -> (size, sha256)。"""
        manifest: dict[str, tuple[int, str]] = {}
        include_paths: list[Path] = []

        skill_md = skill_dir / "SKILL.md"
        if skill_md.is_file():
            include_paths.append(skill_md)

        scripts_dir = skill_dir / "scripts"
        if scripts_dir.is_dir():
            include_paths.extend(p for p in scripts_dir.rglob("*") if p.is_file())

        for p in sorted(include_paths, key=lambda x: x.as_posix()):
            rel = p.relative_to(skill_dir).as_posix()
            try:
                manifest[rel] = (p.stat().st_size, self._sha256_file(p))
            except FileNotFoundError:
                pass

        return manifest

    def _is_source_newer(self, src: Path, dst: Path) -> bool:
        """检测源 skill 是否比目标版本更新。"""
        return self._build_skill_manifest(src) != self._build_skill_manifest(dst)

    def _sync_skills_dir(
        self,
        source_dir: Path | None,
        target_dir: Path,
        label: str,
    ) -> list[str]:
        """同步 skills 从源目录到目标目录。"""
        if not source_dir or not source_dir.is_dir():
            logger.debug("No {} skills dir at {}, skip sync", label, source_dir)
            return []

        try:
            if source_dir.resolve() == target_dir.resolve():
                logger.debug("{} skills dir is the same as target, skip sync", label)
                return []
        except OSError:
            pass

        target_dir.mkdir(parents=True, exist_ok=True)

        skill_names = {
            d.name
            for d in source_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and (d / "SKILL.md").exists()
        }

        to_sync: list[tuple[str, str]] = []
        for name in skill_names:
            src = source_dir / name
            dst = target_dir / name

            if not dst.exists():
                to_sync.append((name, "missing"))
            elif not (dst / "SKILL.md").exists():
                to_sync.append((name, "no_skill_md"))
            elif self._is_source_newer(src, dst):
                to_sync.append((name, f"{label}_updated"))

        synced = []
        for name, reason in sorted(to_sync, key=lambda x: x[0]):
            src = source_dir / name
            dst = target_dir / name
            try:
                shutil.copytree(src, dst, dirs_exist_ok=True)
                synced.append(name)
                logger.info("Synced {} skill: {} -> {} ({})", label, name, dst, reason)
            except Exception as e:
                logger.warning("Failed to copy {} skill {}: {}", label, name, e)

        return synced

    def sync_builtin_skills(self) -> list[str]:
        """同步 builtin skills 到运行时目录。"""
        return self._sync_skills_dir(
            self._builtin_root,
            self._config.skills_dir,
            "builtin",
        )

    async def initialize(
        self,
        *,
        sync_builtin: bool = True,
    ) -> dict[str, Any]:
        """执行完整的 skill 系统初始化流程。

        初始化步骤：
        1. 同步 builtin skills 到运行时目录

        注册表同步已由 SkillStorage.from_config() 统一完成。

        Args:
            sync_builtin: 是否同步内置 skills（默认 True）

        Returns:
            包含各阶段结果的字典:
            {
                "builtin_synced": [...],
            }
        """
        result: dict[str, Any] = {
            "builtin_synced": [],
        }

        # 步骤 1: 同步 builtin skills
        if sync_builtin:
            logger.info("[SkillInitializer] Step 1: syncing builtin skills...")
            result["builtin_synced"] = self.sync_builtin_skills()
            if result["builtin_synced"]:
                logger.info(
                    "[SkillInitializer] Synced {} builtin skill(s): {}",
                    len(result["builtin_synced"]),
                    result["builtin_synced"],
                )
            else:
                logger.info("[SkillInitializer] All builtin skills are up to date")

        logger.info(
            "[SkillInitializer] Skill initialization completed: "
            "builtin={}",
            len(result["builtin_synced"]),
        )

        return result


