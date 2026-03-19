"""Skill 系统初始化器 - 内置技能同步与索引初始化。"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from middleware.config import g_config
from utils.logger import get_logger

if TYPE_CHECKING:
    from core.skill.schema import Skill

logger = get_logger(__name__)


class SkillInitializer:
    """Skill 系统初始化器。

    负责：
    1. 同步内置 skills 到工作目录
    2. 同步磁盘技能到数据库
    3. 同步 embedding 索引
    4. 清理孤儿记录

    Usage:
        initializer = SkillInitializer()
        await initializer.initialize(store, indexer, sync_builtin=True)
    """

    def __init__(self):
        self._builtin_root = self._resolve_builtin_root()
        self._workspace_skills_root = self._resolve_workspace_skills_root()

    def _resolve_builtin_root(self) -> Path:
        """解析 builtin skills 目录路径。

        通过 g_config 获取，支持源码环境和打包环境。
        """
        return g_config.get_builtin_skills_path()

    @staticmethod
    def _resolve_workspace_skills_root() -> Path | None:
        """解析项目 workspace/skills/ 目录路径。

        与 get_builtin_skills_path 相同策略：从 cwd 向上搜索项目根目录，
        查找 workspace/skills/ 子目录。打包环境下不存在此目录，返回 None。
        """
        marker_files = ["pyproject.toml", ".git", "bootstrap.py"]
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            if any((parent / marker).exists() for marker in marker_files):
                ws_skills = parent / "workspace" / "skills"
                if ws_skills.is_dir():
                    return ws_skills
        return None

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
            manifest[rel] = (p.stat().st_size, self._sha256_file(p))

        return manifest

    def _is_builtin_newer(self, src: Path, dst: Path) -> bool:
        """检测 builtin skill 是否比 workspace 版本更新。"""
        return self._build_skill_manifest(src) != self._build_skill_manifest(dst)

    def sync_builtin_skills(self) -> list[str]:
        """同步 builtin skills 到 workspace。

        覆盖条件：缺失、丢失 SKILL.md 或 builtin 已更新

        Returns:
            同步的 skill 名称列表
        """
        if not self._builtin_root.is_dir():
            logger.debug("No builtin skills dir at {}, skip sync", self._builtin_root)
            return []

        workspace_root = g_config.paths.skills_dir
        workspace_root.mkdir(parents=True, exist_ok=True)

        # 收集 builtin skills
        builtin_names = {
            d.name
            for d in self._builtin_root.iterdir()
            if d.is_dir() and not d.name.startswith(".") and (d / "SKILL.md").exists()
        }

        # 确定需要同步的 skills
        to_sync: list[tuple[str, str]] = []
        for name in builtin_names:
            src = self._builtin_root / name
            dst = workspace_root / name

            if not dst.exists():
                to_sync.append((name, "missing"))
            elif not (dst / "SKILL.md").exists():
                to_sync.append((name, "no_skill_md"))
            elif self._is_builtin_newer(src, dst):
                to_sync.append((name, "builtin_updated"))

        # 执行同步
        synced = []
        for name, reason in sorted(to_sync, key=lambda x: x[0]):
            src = self._builtin_root / name
            dst = workspace_root / name
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                synced.append(name)
                logger.info("Synced builtin skill: {} → {} ({})", name, dst, reason)
            except Exception as e:
                logger.warning("Failed to copy builtin skill {}: {}", name, e)

        return synced

    def sync_workspace_skills(self) -> list[str]:
        """同步项目 workspace/skills/ 到运行时 skills 目录。

        仅同步运行时目录中不存在的 skill（不覆盖已有的）。
        这样用户在运行时目录手动修改的 skill 不会被覆盖，
        而 workspace 中新增的 skill 会自动同步过来。

        Returns:
            同步的 skill 名称列表
        """
        if not self._workspace_skills_root or not self._workspace_skills_root.is_dir():
            logger.debug("No workspace skills dir found, skip sync")
            return []

        runtime_skills_dir = g_config.paths.skills_dir
        runtime_skills_dir.mkdir(parents=True, exist_ok=True)

        # 如果 workspace/skills/ 就是运行时目录本身，跳过
        try:
            if self._workspace_skills_root.resolve() == runtime_skills_dir.resolve():
                logger.debug("Workspace skills dir is the same as runtime, skip sync")
                return []
        except OSError:
            pass

        # 收集 workspace skills
        ws_skill_names = {
            d.name
            for d in self._workspace_skills_root.iterdir()
            if d.is_dir() and not d.name.startswith(".") and (d / "SKILL.md").exists()
        }

        # 只同步运行时目录中缺失的或 workspace 版本更新的
        to_sync: list[tuple[str, str]] = []
        for name in ws_skill_names:
            dst = runtime_skills_dir / name
            if not dst.exists():
                to_sync.append((name, "missing"))
            elif not (dst / "SKILL.md").exists():
                to_sync.append((name, "no_skill_md"))
            elif self._is_builtin_newer(self._workspace_skills_root / name, dst):
                to_sync.append((name, "workspace_updated"))

        # 执行同步
        synced = []
        for name, reason in sorted(to_sync, key=lambda x: x[0]):
            src = self._workspace_skills_root / name
            dst = runtime_skills_dir / name
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                synced.append(name)
                logger.info("Synced workspace skill: {} → {} ({})", name, dst, reason)
            except Exception as e:
                logger.warning("Failed to copy workspace skill {}: {}", name, e)

        return synced

    async def initialize(self, store, indexer, *, sync_builtin: bool = True) -> None:
        """执行完整的初始化流程。

        Args:
            store: SkillStore 实例
            indexer: SkillIndexer 实例
            sync_builtin: 是否同步内置 skills
        """
        if sync_builtin:
            synced = self.sync_builtin_skills()
            if synced:
                await store.refresh_from_disk()

        if indexer:
            await indexer.ensure_ready()

        await self._cleanup_orphans(store, indexer)
        await self._sync_to_db(store)
        await self._sync_embedding_index(indexer, store)

    async def _cleanup_orphans(self, store, indexer) -> list[str]:
        """清理磁盘已删除但 DB/向量库残留的 skill。"""
        return await store.cleanup_orphaned_skills(indexer=indexer)

    async def _sync_to_db(self, store) -> None:
        """同步所有本地 skills 到数据库。"""
        await store.sync_all_to_db()

    async def _sync_embedding_index(self, indexer, store) -> None:
        """同步 embedding 索引（批量 + 孤儿清理）。"""
        if not indexer:
            return
        if await indexer.ensure_ready():
            await indexer.index_batch(store.local_cache.values())
            indexer.cleanup_orphans(set(store.local_cache.keys()))


# 便捷函数（向后兼容）
async def init_skill_system(store, indexer, *, sync_builtin: bool = True) -> None:
    """统一执行技能系统初始化任务（向后兼容的便捷函数）。"""
    initializer = SkillInitializer()
    await initializer.initialize(store, indexer, sync_builtin=sync_builtin)


def sync_builtin_skills() -> list[str]:
    """同步 builtin skills 到 workspace（向后兼容的便捷函数）。"""
    initializer = SkillInitializer()
    return initializer.sync_builtin_skills()
