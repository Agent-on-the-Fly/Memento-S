"""Skill 系统初始化器 - 内置技能同步与索引初始化。"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from core.utils.text import to_snake_case
from utils.logger import get_logger
from .config import SkillConfig
from .gateway import SkillGateway

logger = get_logger(__name__)


class SkillInitializer:
    """Skill 系统初始化器。

    负责：
    1. 同步内置 skills 到工作目录
    2. 同步磁盘技能到数据库
    3. 同步 embedding 索引
    4. 清理孤儿记录

    Usage:
        initializer = SkillInitializer(config)
        await initializer.initialize(store, indexer, sync_builtin=True)
    """

    def __init__(self, config: "SkillConfig"):
        self._config = config
        self._builtin_root = config.builtin_skills_dir
        self._workspace_skills_root = self._resolve_workspace_skills_root()

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

    def _is_source_newer(self, src: Path, dst: Path) -> bool:
        """检测源 skill 是否比目标版本更新。"""
        return self._build_skill_manifest(src) != self._build_skill_manifest(dst)

    def _sync_skills_dir(
        self,
        source_dir: Path | None,
        target_dir: Path,
        label: str,
    ) -> list[str]:
        """同步 skills 从源目录到目标目录。

        Args:
            source_dir: 源目录（builtin 或 workspace）
            target_dir: 目标目录（运行时 skills 目录）
            label: 用于日志标签，如 "builtin" 或 "workspace"

        Returns:
            同步的 skill 名称列表
        """
        if not source_dir or not source_dir.is_dir():
            logger.debug("No {} skills dir at {}, skip sync", label, source_dir)
            return []

        # 如果源目录就是目标目录本身，跳过
        try:
            if source_dir.resolve() == target_dir.resolve():
                logger.debug("{} skills dir is the same as target, skip sync", label)
                return []
        except OSError:
            pass

        target_dir.mkdir(parents=True, exist_ok=True)

        # 收集源目录中的 skills
        skill_names = {
            d.name
            for d in source_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and (d / "SKILL.md").exists()
        }

        # 确定需要同步的 skills
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

        # 执行同步
        synced = []
        for name, reason in sorted(to_sync, key=lambda x: x[0]):
            src = source_dir / name
            dst = target_dir / name
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                synced.append(name)
                logger.info("Synced {} skill: {} → {} ({})", label, name, dst, reason)
            except Exception as e:
                logger.warning("Failed to copy {} skill {}: {}", label, name, e)

        return synced

    def sync_builtin_skills(self) -> list[str]:
        """同步 builtin skills 到运行时目录。

        覆盖条件：缺失、丢失 SKILL.md 或 builtin 已更新

        Returns:
            同步的 skill 名称列表
        """
        return self._sync_skills_dir(
            self._builtin_root,
            self._config.skills_dir,
            "builtin",
        )

    def sync_workspace_skills(self) -> list[str]:
        """同步项目 workspace/skills/ 到运行时 skills 目录。

        仅同步运行时目录中不存在的 skill（不覆盖已有的）。
        这样用户在运行时目录手动修改的 skill 不会被覆盖，
        而 workspace 中新增的 skill 会自动同步过来。

        Returns:
            同步的 skill 名称列表
        """
        return self._sync_skills_dir(
            self._workspace_skills_root,
            self._config.skills_dir,
            "workspace",
        )

    async def initialize(
        self,
        store,
        *,
        sync_builtin: bool = True,
        sync_workspace: bool = True,
    ) -> dict[str, Any]:
        """执行完整的 skill 系统初始化流程。

        初始化步骤：
        1. 同步 builtin skills 到运行时目录
        2. 同步 workspace skills 到运行时目录
        3. 刷新磁盘到内存/缓存（捕获所有同步的 + 用户手动添加的）
        4. 同步所有本地 skills 到数据库
        5. 清理孤儿记录（磁盘已删除但 DB/向量库残留）

        Args:
            store: SkillStore 实例
            sync_builtin: 是否同步内置 skills（默认 True）
            sync_workspace: 是否同步 workspace skills（默认 True）

        Returns:
            包含各阶段结果的字典:
            {
                "builtin_synced": [...],      # 同步的 builtin skill 名称列表
                "workspace_synced": [...],    # 同步的 workspace skill 名称列表
                "refreshed": int,             # refresh_from_disk 新增的 skill 数量
                "db_synced": int,             # sync_from_disk 同步的 skill 数量
                "orphans_cleaned": [...],     # 清理的孤儿 skill 名称列表
            }
        """
        result = {
            "builtin_synced": [],
            "workspace_synced": [],
            "refreshed": 0,
            "db_synced": 0,
            "orphans_cleaned": [],
            "vector_synced": 0,
            "alignment": {},
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

        # 步骤 2: 同步 workspace skills
        if sync_workspace:
            logger.info("[SkillInitializer] Step 2: syncing workspace skills...")
            result["workspace_synced"] = self.sync_workspace_skills()
            if result["workspace_synced"]:
                logger.info(
                    "[SkillInitializer] Synced {} workspace skill(s): {}",
                    len(result["workspace_synced"]),
                    result["workspace_synced"],
                )
            else:
                logger.info("[SkillInitializer] All workspace skills are up to date")

        # 步骤 3: 对齐检查（三方集合：file / db / vector）
        logger.info("[SkillInitializer] Step 3: checking alignment (file/db/vector)...")
        file_names = {to_snake_case(n) for n in await store._file.list_names()}
        db_names = {to_snake_case(n) for n in await store._db.list_names()}

        raw_vector_names = await store._vector.list_names()
        vector_name_map = {to_snake_case(n): n for n in raw_vector_names}
        vector_names = set(vector_name_map.keys())

        missing_in_db = file_names - db_names
        missing_in_vector = file_names - vector_names
        orphan_in_db = db_names - file_names
        orphan_in_vector = vector_names - file_names

        result["alignment"] = {
            "file_count": len(file_names),
            "db_count": len(db_names),
            "vector_count": len(vector_names),
            "missing_in_db": sorted(missing_in_db),
            "missing_in_vector": sorted(missing_in_vector),
            "orphan_in_db": sorted(orphan_in_db),
            "orphan_in_vector": sorted(orphan_in_vector),
        }

        logger.info(
            "[SkillInitializer] Alignment diff: missing_in_db={}, missing_in_vector={}, orphan_in_db={}, orphan_in_vector={}",
            len(missing_in_db),
            len(missing_in_vector),
            len(orphan_in_db),
            len(orphan_in_vector),
        )

        # 步骤 4: 以文件为准，清理 DB/Vector 中的孤儿记录
        logger.info(
            "[SkillInitializer] Step 4: removing orphaned records from DB/Vector..."
        )
        for name in sorted(orphan_in_db):
            await store._db.delete(name)
            result["orphans_cleaned"].append(name)

        for name in sorted(orphan_in_vector):
            # vector 里可能是 kebab/snake 历史键，按原始键删除
            raw_name = vector_name_map.get(name, name)
            await store._vector.delete(raw_name)
            if name not in result["orphans_cleaned"]:
                result["orphans_cleaned"].append(name)

        if result["orphans_cleaned"]:
            logger.info(
                "[SkillInitializer] Cleaned up {} orphaned skill(s): {}",
                len(result["orphans_cleaned"]),
                result["orphans_cleaned"],
            )
        else:
            logger.info("[SkillInitializer] No orphaned skills found")

        # 步骤 5: 将文件中缺失到 DB 的 skill 补齐
        logger.info(
            "[SkillInitializer] Step 5: repairing missing DB records from file system..."
        )
        db_synced = 0
        for name in sorted(missing_in_db):
            skill = await store._file.load(name)
            if not skill:
                logger.warning(
                    "[SkillInitializer] Failed to load skill '{}' from file", name
                )
                continue
            await store._db.save(skill.name, skill)
            db_synced += 1

        result["db_synced"] = db_synced
        logger.info("[SkillInitializer] Repaired {} missing DB record(s)", db_synced)

        # 步骤 6: 将文件中缺失到 Vector 的 skill embedding 补齐
        logger.info("[SkillInitializer] Step 6: repairing missing vector embeddings...")
        vector_synced = 0
        await self._close_db_transactions(store)  # 确保 DB 事务已提交

        if not store._embedding_generator or not store._embedding_generator.is_ready:
            logger.warning(
                "[SkillInitializer] Embedding generator not available, skip vector repair"
            )
        else:
            for name in sorted(missing_in_vector):
                skill = await store._file.load(name)
                if not skill:
                    logger.warning(
                        "[SkillInitializer] Failed to load skill '{}' from file", name
                    )
                    continue
                try:
                    vector = await store._embedding_generator.generate_for_skill(skill)
                    if vector:
                        # 统一使用 snake_case 作为内部传递和向量存储键
                        await store._vector.save(to_snake_case(name), vector)
                        vector_synced += 1
                except Exception as e:
                    logger.warning(
                        "Failed to generate embedding for skill '{}': {}", name, e
                    )

        result["vector_synced"] = vector_synced
        result["refreshed"] = result["db_synced"] + result["vector_synced"]
        logger.info(
            "[SkillInitializer] Repaired {} missing vector embedding(s)", vector_synced
        )

        # 最终文件中的 skill 数量（file_names 在步骤3获取，步骤5-6已修复，是最新的）
        final_skill_count = len(file_names) + result["db_synced"]
        logger.info(
            "[SkillInitializer] Skill initialization completed: {} local skills",
            final_skill_count,
        )

        return result

    async def _close_db_transactions(self, store) -> None:
        """关闭所有待处理的 DB 事务，确保事务已提交

        在生成 embedding 前调用，避免事务冲突。
        """
        try:
            # 关闭并重新打开 DB 连接，确保所有事务已提交
            await store._db.close()
            await store._db.init()
            logger.debug("DB transactions committed and connection refreshed")
        except Exception as e:
            logger.warning("Failed to refresh DB connection: {}", e)

    async def _generate_embeddings_safely(self, store) -> int:
        """安全地生成 skills 的 embedding

        在 DB 事务已提交后调用，避免事务冲突。

        Args:
            store: SkillStore 实例

        Returns:
            生成的 embedding 数量
        """
        # 检查是否有 embedding generator
        if not store._embedding_generator or not store._embedding_generator.is_ready:
            logger.debug("Embedding generator not available, skipping")
            return 0

        count = 0
        skills = await store.list_all_skills()

        for skill in skills.values():
            try:
                # 检查是否已有 embedding
                skill_id = (
                    Path(skill.source_dir).name if skill.source_dir else skill.name
                )
                existing = await store._vector.load(skill_id)
                if existing:
                    continue  # 已有 embedding，跳过

                # 生成 embedding
                vector = await store._embedding_generator.generate_for_skill(skill)
                if vector:
                    await store._vector.save(skill_id, vector)
                    count += 1
                    logger.debug("Generated embedding for skill: {}", skill.name)
            except Exception as e:
                logger.warning(
                    "Failed to generate embedding for skill '{}': {}", skill.name, e
                )

        return count


# ==============================================================================
# 模块级引导 API（替代 core/skill/bootstrap.py）
# ==============================================================================

logger_bootstrap = get_logger(__name__ + ".bootstrap")


async def init_skill_system(config: SkillConfig | None = None) -> SkillGateway:
    """初始化技能系统并返回 Gateway 实例。

    调用者需要自行保存返回的 Gateway 实例，不要依赖全局状态。
    注意：此函数执行完整的初始化流程，包括 skills 同步和 embedding 生成。

    Args:
        config: 配置对象，为 None 时自动从 g_config 读取

    Returns:
        SkillGateway 实例

    Example:
        # 生产环境
        gateway = await init_skill_system()

        # 测试环境
        config = SkillConfig(...)
        gateway = await init_skill_system(config)
    """
    # 1. 创建/获取配置
    if config is None:
        config = SkillConfig.from_global_config()

    # 2. 创建 Gateway（内部自动创建所有依赖）
    gateway = await SkillGateway.from_config(config)

    # 3. 执行完整的初始化流程
    initializer = SkillInitializer(config)
    init_result = await initializer.initialize(
        gateway.skill_store,
        sync_builtin=True,
        sync_workspace=True,
    )

    logger_bootstrap.info(
        "Skill system initialized: builtin={}, workspace={}, refreshed={}, db_synced={}",
        len(init_result.get("builtin_synced", [])),
        len(init_result.get("workspace_synced", [])),
        init_result.get("refreshed", 0),
        init_result.get("db_synced", 0),
    )

    return gateway
