# SPDX-License-Identifier: Apache-2.0
"""skill.json 注册表 — skill 系统的本地注册表封装。

封装 SkillConfigManager，对外暴露 skill 系统需要的接口。
直接读写 ~/memento_s/skill.json（由 SkillConfigManager 管理）。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from middleware.config.skill_config_manager import skill_config_manager
from utils.strings import to_kebab_case, to_snake_case
from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level sync guard: prevents duplicate sync_from_disk in multi-init scenarios
# (e.g., bootstrap sync + GUI init both calling init_skill_system)
_sync_initialized: bool = False
_sync_lock = __import__("threading").Lock()


class SkillRegistry:
    """skill.json 的 skill 系统封装层。

    负责维护本地 skill 的注册表（哪些 skill 在本地存在），
    不负责实际的文件读写（由 FileStorage 处理）。

    使用方式：
        registry = SkillRegistry()
        registry.sync_from_disk(skills_dir, builtin_dir)
    """

    def __init__(self) -> None:
        self._manager = skill_config_manager

    # ── 基础读写 ─────────────────────────────────────────────────────────────

    def load(self) -> dict[str, Any]:
        """加载注册表数据"""
        return self._manager.load()

    def save(self, data: dict[str, Any]) -> None:
        """保存注册表数据"""
        self._manager.save(data)

    # ── 注册表操作 ───────────────────────────────────────────────────────────

    def register(self, name: str, meta: dict[str, Any]) -> None:
        """注册一个 skill 条目（同步写入磁盘）"""
        self._manager.register_skill(name, meta)

    def unregister(self, name: str) -> None:
        """移除一个 skill 条目（同步写入磁盘）"""
        self._manager.unregister_skill(name)

    def list_all(self) -> dict[str, dict[str, Any]]:
        """列出所有 skill 条目"""
        return self._manager.list_skills()

    def get(self, name: str) -> dict[str, Any] | None:
        """获取单个 skill 条目"""
        return self._manager.get_skill(name)

    # ── 磁盘同步 ─────────────────────────────────────────────────────────────

    def sync_from_disk(
        self,
        skills_dir: Path,
        builtin_dir: Path | None = None,
    ) -> dict[str, list[str]]:
        """扫描 skills_dir，将结果与注册表对齐。

        对比逻辑：
        - 目录存在 + 注册表没有 → 添加（source="local"）
        - 目录存在 + 注册表有 → 保留（不覆盖用户编辑的 meta）
        - 注册表有 + 目录不存在 → 移除

        Args:
            skills_dir: 用户 skills 目录（~memento_s/skills/）
            builtin_dir: 内置 skills 目录

        Returns:
            {"added": [...], "removed": [...], "unchanged": [...], "errors": [...]}
        """
        global _sync_initialized

        with _sync_lock:
            if _sync_initialized:
                logger.debug("sync_from_disk: already initialized, skipping duplicate sync")
                self._manager.load()
                existing = list(self._manager.list_skills().keys())
                return {
                    "added": [],
                    "removed": [],
                    "unchanged": existing,
                    "errors": [],
                }
            _sync_initialized = True

        self._manager.load()

        result: dict[str, list[str]] = {
            "added": [],
            "removed": [],
            "unchanged": [],
            "errors": [],
        }

        # 收集磁盘上实际存在的 skill 目录（kebab-case）
        # 注意：只扫描 skills_dir（用户目录），location 统一指向用户目录
        # builtin skill 在初始化时已同步到 skills_dir
        disk_skills: dict[str, Path] = {}
        if skills_dir is not None and skills_dir.exists():
            for item in skills_dir.iterdir():
                if not item.is_dir():
                    continue
                skill_md = item / "SKILL.md"
                if not skill_md.exists():
                    continue
                name = to_kebab_case(item.name)
                disk_skills[name] = item

        # 获取当前注册表中的 skill
        registry_skills = self._manager.list_skills()

        # 遍历注册表：移除已删除的
        for name in list(registry_skills.keys()):
            if name not in disk_skills:
                self._manager.unregister_skill(name)
                result["removed"].append(name)
                logger.debug("sync_from_disk: removed '{}' (no longer on disk)", name)

        # 遍历磁盘：添加新 skill
        now = datetime.utcnow().isoformat() + "Z"
        for name, skill_path in disk_skills.items():
            if name not in registry_skills:
                # location 统一指向 skills_dir（用户目录），与代码目录解耦
                location = str(skill_path.resolve())
                # source 判断：检查同名的 skill 是否存在于 builtin_dir（源码目录）
                # 存在于 builtin_dir 说明该 skill 是从 builtin 同步过来的
                source = "builtin"
                if builtin_dir and builtin_dir.exists():
                    if not (builtin_dir / name).exists():
                        source = "local"
                else:
                    source = "local"
                self._manager.register_skill(name, {
                    "location": location,
                    "source": source,
                    "version": 1,
                    "installed_at": now,
                    "status": "active",
                    "tags": [],
                })
                result["added"].append(name)
                logger.debug("sync_from_disk: added '{}' (location={}, source={})", name, location, source)

        # 没有变化
        existing = set(disk_skills.keys()) & set(registry_skills.keys())
        result["unchanged"] = list(existing)

        self._manager.update_sync_time(result.get("errors", []))

        logger.info(
            "sync_from_disk: added={}, removed={}, unchanged={}, errors={}",
            len(result["added"]),
            len(result["removed"]),
            len(result["unchanged"]),
            len(result["errors"]),
        )
        return result

    # ── 工具方法 ─────────────────────────────────────────────────────────────


# ── 模块级实例 ────────────────────────────────────────────────────────────────

registry = SkillRegistry()

__all__ = ["SkillRegistry", "registry"]
