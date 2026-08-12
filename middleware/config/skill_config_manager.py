# SPDX-License-Identifier: Apache-2.0
"""Skill 配置管理器。

管理 ~/memento_s/skill.json 的读写和验证，
遵循与 ConfigManager 一致的模式：
- 模板：skill_config_tpl.json（资源文件）
- Schema：skill_config_schema.json（用于 jsonschema 验证）
- 用户文件：~/memento_s/skill.json（读写）
- 初始化时如果不存在则从模板复制
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import validate

from .schemas.skill_config_schemas import SkillEntry, SkillRegistryConfig
from utils.path_manager import PathManager

logger = logging.getLogger(__name__)


class SkillConfigManager:
    """skill.json 管理器"""

    _PACKAGE = "middleware.config"
    _TEMPLATE = "skill_config_tpl.json"
    _SCHEMA = "skill_config_schema.json"
    _USER_FILE = "skill.json"

    def __init__(self) -> None:
        self._user_path = PathManager.get_project_root_dir() / self._USER_FILE
        self._data: dict[str, Any] = {}
        self._loaded = False
        self._model: SkillRegistryConfig | None = None  # cached Pydantic model

    @property
    def user_path(self) -> Path:
        return self._user_path

    # ── 读写 ────────────────────────────────────────────────────────────────

    def load(self) -> dict[str, Any]:
        """加载用户配置（同步）。

        初始化时自动从模板创建文件（如果不存在）。
        """
        if not self._user_path.exists():
            self._bootstrap_from_template()

        self._data = self._read_json(self._user_path)
        self._validate()
        self._loaded = True
        self._model = None  # invalidate cached Pydantic model
        return self._data

    def save(self, data: dict[str, Any] | None = None) -> None:
        """保存配置到磁盘（同步）。

        Args:
            data: 要保存的数据，如果为 None 则保存当前内存数据
        """
        if data is not None:
            self._data = data
        self._validate()
        self._write_json(self._user_path, self._data)
        self._model = None  # invalidate cached Pydantic model

    # ── skills CRUD ─────────────────────────────────────────────────────────

    def register_skill(self, name: str, meta: dict[str, Any]) -> None:
        """注册或更新一个 skill 条目"""
        self._ensure_loaded()
        self._data["skills"][name] = meta
        self.save()

    def unregister_skill(self, name: str) -> None:
        """移除一个 skill 条目"""
        self._ensure_loaded()
        self._data["skills"].pop(name, None)
        self.save()

    def list_skills(self) -> dict[str, dict[str, Any]]:
        """列出所有 skill 条目（dict 形式）"""
        self._ensure_loaded()
        return dict(self._data["skills"])

    def list_skills_model(self) -> dict[str, SkillEntry]:
        """列出所有 skill 条目（Pydantic SkillEntry 形式）"""
        self._ensure_loaded()
        return {k: SkillEntry.model_validate(v) for k, v in self._data["skills"].items()}

    def get_skill_model(self, name: str) -> SkillEntry | None:
        """获取单个 skill 条目（Pydantic SkillEntry 形式）"""
        raw = self.get_skill(name)
        if raw is None:
            return None
        return SkillEntry.model_validate(raw)

    def register_skill_model(self, name: str, entry: SkillEntry) -> None:
        """注册或更新一个 skill 条目（接收 SkillEntry Pydantic 模型）"""
        self._ensure_loaded()
        self._data["skills"][name] = entry.model_dump(mode="json")
        self.save()

    def get_registry_model(self) -> SkillRegistryConfig:
        """返回 Pydantic SkillRegistryConfig 模型（已加载则缓存）"""
        if not hasattr(self, "_model") or self._model is None:
            self._ensure_loaded()
            self._model = SkillRegistryConfig.model_validate(self._data)
        return self._model

    def get_skill(self, name: str) -> dict[str, Any] | None:
        """获取单个 skill 条目"""
        self._ensure_loaded()
        return self._data["skills"].get(name)

    def update_sync_time(self, errors: list[str] | None = None) -> None:
        """更新 last_sync 时间戳和 sync_errors"""
        self._ensure_loaded()
        self._data.setdefault("index", {})
        self._data["index"]["last_sync"] = datetime.utcnow().isoformat() + "Z"
        self._data["index"]["sync_errors"] = errors if errors is not None else []
        self.save()

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _bootstrap_from_template(self) -> None:
        """初始化：如果用户文件不存在，从模板复制。
        
        兼容迁移：如果旧文件 skill.config.json 存在但 skill.json 不存在，
        则将其内容迁移到新文件。
        """
        old_path = self._user_path.with_name("skill.config.json")
        if old_path.exists() and not self._user_path.exists():
            self._user_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_path, self._user_path)
            logger.info("skill.json migrated from skill.config.json: %s", self._user_path)
            return

        template = self._get_template_path()
        self._user_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template, self._user_path)
        logger.info("skill.json bootstrapped from template: %s", self._user_path)

    def _get_template_path(self) -> Path:
        """获取模板路径（支持源码和打包两种环境）"""
        return self._load_resource_path(self._TEMPLATE)

    def _get_schema_path(self) -> Path:
        """获取 Schema 路径（支持源码和打包两种环境）"""
        return self._load_resource_path(self._SCHEMA)

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_resource_path(filename: str) -> Path:
        """解析资源文件路径（带缓存）"""
        from importlib import resources
        import sys

        # Strategy 1: 源码环境 - 从当前文件向上查找
        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            candidate = parent / filename
            if candidate.exists():
                return candidate

        # Strategy 2: 打包应用 - 使用 importlib.resources
        try:
            ref = resources.files(SkillConfigManager._PACKAGE) / filename
            if ref.is_file():
                return Path(str(ref))
        except (ImportError, TypeError, AttributeError):
            pass

        # Strategy 3: PyInstaller 打包
        if getattr(sys, "frozen", False):
            if hasattr(sys, "_MEIPASS"):
                candidate = Path(sys._MEIPASS) / "middleware" / "config" / filename
                if candidate.exists():
                    return candidate

        raise FileNotFoundError(
            f"Cannot find resource '{filename}' for SkillConfigManager"
        )

    def _validate(self) -> None:
        """用 jsonschema 验证数据"""
        schema_path = self._get_schema_path()
        schema = self._read_json(schema_path)
        try:
            validate(instance=self._data, schema=schema)
        except jsonschema.ValidationError as e:
            logger.error(f"skill.json validation failed: {e.message}")
            raise

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ── 模块级单例 ────────────────────────────────────────────────────────────────

skill_config_manager = SkillConfigManager()

__all__ = ["SkillConfigManager", "skill_config_manager"]
