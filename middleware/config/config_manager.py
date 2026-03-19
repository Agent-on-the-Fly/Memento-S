"""
Memento-S 配置管理模块
仅支持文件配置：
- 模板配置：middleware/config/system_config.json
- 用户配置：~/memento_s/config.json
- 通过 JSON Schema 校验用户配置
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import validate

from middleware.config.config_models import GlobalConfig
from middleware.config.migrations import merge_configs, merge_template_defaults
from utils.path_manager import PathManager

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器（文件模式）"""

    _CONFIG_PACKAGE = "middleware.config"
    _SYSTEM_CONFIG = "system_config.json"
    _USER_TEMPLATE = "user_config_tlp.json"
    _SCHEMA_FILE = "user_config_schema.json"

    def __init__(self, config_path: str | None = None):
        self.user_config_path = (
            Path(config_path).expanduser().resolve()
            if config_path
            else PathManager.get_config_file()
        )
        self._config_data: dict[str, Any] = {}
        self._config: GlobalConfig | None = None

    def __getattr__(self, name: str) -> Any:
        """代理对 _config 属性的访问，允许直接通过 g_config.xxx 访问配置数据。"""
        if self._config is None:
            raise RuntimeError(f"配置尚未加载，无法访问 '{name}'。请先调用 load()")
        return getattr(self._config, name)

    @property
    def user_config_dir(self) -> Path:
        """返回用户配置目录。"""
        return self.user_config_path.parent

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_resource(filename: str) -> dict[str, Any]:
        """从包内资源加载 JSON 文件（带缓存）。"""
        text = (
            resources.files(ConfigManager._CONFIG_PACKAGE)
            .joinpath(filename)
            .read_text(encoding="utf-8")
        )
        return json.loads(text)

    def load_schema(self) -> dict[str, Any]:
        """从包内资源加载 JSON Schema。"""
        return self._load_resource(self._SCHEMA_FILE)

    def load_system_config(self) -> dict[str, Any]:
        """从包内资源加载系统配置。"""
        return self._load_resource(self._SYSTEM_CONFIG)

    def load_user_template(self) -> dict[str, Any]:
        """从包内资源加载用户配置模板。"""
        return self._load_resource(self._USER_TEMPLATE)

    def user_config_exists(self) -> bool:
        """用户配置文件是否存在。"""
        return self.user_config_path.exists()

    def ensure_user_config_dir(self) -> Path:
        """确保用户配置目录存在。"""
        self.user_config_dir.mkdir(parents=True, exist_ok=True)
        return self.user_config_dir

    def ensure_user_config_file(self) -> Path:
        """确保用户配置文件存在，不存在则从模板复制。"""
        self.ensure_user_config_dir()
        if not self.user_config_path.exists():
            template = self.load_user_template()
            self._write_json(self.user_config_path, template)
        return self.user_config_path

    def _load_merged_config(self) -> dict[str, Any]:
        """加载并合并系统配置与用户配置。"""
        system = self.load_system_config()
        user = self._load_user_config()
        return merge_configs(system, user)

    def _load_user_config(self) -> dict[str, Any]:
        """加载用户配置文件。"""
        with open(self.user_config_path, encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        """将数据写入 JSON 文件。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_raw_config(self) -> dict[str, Any]:
        """读取用户完整的 config.json 文件内容。

        Returns:
            用户配置的原始字典数据

        Raises:
            FileNotFoundError: 如果用户配置文件不存在
        """
        if not self.user_config_path.exists():
            raise FileNotFoundError(f"用户配置文件不存在: {self.user_config_path}")
        with open(self.user_config_path, encoding="utf-8") as f:
            return json.load(f)

    def save_raw_config(self, config_data: dict[str, Any]) -> str | None:
        """保存 JSON 配置到用户 config.json。

        会进行 Schema 验证。

        Args:
            config_data: 配置字典

        Returns:
            None 表示保存成功，字符串表示错误信息
        """
        try:
            # JSON Schema 验证
            schema = self.load_schema()
            validate(instance=config_data, schema=schema)
        except jsonschema.ValidationError as e:
            return f"Schema 验证失败: {e.message} (路径: {list(e.path)})"
        except Exception as e:
            return f"验证过程出错: {e}"

        try:
            # 确保目录存在
            self.ensure_user_config_dir()
            # 写入文件
            self._write_json(self.user_config_path, config_data)
            logger.info("用户配置已保存到: %s", self.user_config_path)
            return None
        except Exception as e:
            return f"保存配置文件失败: {e}"

    def _normalize_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """补全默认字段和路径配置。"""
        # 版本号
        config.setdefault("version", "1.0.0")

        # OTA 配置
        if not isinstance(config.get("ota"), dict):
            config["ota"] = {}
        config["ota"].setdefault("url", None)

        # LLM active_profile 修复
        llm = config.get("llm")
        if isinstance(llm, dict):
            profiles = llm.get("profiles")
            active = llm.get("active_profile")
            if isinstance(profiles, dict) and profiles:
                if not active or active not in profiles:
                    preferred = ["default", "kimi", "openai", "mini-sglang"]
                    chosen = next((p for p in preferred if p in profiles), None)
                    llm["active_profile"] = chosen or next(iter(profiles.keys()))

        # Skills execution 字段补全
        skills = config.get("skills")
        if isinstance(skills, dict):
            execution = skills.get("execution")
            if isinstance(execution, dict):
                execution.setdefault("max_reflection_retries", 3)
                execution.setdefault("bash_timeout_sec", 300)
                execution.setdefault("pip_install_timeout_sec", 180)
                execution.setdefault("cli_install_timeout_sec", 300)

        # 路径补全（从 PathManager 获取实际路径）
        paths = config.setdefault("paths", {})
        paths["workspace_dir"] = str(PathManager.get_workspace_dir())
        paths["skills_dir"] = str(PathManager.get_skills_dir())
        paths["db_dir"] = str(PathManager.get_db_dir())
        paths["logs_dir"] = str(PathManager.get_logs_dir())
        paths["venv_dir"] = str(PathManager.get_venv_dir())
        paths["context_dir"] = str(PathManager.get_context_dir())

        return config

    def _validate(self, config: dict[str, Any]) -> GlobalConfig:
        """验证配置，返回强类型对象。"""
        # JSON Schema 校验
        try:
            validate(instance=config, schema=self.load_schema())
        except jsonschema.ValidationError as e:
            logger.warning("Schema 校验失败: %s", e)

        # Pydantic 校验
        try:
            return GlobalConfig.model_validate(config)
        except Exception as e:
            raise ValueError(f"配置 Pydantic 校验失败: {e}") from e

    def load(self) -> GlobalConfig:
        """加载并验证配置，返回强类型配置对象。

        Note: 配置迁移由 bootstrap 统一处理，此方法只负责加载。
        """
        self.ensure_user_config_file()

        # 加载并补全配置
        config = self._load_merged_config()
        config = self._normalize_config(config)

        # 配置写回由 bootstrap 统一处理，这里只加载内存配置

        # 验证并缓存
        typed = self._validate(config)
        self._config_data = typed.to_json_dict()
        self._config = typed
        return typed

    def _save_user_config(self, merged: dict[str, Any]) -> None:
        """保存用户配置（只写入用户配置文件）。"""
        logger.info("[Config] Starting save to: %s", self.user_config_path)
        logger.info("[Config] Input merged data keys: %s", list(merged.keys()))
        if "llm" in merged:
            logger.info(
                "[Config] LLM profiles: %s",
                list(merged["llm"].get("profiles", {}).keys()),
            )
        try:
            template = self.load_user_template()
            user = merge_template_defaults(template, merged)
            logger.info("[Config] After template merge keys: %s", list(user.keys()))

            # 对于完全由用户控制的字段（如 llm.profiles），
            # 恢复为用户的实际值，避免被模板中的默认值覆盖
            if "llm" in merged and "profiles" in merged["llm"]:
                if "llm" not in user:
                    user["llm"] = {}
                user["llm"]["profiles"] = merged["llm"]["profiles"]
                logger.info(
                    "[Config] Restored user profiles after merge: %s",
                    list(user["llm"]["profiles"].keys()),
                )

            if "llm" in user:
                logger.info(
                    "[Config] LLM profiles after merge: %s",
                    list(user["llm"].get("profiles", {}).keys()),
                )

            self._write_json(self.user_config_path, user)
            logger.info("[Config] Successfully saved to: %s", self.user_config_path)

            # 验证文件是否写入
            if self.user_config_path.exists():
                file_size = self.user_config_path.stat().st_size
                logger.info("[Config] File size after save: %s bytes", file_size)
        except Exception as e:
            logger.warning("Failed to save user config: %s", e)

    def _set_by_path(self, config: dict[str, Any], key_path: str, value: Any) -> None:
        """按路径设置字典值。"""
        keys = key_path.split(".")
        current: dict[str, Any] = config
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def get_data_dir(self) -> Path:
        """Return data directory path from PathManager."""
        return PathManager.get_data_dir()

    def get_env(self) -> dict[str, Any]:
        """Return env dict from loaded config."""
        if not self._config:
            raise RuntimeError("Config not loaded, call load() first")
        return self._config.env or {}

    def get_workspace_dir(self) -> Path:
        """Return workspace directory path."""
        if not self._config or not self._config.paths.workspace_dir:
            raise ValueError("Config not loaded or workspace_dir is not set")
        return self._config.paths.workspace_dir

    def get_skills_path(self) -> Path:
        """Return skills directory path."""
        if not self._config or not self._config.paths.skills_dir:
            raise ValueError("Config not loaded or skills_dir is not set")
        return self._config.paths.skills_dir

    def get_db_path(self) -> Path:
        """Return database file path (memento_s.db)."""
        if not self._config or not self._config.paths.db_dir:
            raise ValueError("Config not loaded or db_dir is not set")
        return self._config.paths.db_dir / "memento_s.db"

    def get_db_url(self) -> str:
        """Return default SQLAlchemy async SQLite URL."""
        return f"sqlite+aiosqlite:///{self.get_db_path()}"

    def get_skill_path(self, skill_name: str) -> Path:
        """Return path for a specific skill."""
        if not self._config or not self._config.paths.skills_dir:
            raise ValueError("Config not loaded or skills_dir is not set")
        return self._config.paths.skills_dir / skill_name

    def get_skill_scripts_path(self, skill_source_dir: Path) -> Path:
        """Return the scripts directory for a given skill."""
        return skill_source_dir / "scripts"

    def get_log_path(self, log_name: str) -> Path:
        """Return path for a specific log file."""
        if not self._config or not self._config.paths.logs_dir:
            raise ValueError("Config not loaded or logs_dir is not set")
        return self._config.paths.logs_dir / log_name

    def get_builtin_skills_path(self) -> Path:
        """Return builtin skills directory path.

        Handles both source and packaged environments.
        In source: project_root/builtin/skills
        In package: uses importlib.resources
        """
        # Strategy 1: Source environment - find project root by looking for marker files
        marker_files = ["pyproject.toml", ".git", "bootstrap.py"]
        current_dir = Path.cwd()

        # Search upward from current working directory
        for parent in [current_dir] + list(current_dir.parents):
            if any((parent / marker).exists() for marker in marker_files):
                # Found project root, check for builtin/skills
                builtin_path = parent / "builtin" / "skills"
                if builtin_path.exists():
                    return builtin_path

        # Strategy 2: Flet packaged app - use importlib.resources
        # In packaged apps, files are embedded and __file__ paths don't work
        try:
            # Try to access as a package resource
            builtin_ref = resources.files("memento_s") / "builtin" / "skills"
            # resources.files returns a Traversable, need to check if it exists
            if builtin_ref.is_dir():
                # Convert to path if possible
                return Path(str(builtin_ref))
        except (ImportError, TypeError, AttributeError):
            pass

        # Strategy 3: Try from executable location (for PyInstaller/flet bundles)
        try:
            import sys

            if getattr(sys, "frozen", False) and sys.executable:
                # Running in a bundle (PyInstaller)
                exe_path = str(sys.executable)
                if exe_path:
                    bundle_dir = Path(exe_path).parent
                    builtin_path = bundle_dir / "builtin" / "skills"
                    if builtin_path.exists():
                        return builtin_path
        except (AttributeError, TypeError):
            pass

        # If neither found, raise an error
        raise RuntimeError(
            "Cannot find builtin skills directory. "
            "Source: expected at <project_root>/builtin/skills (looked from CWD). "
            "Package: expected memento_s.builtin.skills resource."
        )

    def get_session_sandbox_dir(self, skill_name: str, session_id: str) -> Path:
        """Get a working directory for skill execution, scoped to a session."""
        if not self._config or not self._config.paths.workspace_dir:
            raise ValueError("Config not loaded or workspace_dir is not set")

        resolved_id = session_id or "default"
        ts = time.strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:4]
        sandbox_dir = (
            self._config.paths.workspace_dir
            / "sessions"
            / resolved_id
            / "sandbox"
            / f"{skill_name}_{ts}_{short_id}"
        )
        sandbox_dir = sandbox_dir.resolve()
        workspace = self._config.paths.workspace_dir.resolve()
        if not sandbox_dir.is_relative_to(workspace):
            raise ValueError(
                f"Sandbox dir '{sandbox_dir}' must be under workspace '{workspace}'"
            )
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        return sandbox_dir

    def get_session_output_dir(self, skill_name: str, session_id: str = "") -> Path:
        """Get output directory for skill execution artifacts."""
        if not self._config or not self._config.paths.workspace_dir:
            raise ValueError("Config not loaded or workspace_dir is not set")

        workspace = self._config.paths.workspace_dir
        ts = time.strftime("%Y%m%d_%H%M%S")

        if session_id:
            output_dir = (
                workspace / "sessions" / session_id / "outputs" / f"{skill_name}_{ts}"
            )
        else:
            short_id = uuid.uuid4().hex[:6]
            output_dir = workspace / "outputs" / f"{skill_name}_{ts}_{short_id}"

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def set(self, key_path: str, value: Any, save: bool = True) -> None:
        """设置配置值并可选保存。"""
        logger.info("[Config] set() called: key_path=%s, save=%s", key_path, save)
        if self._config is None:
            logger.info("[Config] Config not loaded, loading...")
            self.load()
        logger.info(
            "[Config] Before _set_by_path, profiles: %s",
            list(self._config_data.get("llm", {}).get("profiles", {}).keys()),
        )
        self._set_by_path(self._config_data, key_path, value)
        logger.info(
            "[Config] After _set_by_path, profiles: %s",
            list(self._config_data.get("llm", {}).get("profiles", {}).keys()),
        )
        self._config = self._validate(self._config_data)
        self._config_data = self._config.to_json_dict()
        logger.info(
            "[Config] After validation, profiles: %s",
            list(self._config_data.get("llm", {}).get("profiles", {}).keys()),
        )
        if save:
            logger.info("[Config] Calling _save_user_config...")
            self._save_user_config(self._config_data)
            logger.info("[Config] _save_user_config completed")
        else:
            logger.info("[Config] Skipping save (save=False)")

    def save(self) -> None:
        """保存配置到用户配置文件。"""
        if self._config is None:
            raise RuntimeError("Config not loaded, call load() first")
        self._save_user_config(self._config_data)

    def reset_to_default(self) -> None:
        """重置为模板默认配置。"""
        system = self.load_system_config()
        template = self.load_user_template()
        merged = self._normalize_config(merge_configs(system, template))
        self._config = self._validate(merged)
        self._config_data = self._config.to_json_dict()
        self._save_user_config(merged)


# 全局配置实例
g_config = ConfigManager()
