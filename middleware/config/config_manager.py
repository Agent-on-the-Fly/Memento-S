# SPDX-License-Identifier: Apache-2.0
"""
Memento-S 配置管理模块 v2 - 三层隔离架构

架构：
- System Config: 系统配置（只读，来自 system_config.json）
- User Config: 用户配置（读写，来自 ~/memento_s/config.json）
- Runtime Config: 运行时配置（System + User 的合并结果，用于读取）

写入隔离：
- get() 读取 Runtime Config（已合并）
- set() 修改 User Config，然后重新合并到 Runtime
- save() 只保存 User Config 到磁盘
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import time
from datetime import datetime
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import validate

from .schemas.config_models import GlobalConfig
from .schema_meta import SchemaMetadata
from utils.path_manager import PathManager

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器 v2 - 三层隔离架构"""

    _CONFIG_PACKAGE = "middleware.config"
    _SYSTEM_CONFIG = "system_config.json"
    _USER_TEMPLATE = "user_config_tlp.json"
    _SCHEMA_FILE = "user_config_schema.json"

    def __init__(self, config_path: str | None = None):
        """初始化配置管理器。

        Args:
            config_path: 用户配置文件路径，None 则使用默认路径
        """
        self.user_config_path = (
            Path(config_path).expanduser().resolve()
            if config_path
            else PathManager.get_config_file()
        )

        # 三层存储（物理隔离）
        self._system_data: dict[str, Any] = {}  # 系统配置（只读）
        self._user_data: dict[str, Any] = {}  # 用户配置（读写）
        self._runtime_data: dict[str, Any] = {}  # 运行时合并配置（读取）

        # Pydantic 强类型对象
        self._system_config: GlobalConfig | None = None
        self._user_config: GlobalConfig | None = None
        self._runtime_config: GlobalConfig | None = None

    def __getattr__(self, name: str) -> Any:
        """代理对 Runtime Config 属性的访问。"""
        if self._runtime_config is None:
            raise RuntimeError(f"配置尚未加载，无法访问 '{name}'。请先调用 load()")
        return getattr(self._runtime_config, name)

    @property
    def paths(self):
        """访问路径配置。"""
        if self._runtime_config is None:
            raise RuntimeError("配置尚未加载。请先调用 load()")
        return self._runtime_config.paths

    def is_loaded(self) -> bool:
        """检查配置是否已加载。"""
        return self._runtime_config is not None

    def get_db_url(self) -> str:
        """返回数据库连接 URL。"""
        return f"sqlite+aiosqlite:///{self.paths.db_dir / 'memento_s.db'}"

    def get_data_dir(self) -> Path:
        """返回数据目录路径。"""
        return Path(self.paths.data_dir)

    def get_workspace_dir(self) -> Path:
        """返回工作区目录路径。"""
        return Path(self.paths.workspace_dir)

    def get_skills_path(self) -> Path:
        """返回 skills 目录路径。"""
        return Path(self.paths.skills_dir)

    def get_db_dir(self) -> Path:
        """返回数据库目录路径。"""
        return Path(self.paths.db_dir)

    def get_logs_dir(self) -> Path:
        """返回日志目录路径。"""
        return Path(self.paths.logs_dir)

    def get_venv_dir(self) -> Path:
        """返回虚拟环境目录路径。"""
        return Path(self.paths.venv_dir)

    def get_context_dir(self) -> Path:
        """返回上下文目录路径。"""
        return Path(self.paths.context_dir)

    def get_builtin_skills_path(self) -> Path:
        """返回内置 skills 目录路径。

        根据运行时模式选择策略：
        - DEV: 从源码项目根向上查找 builtin/skills
        - PRODUCTION: 从 _MEIPASS 或可执行文件所在目录查找
        """
        from utils.runtime_mode import get_runtime_mode, RuntimeMode

        mode = get_runtime_mode()

        if mode == RuntimeMode.PRODUCTION:
            # PRODUCTION: 从 _MEIPASS 或可执行文件所在目录查找
            import sys

            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                builtin_path = Path(meipass) / "builtin" / "skills"
                if builtin_path.exists():
                    return builtin_path
            # 兜底：从可执行文件所在目录查找
            if sys.executable:
                builtin_path = Path(sys.executable).parent / "builtin" / "skills"
                if builtin_path.exists():
                    return builtin_path

        # DEV: 从源码项目根向上查找
        # 依次向上查找包含 marker 文件的目录作为项目根
        marker_files = ["pyproject.toml", ".git", "bootstrap.py"]
        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            if any((parent / marker).exists() for marker in marker_files):
                builtin_path = parent / "builtin" / "skills"
                if builtin_path.exists():
                    return builtin_path

        raise RuntimeError(
            "无法找到 builtin/skills 目录。"
            "DEV 模式: 期望在项目根目录下。"
            "PRODUCTION 模式: 期望在打包资源中。"
        )

    def get_db_path(self) -> Path:
        """返回数据库文件完整路径。"""
        return self.get_db_dir() / "memento_s.db"

    def get_env(self) -> dict[str, Any]:
        """返回环境变量配置字典。"""
        if self._runtime_config is None:
            raise RuntimeError("配置尚未加载。请先调用 load()")
        return self._runtime_config.env or {}

    def get_log_path(self, log_name: str) -> Path:
        """返回指定日志文件的完整路径。

        Args:
            log_name: 日志文件名

        Returns:
            日志文件完整路径
        """
        return self.get_logs_dir() / log_name

    def get_skill_path(self, skill_name: str) -> Path:
        """返回指定 skill 的目录路径。

        Args:
            skill_name: skill 名称

        Returns:
            skill 目录路径
        """
        return self.get_skills_path() / skill_name

    # ========== 其他属性 ==========

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
        """加载 JSON Schema。"""
        return self._load_resource(self._SCHEMA_FILE)

    def load_system_config(self) -> dict[str, Any]:
        """加载系统配置模板。"""
        return self._load_resource(self._SYSTEM_CONFIG)

    def load_user_template(self) -> dict[str, Any]:
        """加载用户配置模板。"""
        return self._load_resource(self._USER_TEMPLATE)

    def user_config_exists(self) -> bool:
        """检查用户配置文件是否存在。"""
        return self.user_config_path.exists()

    def ensure_user_config_dir(self) -> Path:
        """确保用户配置目录存在。"""
        self.user_config_dir.mkdir(parents=True, exist_ok=True)
        return self.user_config_dir

    def ensure_user_config_file(self) -> Path:
        """确保用户配置文件存在，不存在则创建空配置。"""
        self.ensure_user_config_dir()
        if not self.user_config_path.exists():
            # 创建空用户配置（不从模板复制，避免默认模型）
            # 注意：version 字段会在 load() 方法中从 system_config 获取并写入
            empty_user_config = {
                "app": {"theme": "system", "language": "zh-CN"},
                "llm": {"active_profile": "", "profiles": {}},
                "env": {},
                "im": {
                    "platform": "feishu",
                    "feishu": {"enabled": False},
                    "dingtalk": {"enabled": False},
                    "wecom": {"enabled": False},
                    "wechat": {"enabled": False},
                },
                "gateway": {"enabled": True},
            }
            self._write_json(self.user_config_path, empty_user_config)
        return self.user_config_path

    def load(self) -> GlobalConfig:
        """加载配置：System → User → 合并为 Runtime。

        Returns:
            Runtime Config（System + User 的合并结果）

        Raises:
            RuntimeError: 如果加载失败
        """
        try:
            # 1. 加载系统配置（只读）
            self._system_data = self.load_system_config()

            # 2. 确保用户配置文件存在（如果不存在则创建空配置）
            self.ensure_user_config_file()

            # 3. 加载用户配置
            self._user_data = self._load_user_config()

            # 4. 合并为运行时配置（User 覆盖 System）
            self._runtime_data = self._merge(self._system_data, self._user_data)

            # 5. 补全路径配置（从 PathManager 获取实际路径）
            # 注意：强制覆盖，确保使用 PathManager 的实际路径
            self._runtime_data["paths"] = {
                "data_dir": PathManager.get_data_dir(),
                "workspace_dir": PathManager.get_workspace_dir(),
                "skills_dir": PathManager.get_skills_dir(),
                "db_dir": PathManager.get_db_dir(),
                "logs_dir": PathManager.get_logs_dir(),
                "venv_dir": PathManager.get_venv_dir(),
                "context_dir": PathManager.get_context_dir(),
            }

            # 6. 创建 Pydantic 对象（只验证合并后的运行时配置）
            # 注意：system_data 和 user_data 单独可能不完整，只有合并后才是完整配置
            self._runtime_config = GlobalConfig.model_validate(self._runtime_data)

            logger.info("配置加载成功")
            return self._runtime_config

        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise RuntimeError(f"配置加载失败: {e}") from e

    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值（从 Runtime Config 读取）。

        Args:
            key_path: 配置路径，如 "llm.active_profile" 或 "app.theme"
            default: 默认值，如果路径不存在则返回

        Returns:
            配置值，或默认值
        """
        if self._runtime_config is None:
            raise RuntimeError("配置尚未加载。请先调用 load()")
        return self._get_by_path(self._runtime_data, key_path, default)

    def set(self, key_path: str, value: Any, save: bool = True) -> None:
        """设置配置值（只修改 User Config）。

        Args:
            key_path: 配置路径
            value: 配置值
            save: 是否立即保存到磁盘

        Raises:
            RuntimeError: 配置未加载
            ValueError: 尝试修改系统只读字段
        """
        if self._runtime_config is None:
            raise RuntimeError("配置尚未加载。请先调用 load()")

        # 检查是否是系统只读字段
        if self._is_system_readonly(key_path):
            logger.warning(f"[Config] 拒绝修改系统配置字段: {key_path}")
            raise ValueError(f"系统配置字段不可修改: {key_path}")

        # 修改 User 配置
        self._set_by_path(self._user_data, key_path, value)

        # 重新合并（User 覆盖 System）
        self._runtime_data = self._merge(self._system_data, self._user_data)

        # 重新补全路径配置（从 PathManager 获取实际路径）
        # 注意：强制覆盖，确保使用 PathManager 的实际路径
        self._runtime_data["paths"] = {
            "data_dir": PathManager.get_data_dir(),
            "workspace_dir": PathManager.get_workspace_dir(),
            "skills_dir": PathManager.get_skills_dir(),
            "db_dir": PathManager.get_db_dir(),
            "logs_dir": PathManager.get_logs_dir(),
            "venv_dir": PathManager.get_venv_dir(),
            "context_dir": PathManager.get_context_dir(),
        }

        self._runtime_config = GlobalConfig.model_validate(self._runtime_data)

        logger.info(f"[Config] 设置 {key_path} = {value}")

        # 可选：立即保存
        if save:
            self.save()

    def save(self) -> None:
        """保存 User Config 到磁盘（绝对隔离 System 配置）。

        只保存 _user_data，不包含任何系统配置。
        """
        if self._user_data is None:
            raise RuntimeError("用户配置未加载")

        # 1. 创建备份
        self._create_config_backup()

        # 2. 验证用户配置
        try:
            self._validate_user_config(self._user_data)
        except jsonschema.ValidationError as e:
            logger.error(f"用户配置验证失败: {e.message}")
            raise

        # 3. 直接写入 User 数据（无需过滤，因为本身就是隔离的）
        self._write_json(self.user_config_path, self._user_data)

        logger.info(f"用户配置已保存到: {self.user_config_path}")

    def reload(self) -> GlobalConfig:
        """重新加载配置（从磁盘）。

        Returns:
            重新加载的 Runtime Config
        """
        logger.info("重新加载配置...")
        return self.load()

    def get_raw_user_config(self) -> dict[str, Any]:
        """获取原始 User Config（字典形式）。

        注意：如果配置尚未加载，会从磁盘直接读取。
        """
        # 如果 _user_data 为空，尝试从磁盘读取
        if not self._user_data:
            try:
                return self._load_user_config()
            except Exception:
                return {}
        return copy.deepcopy(self._user_data)

    def save_user_config_direct(self, user_config: dict[str, Any]) -> None:
        """直接保存用户配置到磁盘（不依赖 load()）。

        用于 bootstrap 阶段等需要在 load() 之前写入配置的场景。

        Args:
            user_config: 用户配置字典
        """
        # 确保目录存在
        self.ensure_user_config_dir()

        # 验证配置（可选，bootstrap 阶段可以跳过）
        try:
            self._validate_user_config(user_config)
        except Exception as e:
            logger.warning(f"用户配置验证警告: {e}")

        # 直接写入
        self._write_json(self.user_config_path, user_config)
        logger.debug(f"用户配置已直接保存到: {self.user_config_path}")

    def replace_user_config(self, user_config: dict[str, Any]) -> None:
        """替换整个 User Config。

        Args:
            user_config: 新的用户配置字典（可能包含系统字段，但只会保存用户字段）
        """
        # 0. 确保系统配置已加载（如果尚未加载）
        if not self._system_data:
            self._system_data = self.load_system_config()

        # 1. 用 _is_system_readonly 过滤，只保留用户可写的字段
        filtered_user_data: dict[str, Any] = {}
        for key, value in user_config.items():
            if self._is_system_readonly(key):
                continue
            filtered_user_data[key] = copy.deepcopy(value)

        # 2. 验证提取的用户配置（使用 user schema）
        self._validate_user_config(filtered_user_data)

        # 3. 更新 User 配置
        self._user_data = filtered_user_data

        # 4. 使用 System + User 合并更新运行时
        self._runtime_data = self._merge(self._system_data, self._user_data)

        # 5. 补全路径配置
        self._runtime_data["paths"] = {
            "data_dir": PathManager.get_data_dir(),
            "workspace_dir": PathManager.get_workspace_dir(),
            "skills_dir": PathManager.get_skills_dir(),
            "db_dir": PathManager.get_db_dir(),
            "logs_dir": PathManager.get_logs_dir(),
            "venv_dir": PathManager.get_venv_dir(),
            "context_dir": PathManager.get_context_dir(),
        }

        # 6. 验证运行时配置（使用完整配置）
        self._runtime_config = GlobalConfig.model_validate(self._runtime_data)

        # 7. 保存到磁盘（只保存过滤后的用户字段）
        self._write_json(self.user_config_path, self._user_data)

        logger.info("用户配置已完全替换")

    def reset_to_default(self) -> None:
        """重置 User Config 为默认（空配置）。"""
        empty_config = {
            "version": "1.0.0",
            "app": {"theme": "system", "language": "zh-CN"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {},
            "im": {
                "platform": "feishu",
                "feishu": {"enabled": False},
                "dingtalk": {"enabled": False},
                "wecom": {"enabled": False},
                "wechat": {"enabled": False},
            },
            "gateway": {"enabled": True},
        }
        self.replace_user_config(empty_config)
        logger.info("用户配置已重置为默认值")

    # ========== 内部方法 ==========

    def _merge(self, system: dict, user: dict) -> dict:
        """统一的 merge 策略：SchemaMetadata 驱动。

        规则：
        - x-managed-by: user 的字段 → 直接用 user 值（不递归合并模板）
        - 其他字段 → 递归合并，system 为基础，user 覆盖

        用于 load()、set()、replace_user_config()。

        Args:
            system: 系统配置（基础）
            user: 用户配置（覆盖）

        Returns:
            合并后的配置
        """
        schema = self.load_schema()
        return SchemaMetadata.merge_respecting_metadata(system, user, schema)

    def _is_system_readonly(self, key_path: str) -> bool:
        """检查字段是否是系统只读。

        通过 schema metadata 的 x-managed-by 判断。
        - 如果 key_path 未在 schema 中标记为 x-managed-by: user，则系统只读
        - 支持嵌套路径（如 gateway.mode）

        Args:
            key_path: 配置路径，如 "llm.active_profile" 或 "gateway.mode"

        Returns:
            True 如果是系统只读字段
        """
        schema = self.load_schema()
        return not SchemaMetadata.is_user_managed(schema, key_path)

    def _load_user_config(self) -> dict[str, Any]:
        """从磁盘加载用户配置。"""
        with open(self.user_config_path, encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        """原子写入 JSON 文件。"""
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                import os

                os.fsync(f.fileno())
            except Exception:
                pass
        tmp_path.replace(path)

    def _create_config_backup(self) -> Path | None:
        """创建配置备份。"""
        try:
            if not self.user_config_path.exists():
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"config_backup_{timestamp}.json"
            backup_dir = self.user_config_dir / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)

            backup_path = backup_dir / backup_name
            shutil.copy2(self.user_config_path, backup_path)

            # 清理旧备份（保留最近 10 个）
            self._cleanup_old_backups(backup_dir, max_backups=10)

            logger.debug(f"配置已备份到: {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"配置备份失败: {e}")
            return None

    def _cleanup_old_backups(self, backup_dir: Path, max_backups: int = 10) -> None:
        """清理旧备份。"""
        try:
            backup_files = sorted(
                backup_dir.glob("config_backup_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old_backup in backup_files[max_backups:]:
                old_backup.unlink()
        except Exception as e:
            logger.warning(f"清理旧备份失败: {e}")

    def _validate_user_config(self, user_data: dict[str, Any]) -> None:
        """验证用户配置是否符合 Schema。"""
        schema = self.load_schema()
        validate(instance=user_data, schema=schema)

    def _get_by_path(self, data: dict, key_path: str, default: Any = None) -> Any:
        """按路径获取字典值。"""
        keys = key_path.split(".")
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def _set_by_path(self, data: dict, key_path: str, value: Any) -> None:
        """按路径设置字典值。"""
        keys = key_path.split(".")
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value


# 全局单例实例
g_config = ConfigManager()
