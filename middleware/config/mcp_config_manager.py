# SPDX-License-Identifier: Apache-2.0
"""MCP 配置管理器

独立管理 ~/memento_s/mcp.json，与 ConfigManager（config.json）解耦。

职责：
- 加载/保存 mcp.json
- 提供 get_servers()、set_server()、remove_server() 等配置管理接口
- 初始化时若 mcp.json 不存在则从 mcp_config_tpl.json 复制
- 使用 JSON Schema 验证配置格式
- 支持 local 和 remote 两种类型的 MCP server

支持的 Server 类型：
- local: 本地进程，通过 stdio 通信
- remote: 远程服务，通过 HTTP/SSE 通信
"""

from __future__ import annotations

import copy
import json
import logging
from enum import Enum
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import validate

from .schemas.mcp_config_schemas import McpConfig, McpServerConfig
from utils.path_manager import PathManager

logger = logging.getLogger(__name__)


class McpServerType(str, Enum):
    """MCP Server 类型枚举。"""
    LOCAL = "local"   # 本地进程，通过 stdio 通信
    REMOTE = "remote"  # 远程服务，通过 HTTP/SSE 通信


class McpConfigManager:
    """MCP 配置管理器 - 独立管理 mcp.json"""

    _CONFIG_PACKAGE = "middleware.config"
    _TEMPLATE_FILE = "mcp_config_tpl.json"
    _SCHEMA_FILE = "mcp_config_schema.json"

    def __init__(self):
        self.mcp_config_path = PathManager.get_mcp_config_file()
        self._model: McpConfig | None = None  # cached Pydantic model

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_resource(filename: str) -> dict[str, Any]:
        """从包内资源加载 JSON 文件（带缓存）。"""
        text = (
            resources.files(McpConfigManager._CONFIG_PACKAGE)
            .joinpath(filename)
            .read_text(encoding="utf-8")
        )
        return json.loads(text)

    def load_schema(self) -> dict[str, Any]:
        """加载 JSON Schema。"""
        return self._load_resource(self._SCHEMA_FILE)

    def load_template(self) -> dict[str, Any]:
        """加载 MCP 配置模板。"""
        return self._load_resource(self._TEMPLATE_FILE)

    def mcp_config_exists(self) -> bool:
        """检查 mcp.json 是否存在。"""
        return self.mcp_config_path.exists()

    @property
    def mcp_config_dir(self) -> Path:
        """返回 mcp 配置目录。"""
        return self.mcp_config_path.parent

    def ensure_mcp_config_file(self) -> Path:
        """确保 mcp.json 存在，不存在则从模板复制。"""
        self.mcp_config_dir.mkdir(parents=True, exist_ok=True)
        if not self.mcp_config_path.exists():
            template = self.load_template()
            self._write_json(self.mcp_config_path, template)
            logger.info(f"[McpConfigManager] 从模板创建 mcp.json: {self.mcp_config_path}")
        return self.mcp_config_path

    def load(self) -> dict[str, Any]:
        """加载 mcp.json 并验证格式。"""
        with open(self.mcp_config_path, encoding="utf-8") as f:
            data = json.load(f)
        self._validate(data)
        self._data = copy.deepcopy(data)
        self._model = None  # invalidate cached Pydantic model
        return self._data

    def save(self) -> None:
        """保存 mcp.json 到磁盘。"""
        if not hasattr(self, "_data") or self._data is None:
            raise RuntimeError("配置未加载，无法保存")
        self._write_json(self.mcp_config_path, self._data)
        logger.info(f"[McpConfigManager] 配置已保存: {self.mcp_config_path}")

    def get_mcp_config(self) -> dict[str, Any]:
        """返回原始 mcp 配置字典（供 MCPToolLoader 使用）。"""
        if not hasattr(self, "_data") or self._data is None:
            return self.load()
        return copy.deepcopy(self._data)

    def get_servers(self) -> dict[str, Any]:
        """获取 mcp servers 字典（dict 形式，供 MCPToolLoader 使用）。"""
        config = self.get_mcp_config()
        return config.get("mcp", {})

    def load_model(self) -> McpConfig:
        """加载 mcp.json 并返回 Pydantic 模型。"""
        data = self.get_mcp_config()
        return McpConfig.model_validate(data)

    def get_mcp_config_model(self) -> McpConfig:
        """返回 Pydantic McpConfig 模型（已加载则缓存）。"""
        if not hasattr(self, "_model") or self._model is None:
            self._model = self.load_model()
        return self._model

    def is_enabled(self) -> bool:
        """返回 enabled 字段。"""
        config = self.get_mcp_config()
        return config.get("enabled", False)

    def set_server(self, name: str, config: dict[str, Any]) -> None:
        """添加或更新单个 MCP server。"""
        if not hasattr(self, "_data") or self._data is None:
            self.load()
        if "mcp" not in self._data:
            self._data["mcp"] = {}
        self._data["mcp"][name] = config
        logger.info(f"[McpConfigManager] 更新 server '{name}'")
        self.save()

    def remove_server(self, name: str) -> None:
        """删除单个 MCP server。"""
        if not hasattr(self, "_data") or self._data is None:
            self.load()
        if "mcp" in self._data and name in self._data["mcp"]:
            del self._data["mcp"][name]
            logger.info(f"[McpConfigManager] 删除 server '{name}'")
            self.save()

    def set_enabled(self, enabled: bool) -> None:
        """设置 enabled 字段。"""
        if not hasattr(self, "_data") or self._data is None:
            self.load()
        self._data["enabled"] = enabled
        logger.info(f"[McpConfigManager] enabled = {enabled}")
        self.save()

    def reload(self) -> dict[str, Any]:
        """重新加载配置。"""
        self._data = None
        self._model = None
        return self.load()

    def _validate(self, data: dict[str, Any]) -> None:
        """使用 JSON Schema 验证配置格式。"""
        schema = self.load_schema()
        validate(instance=data, schema=schema)

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


# 全局单例
g_mcp_config_manager = McpConfigManager()
