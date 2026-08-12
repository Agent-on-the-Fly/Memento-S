# SPDX-License-Identifier: Apache-2.0
"""Cross-platform path manager for Memento-S.

统一路径入口，内部委托给 RuntimeMode。
保留 packaged 参数以兼容旧调用，但已废弃。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from utils.runtime_mode import RuntimeMode, get_runtime_mode


class PathManager:
    """Centralized cross-platform path provider.

    所有路径方法内部委托给 RuntimeMode，保持接口兼容。
    """

    # 应用标识（兼容旧代码）
    APP_NAME = "memento_s"
    APP_AUTHOR = "memento_s"

    @classmethod
    def is_packaged_runtime(cls) -> bool:
        """判断是否为打包运行环境（已废弃，使用 RuntimeMode）"""
        warnings.warn(
            "PathManager.is_packaged_runtime() 已废弃，请使用 RuntimeMode",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_runtime_mode() == RuntimeMode.PRODUCTION

    @classmethod
    def _resolve_packaged_mode(cls, packaged: bool | None) -> RuntimeMode:
        """解析运行时模式（兼容旧接口，已废弃）"""
        if packaged is not None:
            warnings.warn(
                "PathManager 的 packaged 参数已废弃，请使用 MEMENTO_ENV 环境变量",
                DeprecationWarning,
                stacklevel=2,
            )
            return RuntimeMode.PRODUCTION if packaged else RuntimeMode.DEV
        return get_runtime_mode()

    # ── 路径方法（委托给 RuntimeMode）────────────────────────

    @classmethod
    def get_home_dir(cls, packaged: bool | None = None) -> Path:
        """返回用户主目录"""
        _ = cls._resolve_packaged_mode(packaged)
        return Path.home()

    @classmethod
    def get_project_root_dir(cls, packaged: bool | None = None) -> Path:
        """返回应用根目录"""
        mode = cls._resolve_packaged_mode(packaged)
        return mode.config_dir

    @classmethod
    def get_config_file(cls, packaged: bool | None = None) -> Path:
        """返回配置文件路径"""
        mode = cls._resolve_packaged_mode(packaged)
        return mode.config_dir / "config.json"

    @classmethod
    def get_data_dir(cls, packaged: bool | None = None) -> Path:
        """返回用户数据根目录"""
        mode = cls._resolve_packaged_mode(packaged)
        return mode.data_dir

    @classmethod
    def get_workspace_dir(cls, packaged: bool | None = None) -> Path:
        """返回工作区目录"""
        mode = cls._resolve_packaged_mode(packaged)
        return mode.workspace_dir

    @classmethod
    def get_skills_dir(cls, packaged: bool | None = None) -> Path:
        """返回 skills 目录"""
        mode = cls._resolve_packaged_mode(packaged)
        return mode.skills_dir

    @classmethod
    def get_db_dir(cls, packaged: bool | None = None) -> Path:
        """返回数据库目录"""
        mode = cls._resolve_packaged_mode(packaged)
        return mode.db_dir

    @classmethod
    def get_logs_dir(cls, packaged: bool | None = None) -> Path:
        """返回日志目录"""
        mode = cls._resolve_packaged_mode(packaged)
        return mode.logs_dir

    @classmethod
    def get_venv_dir(cls, packaged: bool | None = None) -> Path:
        """返回虚拟环境目录"""
        mode = cls._resolve_packaged_mode(packaged)
        return mode.venv_dir

    @classmethod
    def get_mcp_config_file(cls, packaged: bool | None = None) -> Path:
        """返回 MCP 配置文件路径"""
        mode = cls._resolve_packaged_mode(packaged)
        return mode.config_dir / "mcp.json"

    @classmethod
    def get_context_dir(cls, packaged: bool | None = None) -> Path:
        """返回上下文数据目录"""
        mode = cls._resolve_packaged_mode(packaged)
        return mode.context_dir


def _print_mode_paths(mode_name: str, mode: RuntimeMode) -> None:
    """打印指定模式的路径（调试用）"""
    print(f"[{mode_name}]")
    print(f"  project_root_dir: {mode.config_dir}")
    print(f"  data_dir:         {mode.data_dir}")
    print(f"  workspace_dir:    {mode.workspace_dir}")
    print(f"  skills_dir:       {mode.skills_dir}")
    print(f"  db_dir:           {mode.db_dir}")
    print(f"  logs_dir:         {mode.logs_dir}")
    print(f"  venv_dir:         {mode.venv_dir}")


def _main() -> None:
    """自检：打印两种模式的路径"""
    print("[PathManager] self-test")
    print(f"  current_mode: {get_runtime_mode().value}")
    print()
    _print_mode_paths("DEV", RuntimeMode.DEV)
    print()
    _print_mode_paths("PRODUCTION", RuntimeMode.PRODUCTION)


if __name__ == "__main__":
    _main()
