# SPDX-License-Identifier: Apache-2.0
"""统一运行时模式检测

路径策略：
- MEMENTO_HOME: 显式覆盖所有用户数据路径（适用于测试和隔离部署）
- DEV: ~/memento_s
- PRODUCTION: platformdirs.user_data_dir("memento_s", "memento_s")

检测优先级：
1. MEMENTO_ENV 环境变量（dev / production）
2. sys.frozen（PyInstaller 打包自动视为 production）
3. 默认：DEV
"""

from __future__ import annotations

import os
import sys
from enum import Enum
from functools import lru_cache
from pathlib import Path

import platformdirs


def _home_override() -> Path | None:
    """Return the optional application data-root override."""
    value = os.getenv("MEMENTO_HOME", "").strip()
    return Path(value).expanduser() if value else None


class RuntimeMode(Enum):
    """运行时模式枚举"""

    DEV = "dev"
    PRODUCTION = "production"

    # ── 路径计算 ──────────────────────────────────────────

    @property
    def data_dir(self) -> Path:
        """用户数据根目录"""
        if override := _home_override():
            return override
        if self == self.DEV:
            return Path.home() / "memento_s"
        return Path(platformdirs.user_data_dir("memento_s", "memento_s"))

    @property
    def config_dir(self) -> Path:
        """配置文件目录"""
        if override := _home_override():
            return override
        if self == self.DEV:
            return Path.home() / "memento_s"
        return Path(platformdirs.user_config_dir("memento_s", "memento_s"))

    @property
    def logs_dir(self) -> Path:
        """日志目录"""
        if _home_override():
            return self.data_dir / "logs"
        if self == self.DEV:
            return self.data_dir / "logs"
        return Path(platformdirs.user_log_dir("memento_s", "memento_s"))

    @property
    def skills_dir(self) -> Path:
        """Skills 目录"""
        return self.data_dir / "skills"

    @property
    def workspace_dir(self) -> Path:
        """工作区目录"""
        return self.data_dir / "workspace"

    @property
    def db_dir(self) -> Path:
        """数据库目录"""
        return self.data_dir / "db"

    @property
    def venv_dir(self) -> Path:
        """虚拟环境目录"""
        return self.data_dir / ".venv"

    @property
    def context_dir(self) -> Path:
        """上下文目录"""
        return self.data_dir / "context"

    # ── 行为标志 ──────────────────────────────────────────

    @property
    def is_dev(self) -> bool:
        """是否为开发模式"""
        return self == self.DEV

    @property
    def is_production(self) -> bool:
        """是否为生产模式"""
        return self == self.PRODUCTION

    @property
    def is_frozen(self) -> bool:
        """是否 frozen 打包（PyInstaller）"""
        return self == self.PRODUCTION and getattr(sys, "frozen", False)

    @property
    def builtin_root(self) -> Path | None:
        """内置 skills 根目录（仅 production 模式有）"""
        if not self.is_production:
            return None
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass) / "builtin"
        # 兜底：从可执行文件所在目录查找
        if sys.executable:
            return Path(sys.executable).parent / "builtin"
        return None


@lru_cache(maxsize=1)
def detect_runtime_mode() -> RuntimeMode:
    """检测运行时模式（带缓存）

    优先级：
    1. MEMENTO_ENV 环境变量
    2. sys.frozen（PyInstaller 打包视为 production）
    3. 默认：DEV
    """
    # 1. 环境变量优先
    env = os.getenv("MEMENTO_ENV", "").lower()
    if env in {"dev", "development"}:
        return RuntimeMode.DEV
    if env in {"prod", "production"}:
        return RuntimeMode.PRODUCTION

    # 2. PyInstaller 打包自动视为 production
    if getattr(sys, "frozen", False):
        return RuntimeMode.PRODUCTION

    # 3. 默认：DEV
    return RuntimeMode.DEV


def get_runtime_mode() -> RuntimeMode:
    """获取运行时模式（全局单例）"""
    return detect_runtime_mode()


def _print_mode_paths(mode_name: str, mode: RuntimeMode) -> None:
    """打印指定模式的所有路径（调试用）"""
    print(f"[{mode_name}]")
    print(f"  data_dir:      {mode.data_dir}")
    print(f"  config_dir:    {mode.config_dir}")
    print(f"  logs_dir:      {mode.logs_dir}")
    print(f"  skills_dir:    {mode.skills_dir}")
    print(f"  workspace_dir: {mode.workspace_dir}")
    print(f"  db_dir:        {mode.db_dir}")
    print(f"  venv_dir:      {mode.venv_dir}")
    print(f"  context_dir:   {mode.context_dir}")
    if mode.builtin_root:
        print(f"  builtin_root:  {mode.builtin_root}")


def _main() -> None:
    """自检：打印 DEV 和 PRODUCTION 两种模式的路径"""
    print("[RuntimeMode] self-test")
    print(f"  detected_mode: {get_runtime_mode().value}")
    print(f"  sys.frozen:    {getattr(sys, 'frozen', False)}")
    print(f"  MEMENTO_ENV:   {os.getenv('MEMENTO_ENV', '(not set)')}")
    print(f"  MEMENTO_HOME:  {os.getenv('MEMENTO_HOME', '(not set)')}")
    print()
    _print_mode_paths("DEV", RuntimeMode.DEV)
    print()
    _print_mode_paths("PRODUCTION", RuntimeMode.PRODUCTION)


if __name__ == "__main__":
    _main()
