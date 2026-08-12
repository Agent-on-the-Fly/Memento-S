# SPDX-License-Identifier: Apache-2.0
"""日志配置工具 - 统一管理日志截断长度"""

from __future__ import annotations

import os
from typing import Final


# 优先级：环境变量 > 配置文件 > 默认值
def is_verbose_logging() -> bool:
    """检查是否启用详细日志模式

    优先级：
    1. 环境变量 MEMENTO_S_LOG_VERBOSE (true/false)
    2. 配置文件 logging.verbose
    3. 默认值 False
    """
    # 1. 检查环境变量
    env_value = os.getenv("MEMENTO_S_LOG_VERBOSE", "").lower()
    if env_value in ("true", "1", "yes"):
        return True
    if env_value in ("false", "0", "no"):
        return False

    # 2. 检查配置文件
    try:
        from middleware.config import g_config

        if g_config and hasattr(g_config, "logging"):
            return getattr(g_config.logging, "verbose", False)
    except Exception:
        pass

    # 3. 默认值
    return False


# 截断长度常量
class LogTruncation:
    """日志截断长度配置"""

    # 短预览 (用于 info 级别)
    SHORT: Final[int] = 200

    # 中等长度 (用于一般 debug)
    MEDIUM: Final[int] = 1000

    # 长预览 (用于 verbose 模式)
    LONG: Final[int] = 5000

    @classmethod
    def get(cls, default: int = 500, verbose_override: int | None = None) -> int:
        """获取当前截断长度

        Args:
            default: 默认截断长度
            verbose_override: verbose 模式下的覆盖长度 (None 表示不截断)

        Returns:
            当前应该使用的截断长度
        """
        if is_verbose_logging():
            return verbose_override if verbose_override is not None else cls.LONG
        return default

    @classmethod
    def truncate(
        cls, text: str | None, default: int = 500, verbose_override: int | None = None
    ) -> str:
        """截断文本

        Args:
            text: 要截断的文本
            default: 默认截断长度
            verbose_override: verbose 模式下的覆盖长度

        Returns:
            截断后的文本
        """
        if text is None:
            return ""

        text_str = str(text)
        max_len = cls.get(default, verbose_override)

        if len(text_str) <= max_len:
            return text_str

        return text_str[:max_len] + "..."


# 便捷函数
def log_preview(text: str | None, default: int = 500) -> str:
    """获取日志预览文本 (自动根据 verbose 设置截断)"""
    return LogTruncation.truncate(text, default=default, verbose_override=None)


def log_preview_short(text: str | None) -> str:
    """短预览 (200字符，verbose 模式下不截断)"""
    return LogTruncation.truncate(text, default=200, verbose_override=None)


def log_preview_medium(text: str | None) -> str:
    """中等预览 (500字符，verbose 模式下 5000字符)"""
    return LogTruncation.truncate(text, default=500, verbose_override=5000)


def log_preview_long(text: str | None) -> str:
    """长预览 (1000字符，verbose 模式下不截断)"""
    return LogTruncation.truncate(text, default=1000, verbose_override=None)
