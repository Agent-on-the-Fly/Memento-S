"""
Memento-S 配置管理模块（文件模式）

提供：
- JSON 配置文件加载和 Schema 校验
- 用户目录 ~/.memento_s 自动初始化
- 缺失配置文件时从模板自动复制
- 启动自检 bootstrap

示例:
    >>> from bootstrap import bootstrap
    >>> from middleware.config import g_config
    >>> bootstrap()  # 启动自检：目录与配置校验（内部会初始化 g_config）
    >>>
    >>> # 直接访问配置属性
    >>> provider = g_config.llm.current.provider
    >>> model = g_config.llm.current.model
    >>>
    >>> # 或使用 ConfigManager 的方法
    >>> db_url = g_config.get_db_url()
    >>> skills_path = g_config.get_skills_path()
"""

from .config_manager import ConfigManager, g_config
from .config_models import GlobalConfig


__all__ = [
    "ConfigManager",
    "GlobalConfig",
    "g_config",
]
