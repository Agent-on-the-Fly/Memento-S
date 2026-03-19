"""
Memento-S 日志模块

基于 Loguru 的日志封装，支持：
- 控制台输出（开发调试用）
- 文件输出（按天自动分割、压缩、保留策略）
- 异步队列（多线程/协程安全）
- 使用 ConfigManager 获取日志路径

使用方式：
    from utils.logger import logger, setup_logger

    # 初始化（应用启动时调用一次）
    setup_logger()

    # 使用
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.exception("Exception message")  # 自动包含异常堆栈
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

# Global flag to track if logger has been initialized
_logger_initialized = False


def get_logs_path() -> Path:
    """获取日志目录路径"""
    from utils.path_manager import PathManager

    return PathManager.get_logs_dir()


def get_log_path(log_name: Optional[str] = None) -> Path:
    """获取日志文件路径

    Args:
        log_name: 日志文件名，如果为 None 则使用当天的日期文件名

    Returns:
        日志文件的完整路径
    """
    from utils.path_manager import PathManager

    logs_dir = PathManager.get_logs_dir()

    if log_name is None:
        # 默认使用当天的日期作为文件名
        today = datetime.now().strftime("%Y-%m-%d")
        log_name = f"app_{today}.log"

    return logs_dir / log_name


def setup_logger(
    log_file: Optional[str] = None,
    console_level: str = "DEBUG",
    file_level: str = "INFO",
    rotation: str = "00:00",  # 每天午夜轮转
    retention: str = "30 days",
    compression: str = "zip",
    daily_separate: bool = True,  # 是否每天生成独立的日志文件
    enable_console: bool = False,  # 是否启用控制台输出
    clear_existing: bool = True,  # 是否清除现有配置
) -> None:
    """
    初始化 Loguru 日志配置

    Args:
        log_file: 日志文件名，默认使用当天的日期（如 app_2024-03-07.log）
        console_level: 控制台日志级别，默认 "DEBUG"
        file_level: 文件日志级别，默认 "INFO"
        rotation: 日志轮转策略，默认 "00:00"（每天午夜自动分割）
        retention: 日志保留策略，默认 "30 days"
        compression: 历史日志压缩格式，默认 "zip"
        daily_separate: 是否每天生成独立的日志文件，默认 True
        enable_console: 是否启用控制台输出，默认 False

    调用方式：
        # 使用默认配置（按天分割）
        setup_logger()

        # 自定义配置
        setup_logger(
            log_file="myapp.log",
            console_level="INFO",
            file_level="WARNING",
            rotation="1 day",  # 每天轮转
            retention="7 days",
            daily_separate=True
        )

        # 单文件模式（不按天分割，达到 10MB 才轮转）
        setup_logger(
            log_file="app.log",
            rotation="10 MB",
            daily_separate=False
        )

        # 禁用控制台输出（仅写入文件）
        setup_logger(
            enable_console=False
        )
    """
    # Check if already initialized
    global _logger_initialized
    if _logger_initialized:
        # Logger already set up, skip re-initialization
        return

    # 1. 清除 loguru 默认配置
    logger.remove()

    # 2. 确保日志目录存在
    logs_dir = get_logs_path()
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 3. 确定日志文件路径
    if daily_separate:
        # 按天分割模式：文件名包含日期
        if log_file is None:
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = f"app_{today}.log"
        else:
            # 如果提供了文件名，自动添加日期前缀
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = f"{log_file.rsplit('.', 1)[0]}_{today}.log"
        log_file_path = logs_dir / log_file
    else:
        # 单文件模式：固定文件名
        if log_file is None:
            log_file = "app.log"
        log_file_path = logs_dir / log_file

    # 4. 添加控制台输出（开发调试用，可通过 enable_console 禁用）
    if enable_console and sys.stderr is not None:
        logger.add(
            sys.stderr,
            level=console_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>",
        )

    # 5. 添加文件输出（带轮转、压缩、保留策略）
    logger.add(
        str(log_file_path),
        level=file_level,
        rotation=rotation,  # 按时间轮转（如每天午夜）
        retention=retention,  # 历史日志保留策略
        compression=compression,  # 旧日志压缩
        encoding="utf-8",
        enqueue=True,  # 异步队列，线程/协程安全
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    # 6. 记录初始化信息
    logger.info(f"Logger initialized")
    logger.info(f"Log directory: {logs_dir}")
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"Console level: {console_level}, File level: {file_level}")
    logger.info(f"Rotation: {rotation}, Retention: {retention}")

    # Mark as initialized
    _logger_initialized = True


def get_logger(name: str = None):
    """获取配置好的 logger 实例

    Args:
        name: 模块名（可选，用于兼容标准 logging API）

    Returns:
        配置好的 loguru logger 实例
    """
    if name:
        # 返回带上下文的 logger
        return logger.bind(module=name)
    return logger


# 导出
__all__ = ["logger", "setup_logger", "get_logs_path", "get_log_path", "get_logger"]
