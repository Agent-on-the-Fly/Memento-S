"""market 测试配置"""

import pytest


def pytest_configure(config):
    """配置 pytest-asyncio"""
    config.option.asyncio_mode = "auto"
