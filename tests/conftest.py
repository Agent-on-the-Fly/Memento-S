"""Pytest 配置文件

注册自定义标记和全局 fixtures。
"""

import pytest


def pytest_configure(config):
    """配置 pytest，注册自定义标记"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require network)"
    )
