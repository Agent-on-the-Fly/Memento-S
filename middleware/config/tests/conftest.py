# SPDX-License-Identifier: Apache-2.0
"""
middleware.config 测试的共享 fixtures
"""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path

import pytest

from middleware.config import ConfigManager


@pytest.fixture
def temp_config_dir():
    """创建临时配置目录，测试后清理"""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def config_manager(temp_config_dir):
    """创建配置管理器实例（临时目录）"""
    config_path = temp_config_dir / "config.json"
    return ConfigManager(str(config_path))


@pytest.fixture
def config_manager_with_config(temp_config_dir):
    """创建配置管理器并确保配置文件存在"""
    manager = ConfigManager(str(temp_config_dir / "config.json"))
    manager.ensure_user_config_file()
    return manager