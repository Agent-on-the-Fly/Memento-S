# SPDX-License-Identifier: Apache-2.0
"""
ConfigManager 高级功能测试

运行方式:
    python middleware/config/tests/test_config_manager_advanced.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from middleware.config import ConfigManager


def test_save_and_load():
    """测试配置的保存和重新加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        manager = ConfigManager(str(config_path))

        config = manager.load()
        original_version = config.version

        manager.set("app.theme", "dark", save=True)

        manager2 = ConfigManager(str(config_path))
        config2 = manager2.load()
        assert config2.app.theme == "dark"


def test_replace_user_config():
    """测试 replace_user_config 方法"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        manager = ConfigManager(str(config_path))
        manager.ensure_user_config_file()

        new_config = {
            "app": {"theme": "light", "language": "en-US"},
            "llm": {
                "active_profile": "test",
                "profiles": {
                    "test": {
                        "model": "openai/gpt-4",
                        "api_key": "test-key",
                        "base_url": "https://api.openai.com/v1",
                        "max_tokens": 4096,
                        "temperature": 0.7,
                        "timeout": 120,
                    }
                },
            },
            "skills": {},
            "env": {},
        }
        result = manager.replace_user_config(new_config)
        assert result is None

        raw = manager.get_raw_user_config()
        assert raw["llm"]["profiles"]["test"]["model"] == "openai/gpt-4"


def test_reset_to_default():
    """测试重置配置"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        manager = ConfigManager(str(config_path))
        manager.load()
        manager.set("app.theme", "dark", save=True)

        manager.reset_to_default()
        config_after_reset = manager.load()
        assert config_after_reset.app.theme != "dark"


def test_save_should_not_persist_system_only_fields():
    """set/save 不会把 system_config 独有字段写入用户配置文件"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        manager = ConfigManager(str(config_path))
        manager.load()

        manager.set(
            "llm.profiles.test",
            {
                "model": "openai/gpt-4o-mini",
                "api_key": "k",
                "base_url": "https://api.openai.com/v1",
                "max_tokens": 1024,
                "temperature": 0.3,
                "timeout": 60,
                "extra_headers": {},
                "extra_body": {},
            },
            save=False,
        )
        manager.set("llm.active_profile", "test", save=False)
        manager.save()

        raw = manager.get_raw_user_config()

        assert "name" not in raw.get("app", {})
        assert "theme_options" not in raw.get("app", {})
        assert "language_options" not in raw.get("app", {})
        assert "agent" not in raw
        assert "paths" not in raw
        assert "logging" not in raw
        assert "url" not in raw.get("ota", {})
        assert "test" in raw.get("llm", {}).get("profiles", {})
        assert raw.get("llm", {}).get("active_profile") == "test"


    def test_replace_user_config_should_strip_system_only_fields(self):
        """replace_user_config 应剔除 system-only 字段，并保留用户可写字段"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            manager = ConfigManager(str(config_path))
            manager.load()

            payload = {
                "app": {"theme": "dark", "language": "en-US", "name": "HACKED"},
                "llm": {
                    "active_profile": "custom",
                    "profiles": {
                        "custom": {
                            "model": "openai/gpt-4.1",
                            "api_key": "k",
                            "base_url": "https://api.openai.com/v1",
                            "max_tokens": 2048,
                            "temperature": 0.5,
                            "timeout": 120,
                        }
                    },
                },
                "env": {"TAVILY_API_KEY": "abc"},
                "ota": {"url": "https://evil.example", "auto_check": False},
                "gateway": {
                    "enabled": False,
                    "mode": "bridge",
                    "websocket_host": "0.0.0.0",
                    "websocket_port": 9999,
                    "webhook_host": "0.0.0.0",
                    "webhook_port": 9998,
                },
                "paths": {"workspace_dir": "/tmp/evil"},
                "logging": {"level": "DEBUG"},
                "agent": {"max_iterations": 1},
            }

            result = manager.replace_user_config(payload)
            assert result is None

            raw = manager.get_raw_user_config()

            # system-only 字段应被剔除
            assert "paths" not in raw
            assert "logging" not in raw
            assert "agent" not in raw
            assert "url" not in raw.get("ota", {})

            # gateway 全 section 是 user-managed，应被保留
            gw = raw.get("gateway", {})
            assert gw.get("enabled") is False
            assert gw.get("mode") == "bridge"
            assert gw.get("websocket_host") == "0.0.0.0"
            assert gw.get("websocket_port") == 9999
            assert gw.get("webhook_host") == "0.0.0.0"
            assert gw.get("webhook_port") == 9998

            # 用户字段保留
            assert raw.get("app", {}).get("theme") == "dark"
            assert raw.get("llm", {}).get("active_profile") == "custom"


def test_session_directories():
    """测试会话目录创建"""
    manager = ConfigManager()
    config = manager.load()

    workspace = manager.paths.workspace_dir
    skills = manager.paths.skills_dir
    db = manager.paths.db_dir
    assert workspace is not None
    assert skills is not None
    assert db is not None


def test_all_path_methods():
    """测试所有路径获取方法"""
    manager = ConfigManager()
    config = manager.load()

    paths = [
        ("workspace_dir", manager.paths.workspace_dir),
        ("skills_dir", manager.paths.skills_dir),
        ("db_dir", manager.paths.db_dir),
        ("db_url", manager.get_db_url()),
        ("logs_dir", manager.paths.logs_dir),
    ]

    for name, result in paths:
        assert result is not None


if __name__ == "__main__":
    print("=== ConfigManager 高级功能测试 ===\n")

    print("1. 测试保存和加载...")
    test_save_and_load()
    print("  OK\n")

    print("2. 测试 replace_user_config...")
    test_replace_user_config()
    print("  OK\n")

    print("3. 测试重置配置...")
    test_reset_to_default()
    print("  OK\n")

    print("4. 测试 system-only 字段剔除...")
    test_replace_user_config_should_strip_system_only_fields()
    print("  OK\n")

    print("5. 测试会话目录...")
    test_session_directories()
    print("  OK\n")

    print("6. 测试所有路径方法...")
    test_all_path_methods()
    print("  OK\n")

    print("=== 所有测试通过 ===")
