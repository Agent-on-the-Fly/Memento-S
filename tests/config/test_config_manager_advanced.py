"""
ConfigManager 高级功能测试

覆盖保存、修改、重置等操作
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from middleware.config import ConfigManager


def test_save_and_load():
    """测试配置的保存和重新加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        manager = ConfigManager(str(config_path))

        # 初始加载
        config = manager.load()
        original_version = config.version
        print(f"✓ 初始加载成功，版本: {original_version}")

        # 修改配置
        manager.set("app.theme", "dark", save=True)
        print("✓ 修改配置并保存")

        # 重新加载验证
        manager2 = ConfigManager(str(config_path))
        config2 = manager2.load()
        assert config2.app.theme == "dark", "配置修改未生效"
        print(f"✓ 重新加载验证通过，theme: {config2.app.theme}")


def test_replace_user_config():
    """测试 replace_user_config 方法"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        manager = ConfigManager(str(config_path))

        # 先创建初始配置
        manager.ensure_user_config_file()

        # 准备新的配置数据（完整的 LLM 配置）
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

        # 保存配置
        result = manager.replace_user_config(new_config)
        assert result is None, f"保存失败: {result}"
        print("✓ replace_user_config 成功")

        # 验证保存的配置
        raw = manager.get_raw_user_config()
        assert raw["llm"]["profiles"]["test"]["model"] == "openai/gpt-4"
        print("✓ LLM 配置已正确保存")


def test_reset_to_default():
    """测试重置配置"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        manager = ConfigManager(str(config_path))

        # 加载并修改
        manager.load()
        manager.set("app.theme", "dark", save=True)
        print(f"✓ 修改前 theme: {manager.load().app.theme}")

        # 重置
        manager.reset_to_default()
        config_after_reset = manager.load()
        assert config_after_reset.app.theme != "dark", "重置未生效"
        print(f"✓ 重置后 theme: {config_after_reset.app.theme}")


def test_save_should_not_persist_system_only_fields():
    """测试 set/save 不会把 system_config 独有字段写入用户配置文件。"""
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

        # system_config.json 独有字段不应落盘
        assert "name" not in raw.get("app", {}), "system.app.name 不应写入用户配置"
        assert "theme_options" not in raw.get("app", {}), (
            "system.app.theme_options 不应写入用户配置"
        )
        assert "language_options" not in raw.get("app", {}), (
            "system.app.language_options 不应写入用户配置"
        )
        assert "agent" not in raw, "system.agent 不应写入用户配置"
        assert "paths" not in raw, "system.paths 不应写入用户配置"
        assert "logging" not in raw, "system.logging 不应写入用户配置"

        # ota 节点允许存在，但不应包含 system-only 的 url
        ota = raw.get("ota", {})
        assert "url" not in ota, "system.ota.url 不应写入用户配置"

        # 用户变更应保留
        assert "test" in raw.get("llm", {}).get("profiles", {}), "新增模型应被保存"
        assert raw.get("llm", {}).get("active_profile") == "test"


def test_replace_user_config_should_strip_system_only_fields():
    """replace_user_config 应剔除 system-only 字段，并保留用户可写字段。"""
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
        assert result is None, result

        raw = manager.get_raw_user_config()

        # system-only 字段应被剔除或清理
        assert "name" not in raw.get("app", {})
        assert "paths" not in raw
        assert "logging" not in raw
        assert "agent" not in raw
        assert "url" not in raw.get("ota", {})

        # gateway 仅允许 enabled
        gw = raw.get("gateway", {})
        assert gw.get("enabled") is False
        assert "mode" not in gw
        assert "websocket_host" not in gw
        assert "websocket_port" not in gw
        assert "webhook_host" not in gw
        assert "webhook_port" not in gw

        # 用户字段保留
        assert raw.get("app", {}).get("theme") == "dark"
        assert raw.get("llm", {}).get("active_profile") == "custom"


def test_session_directories():
    """测试会话目录创建"""
    manager = ConfigManager()
    config = manager.load()

    # 测试路径属性访问
    workspace = manager.paths.workspace_dir
    skills = manager.paths.skills_dir
    db = manager.paths.db_dir
    assert workspace is not None, "workspace_dir 未设置"
    assert skills is not None, "skills_dir 未设置"
    assert db is not None, "db_dir 未设置"
    print(f"✓ 路径属性访问: workspace={workspace}")


def test_all_path_methods():
    """测试所有路径获取方法"""
    manager = ConfigManager()
    config = manager.load()

    # 直接通过 paths 属性访问
    paths = [
        ("workspace_dir", manager.paths.workspace_dir),
        ("skills_dir", manager.paths.skills_dir),
        ("db_dir", manager.paths.db_dir),
        ("db_url", manager.get_db_url()),
        ("logs_dir", manager.paths.logs_dir),
    ]

    for name, result in paths:
        assert result is not None, f"{name} 返回 None"
        print(f"✓ {name}: {result}")


if __name__ == "__main__":
    print("=== ConfigManager 高级功能测试 ===\n")

    print("1. 测试保存和加载...")
    test_save_and_load()
    print()

    print("2. 测试 replace_user_config...")
    test_replace_user_config()
    print()

    print("3. 测试重置配置...")
    test_reset_to_default()
    print()

    print("4. 测试 system-only 字段剔除...")
    test_replace_user_config_should_strip_system_only_fields()
    print()

    print("5. 测试会话目录...")
    test_session_directories()
    print()

    print("6. 测试所有路径方法...")
    test_all_path_methods()
    print()

    print("=== 所有测试通过 ===")
