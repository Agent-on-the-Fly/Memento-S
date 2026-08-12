# SPDX-License-Identifier: Apache-2.0
"""
配置安全性测试

运行方式:
    python -m pytest middleware/config/tests/test_config_safety.py -v
"""

from __future__ import annotations

import json
import pytest

from middleware.config.config_manager import ConfigManager
from middleware.config.migrations import merge_configs, merge_template_defaults


class TestConfigSafety:
    """配置安全性测试套件"""

    def test_save_does_not_reread_file(self, config_manager):
        """save() 时不重新读取文件，避免覆盖"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            "llm.profiles.test_model",
            {
                "model": "test/model",
                "api_key": "test_key",
                "base_url": "https://test.com",
            },
            save=False,
        )
        config_manager.set("llm.active_profile", "test_model", save=False)

        # 模拟外部修改
        external_config = {
            "version": "1.0.0",
            "app": {"theme": "system", "language": "en-US"},
            "llm": {
                "active_profile": "external_model",
                "profiles": {
                    "external_model": {
                        "model": "external/model",
                        "api_key": "external_key",
                        "base_url": "https://external.com",
                    }
                },
            },
            "env": {},
            "im": {
                "feishu": {"enabled": False},
                "dingtalk": {"enabled": False},
                "wecom": {"enabled": False},
                "wechat": {"enabled": False},
            },
            "gateway": {"enabled": True},
        }
        with open(config_manager.user_config_path, "w", encoding="utf-8") as f:
            json.dump(external_config, f, indent=2)

        config_manager.save()

        with open(config_manager.user_config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        assert "test_model" in saved_config.get("llm", {}).get("profiles", {})

    def test_nested_profiles_not_cleared(self, config_manager):
        """嵌套的 llm.profiles 不会被清除"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        profiles = {
            "model_a": {
                "model": "provider/model_a",
                "api_key": "key_a",
                "base_url": "https://a.com",
            },
            "model_b": {
                "model": "provider/model_b",
                "api_key": "key_b",
                "base_url": "https://b.com",
            },
            "model_c": {
                "model": "provider/model_c",
                "api_key": "key_c",
                "base_url": "https://c.com",
            },
        }
        for name, profile in profiles.items():
            config_manager.set(f"llm.profiles.{name}", profile, save=False)
        config_manager.set("llm.active_profile", "model_b", save=True)

        with open(config_manager.user_config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        saved_profiles = saved_config.get("llm", {}).get("profiles", {})
        assert len(saved_profiles) == 3
        assert all(name in saved_profiles for name in profiles.keys())

    def test_template_defaults_not_polluting(self, config_manager):
        """模板默认值不会污染用户配置"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        minimal_config = {
            "version": "1.0.0",
            "app": {"theme": "dark", "language": "zh-CN"},
            "llm": {
                "active_profile": "my_model",
                "profiles": {
                    "my_model": {
                        "model": "my/model",
                        "api_key": "my_key",
                        "base_url": "https://my.com",
                    }
                },
            },
            "env": {"MY_VAR": "value"},
            "im": {
                "feishu": {"enabled": False},
                "dingtalk": {"enabled": False},
                "wecom": {"enabled": False},
                "wechat": {"enabled": False},
            },
            "gateway": {"enabled": True},
        }
        config_manager.replace_user_config(minimal_config)

        config_manager.load()
        config_manager.set("app.theme", "light", save=True)

        with open(config_manager.user_config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        assert "default" not in saved_config.get("llm", {}).get("profiles", {})

    def test_sanitize_preserves_user_profiles(self, config_manager):
        """replace_user_config 保留所有用户 profiles"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        test_config = {
            "version": "1.0.0",
            "app": {"theme": "system", "language": "zh-CN"},
            "llm": {
                "active_profile": "custom_model",
                "profiles": {
                    "custom_model": {
                        "model": "custom/model",
                        "api_key": "custom_key",
                        "base_url": "https://custom.com",
                        "custom_field": "should_be_preserved",
                    }
                },
            },
            "env": {"CUSTOM_ENV": "value"},
            "im": {
                "feishu": {"enabled": False},
                "dingtalk": {"enabled": False},
                "wecom": {"enabled": False},
                "wechat": {"enabled": False},
            },
            "gateway": {"enabled": True},
        }
        config_manager.replace_user_config(test_config)

        saved = config_manager.get_raw_user_config()

        assert "llm" in saved
        assert "profiles" in saved["llm"]
        assert "custom_model" in saved["llm"]["profiles"]
        assert saved["llm"]["profiles"]["custom_model"]["custom_field"] == "should_be_preserved"

    def test_config_backup_creation(self, config_manager):
        """配置备份机制正常工作"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set("app.theme", "dark", save=True)
        backup_dir = config_manager.user_config_dir / "backups"
        assert backup_dir.exists()

        config_manager.set("app.theme", "light", save=True)
        backup_files = list(backup_dir.glob("config_backup_*.json"))
        assert len(backup_files) > 0

    def test_concurrent_set_operations(self, config_manager):
        """连续的 set 操作不会丢失数据"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        for i in range(5):
            config_manager.set(
                f"llm.profiles.model_{i}",
                {
                    "model": f"provider/model_{i}",
                    "api_key": f"key_{i}",
                    "base_url": f"https://{i}.com",
                },
                save=False,
            )
        config_manager.save()
        config_manager.load()

        profiles = config_manager._runtime_data.get("llm", {}).get("profiles", {})
        assert len(profiles) == 5

    def test_env_section_preserved(self, config_manager):
        """env 部分完全保留"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        env_vars = {
            "VAR_1": "value1",
            "VAR_2": "value2",
            "GITHUB_TOKEN": "secret_token",
            "TAVILY_API_KEY": "api_key",
        }
        for key, value in env_vars.items():
            config_manager.set(f"env.{key}", value, save=False)
        config_manager.save()

        with open(config_manager.user_config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        saved_env = saved_config.get("env", {})
        for key in env_vars:
            assert key in saved_env
            assert saved_env[key] == env_vars[key]

    def test_im_configuration_preserved(self, config_manager):
        """IM 配置完整保留"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        feishu_config = {
            "enabled": True,
            "app_id": "test_app_id",
            "app_secret": "test_secret",
            "encrypt_key": "test_encrypt",
            "verification_token": "test_token",
        }
        config_manager.set("im.feishu", feishu_config, save=True)

        with open(config_manager.user_config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        saved_feishu = saved_config.get("im", {}).get("feishu", {})
        assert saved_feishu.get("enabled") is True
        assert saved_feishu.get("app_id") == "test_app_id"
        assert saved_feishu.get("app_secret") == "test_secret"

    def test_system_fields_not_written(self, config_manager):
        """系统字段不会被写入用户配置"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # ota.url 是 system-only 字段，应拒绝写入
        with pytest.raises(ValueError, match="系统配置字段不可修改"):
            config_manager.set("ota.url", "https://evil.com", save=False)

        # gateway.mode 现在是 user-managed，会通过 _is_system_readonly
        # 但会在 Pydantic 验证时失败（无效的枚举值）
        with pytest.raises(Exception):
            config_manager.set("gateway.mode", "evil_mode", save=False)

        config_manager.save()

        with open(config_manager.user_config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        # ota.url 不应出现在用户配置
        if "ota" in saved_config:
            assert "url" not in saved_config["ota"]


class TestMergeFunctions:
    """测试配置合并函数的安全性"""

    def test_merge_configs_preserves_user_values(self):
        """merge_configs 保留用户值"""
        template = {
            "app": {"theme": "system", "language": "en-US"},
            "llm": {
                "active_profile": "default",
                "profiles": {"default": {"model": "default/model"}},
            },
            "skills": {"timeout": 300},
        }
        user = {
            "app": {"theme": "dark"},
            "llm": {
                "active_profile": "custom",
                "profiles": {"custom": {"model": "custom/model"}},
            },
        }

        merged = merge_configs(template, user)

        assert merged["app"]["theme"] == "dark"
        assert merged["llm"]["active_profile"] == "custom"
        assert "custom" in merged["llm"]["profiles"]
        assert merged["app"]["language"] == "en-US"
        assert merged["skills"]["timeout"] == 300

    def test_merge_template_defaults_does_not_overwrite(self):
        """merge_template_defaults 不会覆盖用户值"""
        template = {
            "app": {"theme": "system", "language": "en-US"},
            "llm": {
                "active_profile": "default",
                "profiles": {"default": {"model": "default/model"}},
            },
        }
        user = {"app": {"theme": "dark"}}

        merged = merge_template_defaults(template, user)

        assert merged["app"]["theme"] == "dark"
        assert merged["app"]["language"] == "en-US"
