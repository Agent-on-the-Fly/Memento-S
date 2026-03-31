"""
配置安全性测试 - 验证 config.json 不会被错误覆盖

测试场景:
1. set() + save() 不会丢失用户配置
2. 嵌套结构（llm.profiles）不会被清除
3. 模板默认值不会污染用户配置
4. 并发/快速操作不会导致配置损坏
5. 配置备份机制正常工作
"""

import json
import tempfile
import shutil
from pathlib import Path
import pytest
from typing import Any

from middleware.config.config_manager import ConfigManager
from middleware.config.migrations import merge_configs, merge_template_defaults


class TestConfigSafety:
    """配置安全性测试套件"""

    @pytest.fixture
    def temp_config_dir(self):
        """创建临时配置目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """创建配置管理器实例"""
        config_path = Path(temp_config_dir) / "config.json"
        manager = ConfigManager(config_path=str(config_path))
        return manager

    def test_save_does_not_reread_file(self, config_manager, temp_config_dir):
        """测试: save() 时不重新读取文件，避免覆盖"""
        # 初始化配置
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 添加一个模型配置
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

        # 模拟外部修改配置文件
        config_path = Path(temp_config_dir) / "config.json"
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

        # 在外部保存前修改文件
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(external_config, f, indent=2)

        # 保存（应该使用内存中的配置，而不是重新读取文件）
        config_manager.save()

        # 验证：保存的配置应该包含 test_model，而不是 external_model
        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        assert "test_model" in saved_config.get("llm", {}).get("profiles", {}), (
            "内存中的 test_model 应该被保存，而不是被外部修改覆盖"
        )

    def test_nested_profiles_not_cleared(self, config_manager, temp_config_dir):
        """测试: 嵌套的 llm.profiles 不会被清除"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 添加多个模型
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

        # 验证所有模型都被保存
        config_path = Path(temp_config_dir) / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        saved_profiles = saved_config.get("llm", {}).get("profiles", {})
        assert len(saved_profiles) == 3, (
            f"应该保存 3 个模型，实际保存 {len(saved_profiles)} 个"
        )
        assert all(name in saved_profiles for name in profiles.keys()), (
            "所有模型都应该存在"
        )

    def test_template_defaults_not_polluting(self, config_manager, temp_config_dir):
        """测试: 模板默认值不会污染用户配置"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 用户只设置一个简单的配置
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

        # 重新加载并保存
        config_manager.load()
        config_manager.set("app.theme", "light", save=True)

        # 验证：不应该添加模板中的其他默认值
        with open(config_manager.user_config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        # 不应该自动添加模板中的其他字段
        assert "default" not in saved_config.get("llm", {}).get("profiles", {}), (
            "不应该自动添加模板的默认模型"
        )

    def test_sanitize_preserves_user_profiles(self, config_manager):
        """测试: _sanitize_user_config 保留所有用户 profiles"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 创建包含多个 profiles 的配置
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
                        "custom_field": "should_be_preserved",  # 自定义字段
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

        # 新架构：物理隔离，无需清洗
        # 直接保存配置，验证物理隔离是否有效
        config_manager.replace_user_config(test_config)

        # 读取保存的配置（应该只包含 User 配置）
        saved = config_manager.get_raw_user_config()

        # 验证所有 profiles 都被保留
        assert "llm" in saved
        assert "profiles" in saved["llm"]
        assert "custom_model" in saved["llm"]["profiles"]
        assert (
            saved["llm"]["profiles"]["custom_model"]["custom_field"]
            == "should_be_preserved"
        )

    def test_config_backup_creation(self, config_manager, temp_config_dir):
        """测试: 配置备份机制正常工作"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 保存一次以创建初始配置
        config_manager.set("app.theme", "dark", save=True)

        # 验证备份目录创建
        backup_dir = Path(temp_config_dir) / "backups"
        assert backup_dir.exists(), "备份目录应该被创建"

        # 再次保存，应该创建备份
        config_manager.set("app.theme", "light", save=True)

        # 验证备份文件存在
        backup_files = list(backup_dir.glob("config_backup_*.json"))
        assert len(backup_files) > 0, "应该创建备份文件"

    def test_concurrent_set_operations(self, config_manager, temp_config_dir):
        """测试: 连续的 set 操作不会丢失数据"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 快速连续的 set 操作
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

        # 最后保存一次
        config_manager.save()

        # 验证所有模型都被保存
        config_manager.load()
        profiles = config_manager._runtime_data.get("llm", {}).get("profiles", {})
        assert len(profiles) == 5, f"应该保存 5 个模型，实际有 {len(profiles)} 个"

    def test_env_section_preserved(self, config_manager, temp_config_dir):
        """测试: env 部分完全保留"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 设置多个环境变量
        env_vars = {
            "VAR_1": "value1",
            "VAR_2": "value2",
            "GITHUB_TOKEN": "secret_token",
            "TAVILY_API_KEY": "api_key",
        }

        for key, value in env_vars.items():
            config_manager.set(f"env.{key}", value, save=False)

        config_manager.save()

        # 验证所有环境变量都被保存
        with open(config_manager.user_config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        saved_env = saved_config.get("env", {})
        for key in env_vars:
            assert key in saved_env, f"环境变量 {key} 应该被保存"
            assert saved_env[key] == env_vars[key], f"环境变量 {key} 的值应该正确"

    def test_im_configuration_preserved(self, config_manager, temp_config_dir):
        """测试: IM 配置完整保留"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 设置飞书配置
        feishu_config = {
            "enabled": True,
            "app_id": "test_app_id",
            "app_secret": "test_secret",
            "encrypt_key": "test_encrypt",
            "verification_token": "test_token",
        }

        config_manager.set("im.feishu", feishu_config, save=True)

        # 验证飞书配置完整保留
        with open(config_manager.user_config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        saved_feishu = saved_config.get("im", {}).get("feishu", {})
        assert saved_feishu.get("enabled") is True
        assert saved_feishu.get("app_id") == "test_app_id"
        assert saved_feishu.get("app_secret") == "test_secret"

    def test_system_fields_not_written(self, config_manager, temp_config_dir):
        """测试: 系统字段不会被写入用户配置"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 尝试设置系统字段（应该抛出异常）
        with pytest.raises(ValueError, match="系统配置字段不可修改"):
            config_manager.set("ota.url", "https://evil.com", save=False)

        with pytest.raises(ValueError, match="系统配置字段不可修改"):
            config_manager.set("gateway.mode", "evil_mode", save=False)

        config_manager.save()

        # 验证系统字段没有被写入
        with open(config_manager.user_config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        # ota.url 不应该出现在用户配置中
        if "ota" in saved_config:
            assert "url" not in saved_config["ota"], "ota.url 不应该被写入用户配置"

        # gateway 的 system-only 字段不应该出现
        if "gateway" in saved_config:
            gateway = saved_config["gateway"]
            assert "mode" not in gateway, "gateway.mode 不应该被写入"
            assert "websocket_host" not in gateway, (
                "gateway.websocket_host 不应该被写入"
            )


class TestMergeFunctions:
    """测试配置合并函数的安全性"""

    def test_merge_configs_preserves_user_values(self):
        """测试: merge_configs 保留用户值"""
        template = {
            "app": {"theme": "system", "language": "en-US"},
            "llm": {
                "active_profile": "default",
                "profiles": {"default": {"model": "default/model"}},
            },
            "skills": {"timeout": 300},
        }

        user = {
            "app": {"theme": "dark"},  # 覆盖 theme
            "llm": {
                "active_profile": "custom",
                "profiles": {"custom": {"model": "custom/model"}},
            },
        }

        merged = merge_configs(template, user)

        # 用户值应该被保留
        assert merged["app"]["theme"] == "dark"
        assert merged["llm"]["active_profile"] == "custom"
        assert "custom" in merged["llm"]["profiles"]

        # 模板中未覆盖的值应该保留
        assert merged["app"]["language"] == "en-US"
        assert merged["skills"]["timeout"] == 300

    def test_merge_template_defaults_does_not_overwrite(self):
        """测试: merge_template_defaults 不会覆盖用户值"""
        template = {
            "app": {"theme": "system", "language": "en-US"},
            "llm": {
                "active_profile": "default",
                "profiles": {"default": {"model": "default/model"}},
            },
        }

        user = {"app": {"theme": "dark"}}

        merged = merge_template_defaults(template, user)

        # 用户值应该保留
        assert merged["app"]["theme"] == "dark"
        # 模板默认值应该被添加
        assert merged["app"]["language"] == "en-US"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
