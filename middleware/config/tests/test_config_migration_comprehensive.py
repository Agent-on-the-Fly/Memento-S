# SPDX-License-Identifier: Apache-2.0
"""
配置迁移综合测试

运行方式:
    python -m pytest middleware/config/tests/test_config_migration_comprehensive.py -v
"""

from __future__ import annotations

import json
import pytest

from middleware.config import ConfigManager, SchemaMetadata
from middleware.config.migrations import merge_template_defaults


class TestConfigMigrationComprehensive:
    """配置迁移综合测试套件"""

    def _create_user_config(self, manager: ConfigManager, data: dict):
        manager.ensure_user_config_dir()
        with open(manager.user_config_path, "w") as f:
            json.dump(data, f, indent=2)

    # ---------- 场景 1: 模板增加字段 ----------

    def test_template_adds_new_top_level_field(self, config_manager):
        """模板增加新的顶层字段"""
        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark", "language": "zh-CN"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {},
        }
        self._create_user_config(config_manager, user_config)

        template = config_manager.load_user_template()
        merged = merge_template_defaults(template, user_config)

        assert "gateway" in merged
        assert merged["gateway"]["enabled"] is True

    def test_template_adds_nested_field(self, config_manager):
        """模板在嵌套结构中增加字段"""
        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark"},
            "llm": {"profiles": {}},
            "env": {},
        }
        self._create_user_config(config_manager, user_config)

        template = config_manager.load_user_template()
        merged = merge_template_defaults(template, user_config)

        assert "language" in merged["app"]
        assert "active_profile" in merged["llm"]

    # ---------- 场景 2: 模板减少字段 ----------

    def test_template_removes_field_user_keeps(self, config_manager):
        """模板删除了字段，但用户配置中保留"""
        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark", "custom_user_field": "value"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {},
            "user_custom_section": {"key": "value"},
        }
        self._create_user_config(config_manager, user_config)

        config_manager.load()
        raw_user = config_manager.get_raw_user_config()

        assert "user_custom_section" in raw_user
        assert raw_user["user_custom_section"]["key"] == "value"

    # ---------- 场景 3: 用户配置增加字段 ----------

    def test_user_adds_custom_field(self, config_manager):
        """用户添加自定义字段"""
        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark", "custom_setting": "my_value"},
            "llm": {
                "active_profile": "my_model",
                "profiles": {
                    "my_model": {
                        "model": "custom/model",
                        "api_key": "sk-test",
                        "base_url": "https://api.test.com",
                        "custom_param": "value",
                    }
                },
            },
            "env": {"MY_CUSTOM_VAR": "value"},
            "custom_section": {"key": "value"},
        }
        self._create_user_config(config_manager, user_config)

        config_manager.load()
        raw = config_manager.get_raw_user_config()

        assert raw["app"]["custom_setting"] == "my_value"
        assert "custom_section" in raw

    # ---------- 场景 4: x-managed-by 标记保护 ----------

    def test_user_managed_fields_not_overwritten(self, config_manager):
        """x-managed-by: user 的字段不会被模板覆盖"""
        user_config = {
            "version": "1.0.0",
            "llm": {
                "active_profile": "user_model",
                "profiles": {
                    "user_model": {
                        "model": "user/custom-model",
                        "api_key": "sk-user",
                        "base_url": "https://user.api.com",
                    }
                },
            },
            "env": {"USER_VAR": "value"},
        }
        self._create_user_config(config_manager, user_config)

        config_manager.load()
        raw = config_manager.get_raw_user_config()

        assert "user_model" in raw["llm"]["profiles"]
        assert raw["llm"]["profiles"]["user_model"]["model"] == "user/custom-model"

    def test_env_fully_user_controlled(self, config_manager):
        """env 完全由用户控制"""
        user_config = {
            "version": "1.0.0",
            "env": {"OPENAI_API_KEY": "sk-test", "CUSTOM_VAR": "value"},
        }
        self._create_user_config(config_manager, user_config)

        template = config_manager.load_user_template()
        schema = config_manager.load_schema()
        merged = SchemaMetadata.merge_respecting_metadata(template, user_config, schema)

        assert merged["env"]["OPENAI_API_KEY"] == "sk-test"
        assert merged["env"]["CUSTOM_VAR"] == "value"

    # ---------- 场景 5: 边缘情况 ----------

    def test_empty_user_config(self, config_manager):
        """空用户配置"""
        config_manager.ensure_user_config_file()
        config = config_manager.load()
        assert config.version
        assert config.llm

    def test_deeply_nested_structure(self, config_manager):
        """深层嵌套结构"""
        user_config = {
            "version": "1.0.0",
            "skills": {
                "execution": {
                    "timeout_sec": 600,
                    "nested": {"level3": {"level4": "deep_value"}},
                }
            },
        }
        self._create_user_config(config_manager, user_config)

        template = config_manager.load_user_template()
        merged = merge_template_defaults(template, user_config)

        assert merged["skills"]["execution"]["timeout_sec"] == 600
        assert merged["skills"]["execution"]["nested"]["level3"]["level4"] == "deep_value"

    def test_type_mismatch_handling(self, config_manager):
        """类型不匹配处理"""
        user_config = {
            "version": "1.0.0",
            "app": "should_be_object",
        }
        self._create_user_config(config_manager, user_config)

        with pytest.raises(Exception):
            config_manager.load()

    def test_null_values_preservation(self, config_manager):
        """null 值保留"""
        user_config = {"version": "1.0.0", "env": {"NULL_VAR": None, "EMPTY_VAR": ""}}
        self._create_user_config(config_manager, user_config)

        template = config_manager.load_user_template()
        merged = merge_template_defaults(template, user_config)

        assert merged["env"]["NULL_VAR"] is None
        assert merged["env"]["EMPTY_VAR"] == ""

    def test_array_in_config(self, config_manager):
        """字典类型处理（extra_headers）"""
        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark"},
            "llm": {
                "profiles": {
                    "test": {
                        "model": "test/model",
                        "api_key": "sk-test",
                        "base_url": "https://test.com",
                        "extra_headers": {"Authorization": "Bearer token"},
                    }
                }
            },
            "env": {},
        }
        self._create_user_config(config_manager, user_config)

        config_manager.load()
        raw = config_manager.get_raw_user_config()
        assert isinstance(raw["llm"]["profiles"]["test"]["extra_headers"], dict)
        assert raw["llm"]["profiles"]["test"]["extra_headers"]["Authorization"] == "Bearer token"

    def test_unicode_and_special_chars(self, config_manager):
        """Unicode 和特殊字符"""
        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark", "language": "zh-CN"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {"UNICODE_VAR": "中文测试", "SPECIAL": "!@#$%^&*()"},
        }
        self._create_user_config(config_manager, user_config)

        config_manager.load()
        raw = config_manager.get_raw_user_config()
        assert raw["env"]["UNICODE_VAR"] == "中文测试"
        assert raw["env"]["SPECIAL"] == "!@#$%^&*()"

    # ---------- 场景 6: 配置变更检测 ----------

    def test_change_detection(self, config_manager):
        """配置变更检测"""
        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark", "language": "zh-CN"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {},
        }
        self._create_user_config(config_manager, user_config)

        config_manager.load()
        config_manager.set("app.theme", "light", save=True)

        raw = config_manager.get_raw_user_config()
        assert raw["app"]["theme"] == "light"

    def test_multiple_changes_batch_save(self, config_manager):
        """批量修改后保存"""
        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark", "language": "zh-CN"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {},
        }
        self._create_user_config(config_manager, user_config)

        config_manager.load()
        config_manager.set("llm.active_profile", "model1", save=False)
        config_manager.set("env.BATCH_VAR1", "value1", save=False)
        config_manager.set("env.BATCH_VAR2", "value2", save=False)
        config_manager.save()

        raw = config_manager.get_raw_user_config()
        assert raw["llm"]["active_profile"] == "model1"
        assert raw["env"]["BATCH_VAR1"] == "value1"
        assert raw["env"]["BATCH_VAR2"] == "value2"
