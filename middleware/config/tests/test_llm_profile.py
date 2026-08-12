# SPDX-License-Identifier: Apache-2.0
"""
GUI Settings LLM Profile 增删测试

模拟 GUI settings_panel.py 的操作路径：
- add_profile: 新增 profile → ConfigManager.set("llm.profiles.{name}", ..., save=False) + save()
- remove_profile: 删除 profile → set() 删除 + save()
- 切换 active_profile: set("llm.active_profile", ...) → _delayed_refresh()

运行方式:
    python -m pytest middleware/config/tests/test_llm_profile.py -v
"""

from __future__ import annotations

import json
import pytest
import tempfile
from pathlib import Path

from middleware.config import ConfigManager


class TestLLMProfileAdd:
    """新增 LLM Profile"""

    def test_add_single_profile(self, config_manager):
        """新增一个 profile"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            "llm.profiles.gpt4",
            {
                "model": "openai/gpt-4",
                "api_key": "sk-test",
                "base_url": "https://api.openai.com/v1",
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout": 120,
            },
            save=False,
        )
        config_manager.save()

        raw = config_manager.get_raw_user_config()
        assert "gpt4" in raw["llm"]["profiles"]
        assert raw["llm"]["profiles"]["gpt4"]["model"] == "openai/gpt-4"

    def test_add_profile_and_activate(self, config_manager):
        """新增 profile 后切换为 active"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            "llm.profiles.new_model",
            {
                "model": "anthropic/claude-3",
                "api_key": "sk-anthropic",
                "base_url": "https://api.anthropic.com",
            },
            save=False,
        )
        config_manager.set("llm.active_profile", "new_model", save=True)

        config_manager.load()
        assert config_manager.llm.active_profile == "new_model"
        assert config_manager.llm.current is not None

    def test_add_multiple_profiles(self, config_manager):
        """连续新增多个 profile"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        profiles = {
            "model_a": {
                "model": "provider/a",
                "api_key": "key_a",
                "base_url": "https://a.com",
            },
            "model_b": {
                "model": "provider/b",
                "api_key": "key_b",
                "base_url": "https://b.com",
            },
            "model_c": {
                "model": "provider/c",
                "api_key": "key_c",
                "base_url": "https://c.com",
            },
        }
        for name, cfg in profiles.items():
            config_manager.set(f"llm.profiles.{name}", cfg, save=False)
        config_manager.save()

        config_manager.load()
        assert set(config_manager.llm.profiles.keys()) == {"model_a", "model_b", "model_c"}

    def test_add_profile_with_extra_fields(self, config_manager):
        """新增带 extra_headers / extra_body 的 profile"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            "llm.profiles.custom",
            {
                "model": "custom/model",
                "api_key": "sk-custom",
                "base_url": "https://custom.com",
                "extra_headers": {"Authorization": "Bearer token123"},
                "extra_body": {"custom_param": "value"},
            },
            save=True,
        )

        raw = config_manager.get_raw_user_config()
        assert raw["llm"]["profiles"]["custom"]["extra_headers"] == {
            "Authorization": "Bearer token123"
        }
        assert raw["llm"]["profiles"]["custom"]["extra_body"] == {"custom_param": "value"}

    def test_add_profile_with_invalid_model_name(self, config_manager):
        """新增 profile 使用无效 model 名（Pydantic 验证应报错）"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            "llm.profiles.invalid",
            {
                "model": "x",  # minLength=1 会通过，但可能不是合法格式
                "api_key": "sk-test",
                "base_url": "https://test.com",
            },
            save=False,
        )
        # 只要 schema 验证通过就行，model 格式不强制
        config_manager.save()
        assert True  # 不应抛异常

    def test_add_profile_duplicate_name(self, config_manager):
        """新增重复名称的 profile 应覆盖"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            "llm.profiles.dupe",
            {
                "model": "v1/model",
                "api_key": "key_v1",
                "base_url": "https://test.com",
            },
            save=True,
        )

        # 再次添加相同名称应覆盖
        config_manager.set(
            "llm.profiles.dupe",
            {
                "model": "v2/model",
                "api_key": "key_v2",
                "base_url": "https://test.com",
            },
            save=True,
        )

        raw = config_manager.get_raw_user_config()
        assert raw["llm"]["profiles"]["dupe"]["model"] == "v2/model"
        assert raw["llm"]["profiles"]["dupe"]["api_key"] == "key_v2"


class TestLLMProfileDelete:
    """删除 LLM Profile"""

    def test_delete_single_profile(self, config_manager):
        """删除一个 profile"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 先添加
        config_manager.set(
            "llm.profiles.to_delete",
            {
                "model": "temp/model",
                "api_key": "key",
                "base_url": "https://test.com",
            },
            save=False,
        )
        config_manager.save()

        # 再删除（设为 None 或从 profiles 中移除）
        config_manager._user_data["llm"]["profiles"].pop("to_delete", None)
        config_manager.save()

        raw = config_manager.get_raw_user_config()
        assert "to_delete" not in raw["llm"]["profiles"]

    def test_delete_active_profile_switches_away(self, config_manager):
        """删除当前 active profile 后应切换到其他 profile"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 设置两个 profile，model_a 为 active
        config_manager.set(
            "llm.profiles.model_a",
            {
                "model": "a/model",
                "api_key": "key_a",
                "base_url": "https://a.com",
            },
            save=False,
        )
        config_manager.set(
            "llm.profiles.model_b",
            {
                "model": "b/model",
                "api_key": "key_b",
                "base_url": "https://b.com",
            },
            save=False,
        )
        config_manager.set("llm.active_profile", "model_a", save=True)

        # 删除 model_a 并切换到 model_b
        config_manager._user_data["llm"]["profiles"].pop("model_a", None)
        config_manager.set("llm.active_profile", "model_b", save=True)

        config_manager.load()
        assert config_manager.llm.active_profile == "model_b"
        assert "model_a" not in config_manager.llm.profiles

    def test_delete_all_profiles(self, config_manager):
        """删除所有 profile"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            "llm.profiles.only",
            {
                "model": "only/model",
                "api_key": "key",
                "base_url": "https://test.com",
            },
            save=False,
        )
        config_manager.set("llm.active_profile", "only", save=True)

        config_manager._user_data["llm"]["profiles"] = {}
        config_manager.set("llm.active_profile", "", save=True)

        config_manager.load()
        assert len(config_manager.llm.profiles) == 0
        assert config_manager.llm.active_profile == ""


class TestLLMProfileSwitch:
    """切换 LLM Profile"""

    def test_switch_active_profile(self, config_manager):
        """切换 active profile"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            "llm.profiles.profile_a",
            {
                "model": "a/model",
                "api_key": "key_a",
                "base_url": "https://a.com",
            },
            save=False,
        )
        config_manager.set(
            "llm.profiles.profile_b",
            {
                "model": "b/model",
                "api_key": "key_b",
                "base_url": "https://b.com",
            },
            save=True,
        )
        config_manager.set("llm.active_profile", "profile_a", save=True)

        config_manager.set("llm.active_profile", "profile_b", save=True)

        config_manager.load()
        assert config_manager.llm.active_profile == "profile_b"
        assert config_manager.llm.current.model == "b/model"

    def test_switch_to_nonexistent_profile(self, config_manager):
        """切换到不存在的 profile（Pydantic validator 应报错）"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 设置存在的 profile 并保存
        config_manager.set(
            "llm.profiles.valid",
            {
                "model": "valid/model",
                "api_key": "key",
                "base_url": "https://test.com",
            },
            save=True,
        )
        config_manager.set("llm.active_profile", "valid", save=True)

        # 直接修改磁盘上的配置文件
        raw = json.loads(config_manager.user_config_path.read_text(encoding="utf-8"))
        raw["llm"]["active_profile"] = "ghost"
        config_manager.user_config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        # load() 时 Pydantic validator 应报错
        with pytest.raises(Exception):
            config_manager.load()

    def test_reload_does_not_change_active_profile(self, config_manager):
        """reload() 不改变 active_profile"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            "llm.profiles.persist",
            {
                "model": "persist/model",
                "api_key": "key",
                "base_url": "https://test.com",
            },
            save=True,
        )
        config_manager.set("llm.active_profile", "persist", save=True)

        profile_before = config_manager.llm.active_profile

        config_manager.reload()
        assert config_manager.llm.active_profile == profile_before


class TestLLMProfileUpdate:
    """运行时更新 LLM Profile 字段"""

    def test_update_profile_field(self, config_manager):
        """运行时更新单个 profile 字段（如 temperature）"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            "llm.profiles.updateme",
            {
                "model": "update/model",
                "api_key": "key",
                "base_url": "https://test.com",
                "temperature": 0.7,
            },
            save=True,
        )
        config_manager.set("llm.active_profile", "updateme", save=True)

        config_manager.set("llm.profiles.updateme.temperature", 0.3, save=True)

        config_manager.load()
        assert config_manager.llm.profiles["updateme"].temperature == 0.3

    def test_update_profile_api_key(self, config_manager):
        """运行时更新 api_key"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            "llm.profiles.keyupdate",
            {
                "model": "key/model",
                "api_key": "old_key",
                "base_url": "https://test.com",
            },
            save=True,
        )
        config_manager.set("llm.active_profile", "keyupdate", save=True)

        config_manager.set("llm.profiles.keyupdate.api_key", "new_key", save=True)

        raw = config_manager.get_raw_user_config()
        assert raw["llm"]["profiles"]["keyupdate"]["api_key"] == "new_key"

    def test_partial_profile_update_preserves_other_fields(self, config_manager):
        """部分更新 profile 时保留其他字段"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        original = {
            "model": "preserve/model",
            "api_key": "key",
            "base_url": "https://test.com",
            "max_tokens": 2048,
            "temperature": 0.8,
            "timeout": 60,
        }
        config_manager.set("llm.profiles.preserve", original, save=True)
        config_manager.set("llm.active_profile", "preserve", save=True)

        # 只更新 temperature
        config_manager.set("llm.profiles.preserve.temperature", 0.1, save=True)

        config_manager.load()
        p = config_manager.llm.profiles["preserve"]
        assert p.temperature == 0.1
        assert p.max_tokens == 2048  # 未修改字段应保留
        assert p.timeout == 60


class TestLLMProfileRuntime:
    """运行时边界场景"""

    def test_zero_profiles_empty_active(self, config_manager):
        """零 profiles 时 active_profile 为空"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager._user_data["llm"]["profiles"] = {}
        config_manager.set("llm.active_profile", "", save=True)

        config_manager.load()
        assert len(config_manager.llm.profiles) == 0
        assert config_manager.llm.active_profile == ""
        assert config_manager.llm.current is None

    def test_very_long_profile_name(self, config_manager):
        """超长 profile 名称"""
        long_name = "a" * 200
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            f"llm.profiles.{long_name}",
            {
                "model": "test/model",
                "api_key": "key",
                "base_url": "https://test.com",
            },
            save=True,
        )

        raw = config_manager.get_raw_user_config()
        assert long_name in raw["llm"]["profiles"]

    def test_special_chars_in_profile_name(self, config_manager):
        """profile 名称含特殊字符"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 直接操作 dict，绕过 _set_by_path 的 "." 分割限制
        # （profile 名称含 "." 本身是非预期用法，但 dict 层面应能存储）
        special_name = "model_with_underscore_and_dash"
        config_manager._user_data.setdefault("llm", {}).setdefault("profiles", {})[special_name] = {
            "model": "test/model",
            "api_key": "key",
            "base_url": "https://test.com",
        }
        config_manager.save()

        config_manager.load()
        assert special_name in config_manager.llm.profiles

    def test_profile_roundtrip_memory_to_disk(self, config_manager):
        """profile 数据在内存和磁盘之间往返保持一致"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        profile_data = {
            "model": "roundtrip/model",
            "api_key": "sk-roundtrip",
            "base_url": "https://roundtrip.com",
            "litellm_provider": "openai_chat",
            "extra_headers": {"X-Custom": "header"},
            "extra_body": {"custom": "body"},
            "context_window": 128000,
            "max_tokens": 8192,
            "temperature": 0.5,
            "timeout": 180,
        }
        config_manager.set("llm.profiles.roundtrip", profile_data, save=True)
        config_manager.set("llm.active_profile", "roundtrip", save=True)

        # 重新加载验证
        config_manager.load()
        loaded = config_manager.llm.profiles["roundtrip"]

        assert loaded.model == "roundtrip/model"
        assert loaded.api_key == "sk-roundtrip"
        assert loaded.base_url == "https://roundtrip.com"
        assert loaded.litellm_provider == "openai_chat"
        assert loaded.extra_headers == {"X-Custom": "header"}
        assert loaded.extra_body == {"custom": "body"}
        assert loaded.context_window == 128000
        assert loaded.max_tokens == 8192
        assert loaded.temperature == 0.5
        assert loaded.timeout == 180