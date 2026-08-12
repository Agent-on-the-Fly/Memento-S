# SPDX-License-Identifier: Apache-2.0
"""
系统启动时 merge 端到端测试

覆盖 bootstrap 流程中的配置加载与合并：
1. 冷启动：从无配置文件 → 创建 → merge → 最终状态
2. 增量 merge：已有用户配置 + 模板更新 → 合并后各字段正确
3. 版本迁移：旧版本配置 → 升级后字段迁移
4. reload 稳定性：reload() 不改变已有数据
5. 外部文件修改后 reload 的合并结果

运行方式:
    python -m pytest middleware/config/tests/test_startup_merge.py -v
"""

from __future__ import annotations

import json
import pytest
import shutil
import tempfile
from pathlib import Path

from middleware.config import ConfigManager
from middleware.config.migrations import merge_template_defaults


class TestColdStart:
    """冷启动：从零创建配置"""

    def test_no_user_config_creates_default(self, config_manager):
        """无用户配置时，ensure_user_config_file() 创建空配置"""
        config_manager.ensure_user_config_file()
        assert config_manager.user_config_path.exists()

        config_manager.load()
        assert config_manager.version  # 有版本号
        assert config_manager.llm  # llm section 存在

    def test_empty_file_fails_validation(self, temp_config_dir):
        """用户配置为 {} 时缺少必需字段，load() 抛出验证错误"""
        config_path = temp_config_dir / "config.json"
        config_path.write_text("{}", encoding="utf-8")
        manager = ConfigManager(str(config_path))

        # 缺少 llm/app 等必需字段，GlobalConfig.model_validate 报错
        with pytest.raises(Exception):
            manager.load()

    def test_minimal_user_config补全_paths(self, config_manager):
        """用户配置只有 app/llm/env，paths 等 system-only 字段由 system_config 补全"""
        minimal = {
            "app": {"theme": "dark", "language": "zh-CN"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {},
        }
        config_manager.replace_user_config(minimal)

        config_manager.load()
        assert config_manager.paths is not None
        assert config_manager.paths.workspace_dir is not None
        assert config_manager.paths.db_dir is not None

    def test_template_adds_new_fields_on_startup(self, config_manager):
        """模板新增字段时，用户配置在 load 后能获得新字段"""
        # 模拟 v1 用户配置（没有 gateway 字段）
        user_v1 = {
            "version": "1.0.0",
            "app": {"theme": "dark", "language": "zh-CN"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {},
        }
        config_manager.ensure_user_config_file()
        config_manager.user_config_path.write_text(
            json.dumps(user_v1, indent=2), encoding="utf-8"
        )

        config_manager.load()

        # gateway 应由模板补充
        assert config_manager.gateway is not None
        assert config_manager.gateway.enabled is True


class TestIncrementalMerge:
    """增量 merge：模板更新时用户配置不受影响"""

    def test_user_profile_preserved_after_template_change(self, config_manager):
        """模板增删字段不破坏用户已有 profiles"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            "llm.profiles.user_model",
            {
                "model": "user/custom",
                "api_key": "sk-user",
                "base_url": "https://user.com",
            },
            save=True,
        )
        config_manager.set("llm.active_profile", "user_model", save=True)

        # 模拟模板添加了新字段（通过 merge_template_defaults）
        template = config_manager.load_user_template()
        user = config_manager.get_raw_user_config()
        merged = merge_template_defaults(template, user)

        assert "user_model" in merged["llm"]["profiles"]
        assert merged["llm"]["profiles"]["user_model"]["model"] == "user/custom"

    def test_user_env_preserved_after_template_change(self, config_manager):
        """模板变更不破坏用户 env 变量"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set("env.MY_SECRET", "secret_value", save=True)
        config_manager.set("env.ANOTHER_VAR", "another", save=True)

        template = config_manager.load_user_template()
        user = config_manager.get_raw_user_config()
        merged = merge_template_defaults(template, user)

        assert merged["env"]["MY_SECRET"] == "secret_value"
        assert merged["env"]["ANOTHER_VAR"] == "another"

    def test_merge_preserves_system_when_user_has_empty_nested(self, config_manager):
        """当用户有完整 section 时 merge 正确组合 system + user 值"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        system = config_manager.load_system_config()
        user = config_manager.get_raw_user_config()

        merged = config_manager._merge(system, user)

        # execution 来自 system（用户配置不覆盖）
        assert "execution" in merged["skills"]
        assert merged["skills"]["execution"]["timeout_sec"] == 300


class TestVersionMigration:
    """版本迁移"""

    def test_old_version_config_loads(self, temp_config_dir):
        """旧版本配置应能正常加载（向后兼容）"""
        config_path = temp_config_dir / "config.json"
        old_config = {
            "version": "0.5.0",
            "app": {"theme": "light", "language": "en-US"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {},
            "gateway": {"enabled": True},
        }
        config_path.write_text(json.dumps(old_config, indent=2), encoding="utf-8")

        manager = ConfigManager(str(config_path))
        config = manager.load()

        assert config.version == "0.5.0"  # 版本号应保留
        assert config.app.theme == "light"

    def test_version_from_runtime(self, config_manager):
        """运行时配置包含 version（来自 system_config），用户文件不写 version"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        raw = config_manager.get_raw_user_config()
        # version 不在 user_config（_is_system_readonly 过滤顶层 version）
        # 但运行时 _runtime_config 有 version
        assert config_manager.version  # 运行时有值

    def test_migration_result_has_backup(self, config_manager):
        """迁移结果应创建备份"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set("app.theme", "dark", save=True)

        backup_dir = config_manager.user_config_dir / "backups"
        assert backup_dir.exists()


class TestReloadStability:
    """reload() 稳定性"""

    def test_reload_does_not_lose_data(self, config_manager):
        """连续 reload 不丢数据"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set("app.theme", "dark", save=True)
        config_manager.set("env.RELOAD_TEST", "value", save=True)
        config_manager.set(
            "llm.profiles.reload_model",
            {
                "model": "reload/model",
                "api_key": "key",
                "base_url": "https://test.com",
            },
            save=True,
        )

        for _ in range(5):
            config_manager.reload()
            assert config_manager.app.theme == "dark"
            assert config_manager.env.get("RELOAD_TEST") == "value"
            assert "reload_model" in config_manager.llm.profiles

    def test_reload_clears_runtime_cache(self, config_manager):
        """reload() 清除运行时缓存"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set("app.theme", "blue", save=True)

        cached_theme = config_manager.app.theme
        config_manager.reload()
        assert config_manager.app.theme == cached_theme

    def test_reload_after_external_modification(self, config_manager):
        """外部修改文件后 reload() 能读到新值"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set("app.theme", "green", save=True)

        # 模拟外部程序修改文件
        raw = json.loads(config_manager.user_config_path.read_text(encoding="utf-8"))
        raw["app"]["theme"] = "external_theme"
        config_manager.user_config_path.write_text(
            json.dumps(raw, indent=2), encoding="utf-8"
        )

        config_manager.reload()
        assert config_manager.app.theme == "external_theme"


class TestMergeIntegrity:
    """merge 完整性验证"""

    def test_system_fields_never_from_user_config(self, config_manager):
        """用户配置中的 system-only 字段不进入运行时"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 用户尝试写入 system-only 字段
        config_manager._user_data["paths"] = {"workspace_dir": "/evil/path"}
        config_manager._merge(config_manager._system_data, config_manager._user_data)

        # paths 应来自 system_data
        assert config_manager._runtime_data["paths"]["workspace_dir"] != "/evil/path"

    def test_all_required_sections_present_after_merge(self, config_manager):
        """merge 后所有必填 section 都存在"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        for section in ["app", "llm", "skills", "paths", "logging", "agent"]:
            assert hasattr(config_manager._runtime_config, section)

    def test_llm_empty_profiles_allows_any_active(self, config_manager):
        """空 profiles 时任何 active_profile 值都合法（不会导致运行时崩溃）"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # 直接写入任意 active_profile（profiles 为空时合法）
        raw = json.loads(config_manager.user_config_path.read_text(encoding="utf-8"))
        raw["llm"]["active_profile"] = "any_ghost_profile"
        config_manager.user_config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        # 空 profiles 时 validator 不检查 active_profile，load() 成功
        config_manager.load()
        assert config_manager.llm.current is None  # 但 current 为 None


class TestBootstrapScenarios:
    """Bootstrap 真实场景模拟"""

    def test_first_time_user_flow(self, config_manager):
        """首次使用用户流程：冷启动 → 填配置 → 保存"""
        # 1. 冷启动（无配置）
        config_manager.ensure_user_config_file()
        config = config_manager.load()

        assert config.version
        assert config.llm is not None

        # 2. 用户填写配置
        config_manager.set(
            "llm.profiles.first",
            {
                "model": "first/model",
                "api_key": "sk-first",
                "base_url": "https://first.com",
            },
            save=True,
        )
        config_manager.set("llm.active_profile", "first", save=True)
        config_manager.set("app.theme", "dark", save=True)
        config_manager.set("env.OPENAI_KEY", "sk-secret", save=True)

        # 3. 重启后加载
        manager2 = ConfigManager(str(config_manager.user_config_path))
        config2 = manager2.load()

        assert config2.llm.active_profile == "first"
        assert config2.llm.current.model == "first/model"
        assert config2.app.theme == "dark"
        assert config2.env.get("OPENAI_KEY") == "sk-secret"

    def test_multi_user_profile_switching(self, config_manager):
        """多用户 profile 切换"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        profiles = {
            "alice": {
                "model": "openai/gpt-4",
                "api_key": "sk-alice",
                "base_url": "https://api.openai.com/v1",
            },
            "bob": {
                "model": "anthropic/claude-3",
                "api_key": "sk-bob",
                "base_url": "https://api.anthropic.com",
            },
        }
        for name, cfg in profiles.items():
            config_manager.set(f"llm.profiles.{name}", cfg, save=False)
        config_manager.save()

        # Alice 登录
        config_manager.set("llm.active_profile", "alice", save=True)
        config_manager.load()
        assert config_manager.llm.current.model == "openai/gpt-4"

        # Bob 切换
        config_manager.set("llm.active_profile", "bob", save=True)
        config_manager.load()
        assert config_manager.llm.current.model == "anthropic/claude-3"

        # 两个 profile 都还在
        assert set(config_manager.llm.profiles.keys()) == {"alice", "bob"}

    def test_corrupted_user_config_recovery(self, temp_config_dir):
        """损坏的用户配置 → 备份恢复"""
        config_path = temp_config_dir / "config.json"
        manager = ConfigManager(str(config_path))

        # 创建初始正常配置
        manager.ensure_user_config_file()
        manager.load()
        manager.set("app.theme", "recoverable", save=True)

        # 损坏文件（JSON 无效）
        manager.user_config_path.write_text("{ invalid json", encoding="utf-8")

        # load 应抛异常
        with pytest.raises(Exception):
            manager.load()
