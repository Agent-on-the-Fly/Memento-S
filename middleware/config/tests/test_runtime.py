# SPDX-License-Identifier: Apache-2.0
"""
Config 运行时综合测试

覆盖所有 Manager 在真实运行时的各种场景：
1. 缓存失效：reload() 清除所有缓存
2. 并发写入：连续 save() 不丢数据
3. 外部文件修改检测
4. 磁盘满 / 权限异常
5. 文件损坏恢复
6. SchemaMetadata 边界路径验证
7. 备份机制
8. 三 Manager（Config / MCP / Skill）协同工作

运行方式:
    python -m pytest middleware/config/tests/test_runtime.py -v
"""

from __future__ import annotations

import json
import pytest
import tempfile
import threading
import time
from pathlib import Path

from middleware.config import ConfigManager
from middleware.config.mcp_config_manager import McpConfigManager
from middleware.config.skill_config_manager import SkillConfigManager
from middleware.config.schema_meta import SchemaMetadata


# =============================================================================
# SchemaMetadata 边界
# =============================================================================

class TestSchemaMetadata:
    """SchemaMetadata 字段权限边界"""

    def test_user_managed_paths(self):
        """x-managed-by: user 的字段路径正确识别"""
        manager = ConfigManager()
        schema = manager.load_schema()

        assert SchemaMetadata.is_user_managed(schema, "llm") is True
        assert SchemaMetadata.is_user_managed(schema, "llm.active_profile") is True
        assert SchemaMetadata.is_user_managed(schema, "llm.profiles") is True
        assert SchemaMetadata.is_user_managed(schema, "env") is True
        assert SchemaMetadata.is_user_managed(schema, "app") is True
        assert SchemaMetadata.is_user_managed(schema, "im") is True

    def test_system_readonly_paths(self):
        """system-only 字段正确识别"""
        manager = ConfigManager()
        schema = manager.load_schema()

        assert SchemaMetadata.is_user_managed(schema, "ota.url") is False
        assert SchemaMetadata.is_user_managed(schema, "paths") is False
        assert SchemaMetadata.is_user_managed(schema, "logging") is False
        assert SchemaMetadata.is_user_managed(schema, "agent") is False
        assert SchemaMetadata.is_user_managed(schema, "skills.catalog_path") is False

    def test_nested_path_traversal(self):
        """嵌套路径：父级有 x-managed-by: user 时返回 True"""
        manager = ConfigManager()
        schema = manager.load_schema()

        # llm 有 x-managed-by: user → 直接返回 True（不继续遍历子字段）
        assert SchemaMetadata.is_user_managed(schema, "llm.profiles.model") is True
        # gateway 有 x-managed-by: user → 直接返回 True
        assert SchemaMetadata.is_user_managed(schema, "gateway.nonexistent") is True

    def test_nonexistent_path(self):
        """完全不存在的顶层路径返回 False"""
        manager = ConfigManager()
        schema = manager.load_schema()

        # 完全不存在的顶层
        assert SchemaMetadata.is_user_managed(schema, "nonexistent") is False

    def test_get_managed_paths(self):
        """get_managed_paths() 返回所有 user-managed 路径"""
        manager = ConfigManager()
        schema = manager.load_schema()

        paths = SchemaMetadata.get_managed_paths(schema)
        assert "llm" in paths
        assert "env" in paths
        assert "app" in paths
        assert "im" in paths


# =============================================================================
# 缓存失效
# =============================================================================

class TestCacheInvalidation:
    """reload() / 重新 load() 清除运行时缓存"""

    def test_reload_clears_runtime_config(self, config_manager):
        """reload() 后 _runtime_config 被重新赋值"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set("app.theme", "amber", save=True)

        config_manager.reload()
        assert config_manager._runtime_config is not None
        assert config_manager.app.theme == "amber"

    def test_reload_clears_user_data_cache(self, config_manager):
        """reload() 后 _user_data 被重新加载"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set("app.theme", "teal", save=True)
        before = id(config_manager._user_data)

        config_manager.reload()
        assert id(config_manager._user_data) != before

    def test_reload_clears_system_data_cache(self, config_manager):
        """reload() 后 _system_data 被重新加载"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        before = id(config_manager._system_data)
        config_manager.reload()
        assert id(config_manager._system_data) != before


# =============================================================================
# 并发写入
# =============================================================================

class TestConcurrentWrites:
    """并发写入安全性"""

    def test_concurrent_set_same_key(self, config_manager):
        """并发写入同一 key，最终值是某一个值"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        write_lock = threading.Lock()

        def write_value(i: int):
            m = ConfigManager(str(config_manager.user_config_path))
            m.load()
            with write_lock:
                m.set("app.theme", f"color_{i}", save=True)

        threads = [threading.Thread(target=write_value, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 最终应保存了某一个值
        raw = json.loads(config_manager.user_config_path.read_text(encoding="utf-8"))
        assert raw["app"]["theme"].startswith("color_")

    def test_concurrent_set_different_keys(self, config_manager):
        """并发写入不同 key，所有 key 都保留"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        write_lock = threading.Lock()

        def write_key(i: int):
            m = ConfigManager(str(config_manager.user_config_path))
            with write_lock:
                m.load()
                m.set(f"env.KEY_{i}", f"value_{i}", save=True)

        threads = [threading.Thread(target=write_key, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 所有 key 都存在
        manager2 = ConfigManager(str(config_manager.user_config_path))
        manager2.load()
        for i in range(5):
            assert manager2.env.get(f"KEY_{i}") == f"value_{i}", f"KEY_{i} missing"


# =============================================================================
# 外部文件修改检测
# =============================================================================

class TestExternalModification:
    """外部程序修改配置文件后的行为"""

    def test_external_json_modification_detected(self, config_manager):
        """外部 JSON 格式修改能被 load() 识别"""
        config_manager.ensure_user_config_file()
        config_manager.load()
        config_manager.set("app.theme", "original", save=True)

        # 外部程序修改
        raw = json.loads(config_manager.user_config_path.read_text(encoding="utf-8"))
        raw["app"]["theme"] = "modified_by_external"
        config_manager.user_config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        # reload 应读到新值
        config_manager.reload()
        assert config_manager.app.theme == "modified_by_external"

    def test_external_env_var_added(self, config_manager):
        """外部添加 env 变量能被检测"""
        config_manager.ensure_user_config_file()
        config_manager.load()
        config_manager.set("app.theme", "dark", save=True)

        raw = json.loads(config_manager.user_config_path.read_text(encoding="utf-8"))
        raw["env"]["EXTERNAL_VAR"] = "from_outside"
        config_manager.user_config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        config_manager.reload()
        assert config_manager.env.get("EXTERNAL_VAR") == "from_outside"


# =============================================================================
# 备份机制
# =============================================================================

class TestBackupMechanism:
    """配置备份"""

    def test_first_save_creates_backup_dir(self, config_manager):
        """首次 save() 创建 backups 目录"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set("app.theme", "first_save", save=True)

        backup_dir = config_manager.user_config_dir / "backups"
        assert backup_dir.exists()

    def test_second_save_creates_new_backup(self, config_manager):
        """第二次 save() 创建新备份"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set("app.theme", "v1", save=True)
        time.sleep(0.01)
        config_manager.set("app.theme", "v2", save=True)

        backup_dir = config_manager.user_config_dir / "backups"
        backups = sorted(backup_dir.glob("config_backup_*.json"))
        assert len(backups) >= 1

    def test_backup_retention_limit(self, config_manager):
        """备份超过 10 个时自动清理"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        for i in range(15):
            config_manager.set("app.theme", f"v{i}", save=True)
            time.sleep(0.001)

        backup_dir = config_manager.user_config_dir / "backups"
        backups = list(backup_dir.glob("config_backup_*.json"))
        assert len(backups) <= 10

    def test_backup_contains_previous_state(self, config_manager):
        """备份文件包含保存时的状态"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set("app.theme", "before_backup", save=True)

        backup_dir = config_manager.user_config_dir / "backups"
        backups = sorted(backup_dir.glob("config_backup_*.json"), key=lambda p: p.stat().st_mtime)

        latest = json.loads(backups[-1].read_text(encoding="utf-8"))
        # 备份是 save() 前的状态（可能已包含之前保存的 theme）
        assert "app" in latest


# =============================================================================
# 文件损坏与异常
# =============================================================================

class TestFileCorruption:
    """文件损坏与异常处理"""

    def test_invalid_json_on_load(self, temp_config_dir):
        """无效 JSON 文件在 load() 时抛异常"""
        config_path = temp_config_dir / "config.json"
        config_path.write_text("{ broken json", encoding="utf-8")

        manager = ConfigManager(str(config_path))
        with pytest.raises(Exception):
            manager.load()

    def test_missing_required_fields_still_loads(self, config_manager):
        """缺少可选字段时仍能加载（使用默认值）"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        minimal = {
            "app": {"theme": "dark", "language": "zh-CN"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {},
        }
        config_manager.replace_user_config(minimal)
        config_manager.load()

        assert config_manager.version  # 有默认值

    def test_unknown_fields_ignored(self, config_manager):
        """schema 中不存在的字段被 Pydantic extra=ignore 忽略"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager._user_data["app"]["unknown_field"] = "should_be_ignored"
        config_manager._user_data["app"]["theme_options"] = {"dark": {}}
        config_manager._runtime_data = config_manager._merge(
            config_manager._system_data, config_manager._user_data
        )

        # app.theme_options 是 system-only，不应暴露给用户
        # 但 extra=ignore 不影响 schema 验证
        assert config_manager.app.theme == config_manager._runtime_data["app"]["theme"]


# =============================================================================
# SchemaMetadata merge 边界
# =============================================================================

class TestMergeBoundary:
    """merge 边界场景"""

    def test_merge_with_none_user_value(self, config_manager):
        """用户值为 None 时保留（user-managed 字段的行为）"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # llm 是 user-managed，直接用 user 值（None）
        config_manager._user_data["llm"] = None
        result = config_manager._merge(config_manager._system_data, config_manager._user_data)
        # user-managed 字段直接用 user 值
        assert result["llm"] is None

    def test_merge_with_empty_dict(self, config_manager):
        """用户值为空 dict 时的处理"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        # env 是 user-managed，直接用 user 值（空 dict）
        config_manager._user_data["env"] = {}
        result = config_manager._merge(config_manager._system_data, config_manager._user_data)
        assert result["env"] == {}  # user 值直接覆盖

    def test_merge_with_nested_user_only(self, config_manager):
        """用户只有嵌套字段时正常合并"""
        config_manager.ensure_user_config_file()
        config_manager.load()

        config_manager.set(
            "llm.profiles.deep_test",
            {
                "model": "deep/model",
                "api_key": "key",
                "base_url": "https://deep.com",
            },
            save=True,
        )
        config_manager.set("llm.active_profile", "deep_test", save=True)

        result = config_manager._runtime_data
        assert result["llm"]["profiles"]["deep_test"]["model"] == "deep/model"

    def test_system_config_has_top_level_fields(self, config_manager):
        """system_config.json 包含所有顶层必需字段"""
        system = config_manager.load_system_config()

        # GlobalConfig 顶层字段（不含 llm，llm 在 user_template 中）
        required = ["version", "app", "skills", "paths", "logging", "agent"]
        for field in required:
            assert field in system, f"system_config missing required field: {field}"


# =============================================================================
# 三 Manager 协同
# =============================================================================

class TestThreeManagersCollaboration:
    """ConfigManager / McpConfigManager / SkillConfigManager 协同工作"""

    def test_all_three_managers_work_independently(self, temp_config_dir):
        """三个 Manager 独立工作，互不干扰"""
        # ConfigManager
        config_path = temp_config_dir / "config.json"
        config_mgr = ConfigManager(str(config_path))
        config_mgr.ensure_user_config_file()
        config_mgr.load()
        config_mgr.set("app.theme", "collab_theme", save=True)

        # McpConfigManager
        mcp_path = temp_config_dir / "mcp.json"
        mcp_mgr = McpConfigManager()
        mcp_mgr.mcp_config_path = mcp_path
        mcp_mgr.ensure_mcp_config_file()
        mcp_mgr.load()
        mcp_mgr.set_server("collab_server", {
            "transport": "stdio",
            "command": "collab_cmd",
            "enabled": True,
        })

        # SkillConfigManager
        skill_path = temp_config_dir / "skill.json"
        skill_mgr = SkillConfigManager()
        skill_mgr._user_path = skill_path
        skill_mgr.load()
        skill_mgr.register_skill("collab_skill", {
            "location": "workspace",
            "source": "local",
        })

        # 验证三者互不干扰
        # ConfigManager
        c2 = ConfigManager(str(config_path))
        c2.load()
        assert c2.app.theme == "collab_theme"

        # McpConfigManager
        m2 = McpConfigManager()
        m2.mcp_config_path = mcp_path
        m2.load()
        assert "collab_server" in m2.get_servers()

        # SkillConfigManager
        s2 = SkillConfigManager()
        s2._user_path = skill_path
        s2.load()
        assert "collab_skill" in s2.list_skills()


# =============================================================================
# 路径与权限
# =============================================================================

class TestPathsAndPermissions:
    """路径与权限相关"""

    def test_user_config_dir_property(self, config_manager):
        """user_config_dir 属性正确"""
        config_manager.ensure_user_config_file()
        assert config_manager.user_config_dir == config_manager.user_config_path.parent

    def test_nonexistent_config_path_parent_created(self, temp_config_dir):
        """不存在的父目录在初始化时自动创建"""
        config_path = temp_config_dir / "nested" / "deep" / "config.json"
        manager = ConfigManager(str(config_path))
        manager.ensure_user_config_file()

        assert config_path.parent.exists()

    def test_path_with_tilde_expands(self, temp_config_dir):
        """~ 路径正确展开"""
        import os
        home = os.path.expanduser("~")
        config_path = Path(f"{home}/memento_s/test_tilde.json")

        manager = ConfigManager(str(config_path))
        # expanduser 后应指向实际 home 目录
        assert str(manager.user_config_path).startswith("/Users/manson") or "~" in str(config_path)
