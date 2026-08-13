# SPDX-License-Identifier: Apache-2.0
"""
Skill Config 运行时读写测试

测试 SkillConfigManager 对 ~/memento_s/skill.json 的运行时管理：
1. 初始化：从模板创建
2. Skill 注册/注销/列表
3. SkillEntry Pydantic 模型 CRUD
4. index.last_sync 更新
5. Schema 验证
6. 运行时 reload

运行方式:
    python -m pytest middleware/config/tests/test_skill_config.py -v
"""

from __future__ import annotations

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone

from middleware.config.skill_config_manager import SkillConfigManager


class TestSkillBootstrap:
    """Skill 配置初始化"""

    def test_no_file_creates_from_template(self, temp_config_dir):
        """skill.json 不存在时从模板创建"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"

        assert not manager._user_path.exists()
        manager.load()
        assert manager._user_path.exists()

        data = manager._data
        assert "version" in data
        assert data["version"] == 1
        assert "skills" in data
        assert data["skills"] == {}

    def test_template_structure(self, temp_config_dir):
        """模板包含正确的结构"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        assert manager._data["version"] == 1
        assert "skills" in manager._data
        assert "index" in manager._data
        assert manager._data["index"]["last_sync"] is None
        assert manager._data["index"]["sync_errors"] == []

    def test_bootstrap_creates_parent_dir(self, temp_config_dir):
        """bootstrap 时创建父目录"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "nested" / "dir" / "skill.json"

        manager.load()
        assert manager._user_path.parent.exists()


class TestSkillCRUD:
    """Skill 增删改"""

    def test_register_single_skill(self, temp_config_dir):
        """注册单个 skill"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        manager.register_skill(
            "test-skill",
            {
                "location": "/tmp/test-skills/test-skill",
                "source": "builtin",
                "version": 1,
                "installed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "status": "active",
                "tags": ["test"],
            },
        )

        skills = manager.list_skills()
        assert "test-skill" in skills
        assert skills["test-skill"]["location"] == "/tmp/test-skills/test-skill"
        assert skills["test-skill"]["status"] == "active"

    def test_register_updates_existing(self, temp_config_dir):
        """注册同名 skill 时更新"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        manager.register_skill(
            "update-test",
            {
                "location": "/tmp/skills/builtin/update-test",
                "source": "builtin",
                "version": 1,
            },
        )
        manager.register_skill(
            "update-test",
            {
                "location": "/tmp/workspace/update-test",
                "source": "local",
                "version": 2,
            },
        )

        skill = manager.get_skill("update-test")
        assert skill["version"] == 2
        assert skill["location"] == "/tmp/workspace/update-test"

    def test_unregister_skill(self, temp_config_dir):
        """注销 skill"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        manager.register_skill("to-remove", {
            "location": "/tmp/builtin/to-remove",
            "source": "builtin",
        })
        manager.unregister_skill("to-remove")

        assert manager.get_skill("to-remove") is None
        assert "to-remove" not in manager.list_skills()

    def test_unregister_nonexistent_no_error(self, temp_config_dir):
        """注销不存在的 skill 不报错"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        manager.unregister_skill("ghost-skill")  # 不应抛异常

    def test_get_nonexistent_returns_none(self, temp_config_dir):
        """获取不存在的 skill 返回 None"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        assert manager.get_skill("does-not-exist") is None

    def test_register_multiple_skills(self, temp_config_dir):
        """注册多个 skill"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        for i in range(10):
            manager.register_skill(f"skill-{i}", {
                "location": "/tmp/skills/skill-" + str(i),
                "source": "local",
                "version": 1,
            })

        skills = manager.list_skills()
        assert len(skills) == 10
        for i in range(10):
            assert f"skill-{i}" in skills

    def test_list_skills_returns_dict(self, temp_config_dir):
        """list_skills() 返回 dict"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        manager.register_skill("dict-test", {
            "location": "/tmp/builtin/dict-test",
            "source": "builtin",
        })

        skills = manager.list_skills()
        assert isinstance(skills, dict)
        assert "dict-test" in skills


class TestSkillPydanticModel:
    """SkillEntry Pydantic 模型"""

    def test_get_skill_model(self, temp_config_dir):
        """get_skill_model() 返回 SkillEntry 模型"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        manager.register_skill("pydantic-test", {
            "location": "/tmp/cloud/pydantic-test",
            "source": "cloud",
            "version": 3,
            "installed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "disabled",
            "tags": ["ai", "test"],
        })

        entry = manager.get_skill_model("pydantic-test")
        assert entry is not None
        assert entry.location == "/tmp/cloud/pydantic-test"  # Literal 是字符串
        assert entry.version == 3
        assert entry.status == "disabled"
        assert "ai" in entry.tags

    def test_register_skill_model(self, temp_config_dir):
        """register_skill_model() 接受 SkillEntry 模型"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        from middleware.config.schemas.skill_config_schemas import SkillEntry

        from datetime import datetime, timezone
        entry = SkillEntry(
            location="/tmp/workspace/model-reg",
            source="local",
            version=5,
            installed_at=datetime.now(timezone.utc).isoformat(),
            status="active",
            tags=["pydantic"],
        )
        manager.register_skill_model("model-reg", entry)

        skill = manager.get_skill("model-reg")
        assert skill["version"] == 5
        assert skill["location"] == "/tmp/workspace/model-reg"

    def test_get_registry_model(self, temp_config_dir):
        """get_registry_model() 返回完整 SkillRegistryConfig"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        manager.register_skill("registry-test", {
            "location": "/tmp/builtin/registry-test",
            "source": "builtin",
            "version": 1,
        })

        registry = manager.get_registry_model()
        assert "registry-test" in registry.skills


class TestSkillIndex:
    """同步 index 管理"""

    def test_update_sync_time(self, temp_config_dir):
        """update_sync_time() 更新 last_sync 时间戳"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        manager.update_sync_time()

        assert manager._data["index"]["last_sync"] is not None
        assert manager._data["index"]["sync_errors"] == []

    def test_update_sync_time_with_errors(self, temp_config_dir):
        """update_sync_time() 记录 sync_errors"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        errors = ["skill-1 failed to load", "skill-2 not found"]
        manager.update_sync_time(errors)

        assert manager._data["index"]["sync_errors"] == errors
        assert manager._data["index"]["last_sync"] is not None

    def test_sync_time_persists(self, temp_config_dir):
        """sync 时间戳保存到磁盘"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        manager.register_skill("persist-test", {
            "location": "/tmp/builtin/persist-test",
            "source": "builtin",
        })
        manager.update_sync_time(["error1"])

        # 重新加载
        manager2 = SkillConfigManager()
        manager2._user_path = temp_config_dir / "skill.json"
        manager2.load()

        assert manager2._data["index"]["sync_errors"] == ["error1"]
        assert manager2._data["index"]["last_sync"] is not None


class TestSkillValidation:
    """Schema 验证"""

    def test_valid_location_accepts_any_path(self, temp_config_dir):
        """location 现在是绝对路径字符串，接受任何有效路径格式"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        # 任意路径字符串都合法（不再有 enum 限制）
        manager.register_skill("any-path", {
            "location": "/absolute/path/to/my-skill",
            "source": "local",
        })
        skill = manager.get_skill("any-path")
        assert skill["location"] == "/absolute/path/to/my-skill"

    def test_invalid_source_rejected(self, temp_config_dir):
        """无效 source 被拒绝"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        with pytest.raises(Exception):
            manager.register_skill("bad-source", {
                "location": "/tmp/builtin/bad-source",
                "source": "invalid_source",
            })

    def test_invalid_status_rejected(self, temp_config_dir):
        """无效 status 被拒绝"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        with pytest.raises(Exception):
            manager.register_skill("bad-status", {
                "location": "/tmp/builtin/bad-status",
                "source": "builtin",
                "status": "invalid_status",
            })

    def test_wrong_version_type_rejected(self, temp_config_dir):
        """version 类型错误被拒绝"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        with pytest.raises(Exception):
            manager.register_skill("bad-version", {
                "location": "/tmp/builtin/bad-version",
                "source": "builtin",
                "version": "not_a_number",  # 应为 integer
            })


class TestSkillRuntimeReload:
    """运行时 reload"""

    def test_manual_reload(self, temp_config_dir):
        """手动调用 load() 重新加载"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        manager.register_skill("reload-test", {
            "location": "/tmp/builtin/reload-test",
            "source": "builtin",
        })

        manager.load()  # 重新加载
        assert "reload-test" in manager.list_skills()

    def test_concurrent_registration(self, temp_config_dir):
        """连续注册不丢数据"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        for i in range(20):
            manager.register_skill(f"concurrent-{i}", {
                "location": "/tmp/workspace/concurrent-" + str(i),
                "source": "local",
            })

        skills = manager.list_skills()
        assert len(skills) == 20


class TestSkillEdgeCases:
    """边界场景"""

    def test_empty_tags_list(self, temp_config_dir):
        """tags 可以为空列表"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        manager.register_skill("no-tags", {
            "location": "/tmp/builtin/no-tags",
            "source": "builtin",
            "tags": [],
        })

        skill = manager.get_skill("no-tags")
        assert skill["tags"] == []

    def test_special_chars_in_name(self, temp_config_dir):
        """skill 名称含特殊字符"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        name = "my-awesome_skill.v2"
        manager.register_skill(name, {
            "location": "/tmp/workspace/my-awesome_skill.v2",
            "source": "local",
        })

        assert manager.get_skill(name) is not None

    def test_save_with_custom_data(self, temp_config_dir):
        """save(data=...) 直接保存自定义数据"""
        manager = SkillConfigManager()
        manager._user_path = temp_config_dir / "skill.json"
        manager.load()

        custom_data = {
            "version": 1,
            "skills": {
                "custom-save": {
                    "location": "/tmp/cloud/custom-save",
                    "source": "cloud",
                    "version": 99,
                }
            },
            "index": {
                "last_sync": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "sync_errors": [],
            },
        }
        manager.save(custom_data)

        # 重新加载验证
        manager2 = SkillConfigManager()
        manager2._user_path = temp_config_dir / "skill.json"
        manager2.load()

        assert "custom-save" in manager2.list_skills()
        assert manager2._data["skills"]["custom-save"]["version"] == 99
