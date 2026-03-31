"""
配置迁移综合测试 - 覆盖各种边界情况

测试场景：
1. 模板增加字段
2. 模板减少字段
3. 用户配置增加字段
4. 嵌套结构变化
5. 类型变化
6. 空配置处理
7. x-managed-by 标记字段保护
"""

import json
import tempfile
from pathlib import Path
import pytest
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from middleware.config import ConfigManager, SchemaMetadata
from middleware.config.migrations import merge_template_defaults


class TestConfigMigrationComprehensive:
    """配置迁移综合测试套件"""

    @pytest.fixture
    def temp_config_dir(self):
        """创建临时配置目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """创建配置管理器实例"""
        config_path = temp_config_dir / "config.json"
        return ConfigManager(str(config_path))

    def _create_user_config(self, config_manager: ConfigManager, data: dict):
        """创建用户配置文件"""
        config_manager.ensure_user_config_dir()
        with open(config_manager.user_config_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_user_config(self, config_manager: ConfigManager) -> dict:
        """加载用户配置文件"""
        with open(config_manager.user_config_path, "r") as f:
            return json.load(f)

    # ========== 场景 1: 模板增加字段 ==========

    def test_template_adds_new_top_level_field(self, config_manager):
        """测试: 模板增加新的顶层字段"""
        print("\n【场景 1.1】模板增加新的顶层字段")

        # 用户配置（旧）
        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark", "language": "zh-CN"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {},
        }
        self._create_user_config(config_manager, user_config)

        # 模拟模板合并（添加 gateway 字段）
        template = config_manager.load_user_template()
        merged = merge_template_defaults(template, user_config)

        # 验证：新字段被添加
        assert "gateway" in merged, "模板新增的 gateway 字段应该被添加"
        assert merged["gateway"]["enabled"] == True, "gateway.enabled 应该有默认值"
        print("  ✓ 新顶层字段 gateway 被正确添加")

    def test_template_adds_nested_field(self, config_manager):
        """测试: 模板在嵌套结构中增加字段"""
        print("\n【场景 1.2】模板在嵌套结构中增加字段")

        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark"},  # 缺少 language
            "llm": {"profiles": {}},  # 缺少 active_profile
            "env": {},
        }
        self._create_user_config(config_manager, user_config)

        template = config_manager.load_user_template()
        merged = merge_template_defaults(template, user_config)

        # 验证：嵌套字段被补充
        assert "language" in merged["app"], "app.language 应该被补充"
        assert "active_profile" in merged["llm"], "llm.active_profile 应该被补充"
        print("  ✓ 嵌套字段被正确补充")

    # ========== 场景 2: 模板减少字段 ==========

    def test_template_removes_field_user_keeps(self, config_manager):
        """测试: 模板删除了字段，但用户配置中保留"""
        print("\n【场景 2.1】模板删除字段，用户保留旧字段")

        # 用户配置（包含自定义字段，模板中可能不存在）
        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark", "custom_user_field": "value"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {},
            "user_custom_section": {"key": "value"},  # 用户自定义章节
        }
        self._create_user_config(config_manager, user_config)

        # 加载配置
        config = config_manager.load()

        # 重新读取用户配置
        raw_user = config_manager.get_raw_user_config()

        # 用户配置应该保留自定义字段
        assert "user_custom_section" in raw_user, "用户自定义章节应该保留"
        assert raw_user["user_custom_section"]["key"] == "value"
        print("  ✓ 用户自定义字段被保留")

    # ========== 场景 3: 用户配置增加字段 ==========

    def test_user_adds_custom_field(self, config_manager):
        """测试: 用户添加自定义字段"""
        print("\n【场景 3.1】用户添加自定义字段")

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
                        "custom_param": "value",  # 自定义参数
                    }
                },
            },
            "env": {"MY_CUSTOM_VAR": "value"},
            "custom_section": {"key": "value"},  # 自定义章节
        }
        self._create_user_config(config_manager, user_config)

        # 加载配置
        config = config_manager.load()

        # 验证：自定义字段保留
        raw = config_manager.get_raw_user_config()
        assert raw["app"]["custom_setting"] == "my_value", "自定义字段应保留"
        assert "custom_section" in raw, "自定义章节应保留"
        print("  ✓ 用户自定义字段和章节被保留")

    # ========== 场景 4: x-managed-by 标记保护 ==========

    def test_user_managed_fields_not_overwritten(self, config_manager):
        """测试: x-managed-by: user 的字段不会被模板覆盖"""
        print("\n【场景 4.1】用户控制字段不会被模板覆盖")

        # 用户配置（已有自定义模型）
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

        # 加载配置
        config = config_manager.load()

        # 重新读取用户配置（应该保留原样）
        raw = config_manager.get_raw_user_config()

        # 验证：用户的模型保留
        assert "user_model" in raw["llm"]["profiles"], "用户模型应保留"
        assert raw["llm"]["profiles"]["user_model"]["model"] == "user/custom-model"
        print("  ✓ 用户控制字段未被模板覆盖")

    def test_env_fully_user_controlled(self, config_manager):
        """测试: env 完全由用户控制"""
        print("\n【场景 4.2】env 章节完全由用户控制")

        user_config = {
            "version": "1.0.0",
            "env": {"OPENAI_API_KEY": "sk-test", "CUSTOM_VAR": "value"},
        }
        self._create_user_config(config_manager, user_config)

        template = config_manager.load_user_template()
        schema = config_manager.load_schema()

        # 使用 SchemaMetadata 合并
        merged = SchemaMetadata.merge_respecting_metadata(template, user_config, schema)

        # env 应该完全使用用户值
        assert merged["env"]["OPENAI_API_KEY"] == "sk-test", "env 应完全用户控制"
        assert merged["env"]["CUSTOM_VAR"] == "value", "自定义环境变量应保留"
        print("  ✓ env 章节完全用户控制")

    # ========== 场景 5: 边缘情况 ==========

    def test_empty_user_config(self, config_manager):
        """测试: 空用户配置"""
        print("\n【场景 5.1】空用户配置")

        # 创建空配置（但包含必需的最小结构）
        config_manager.ensure_user_config_file()

        # 应该能正常加载（使用模板默认值）
        config = config_manager.load()
        assert config.version, "应该有版本号"
        assert config.llm, "应该有 llm 配置"
        print("  ✓ 空配置能正常加载并使用默认值")

    def test_deeply_nested_structure(self, config_manager):
        """测试: 深层嵌套结构"""
        print("\n【场景 5.2】深层嵌套结构")

        user_config = {
            "version": "1.0.0",
            "skills": {
                "execution": {
                    "timeout_sec": 600,  # 用户自定义
                    "nested": {"level3": {"level4": "deep_value"}},
                }
            },
        }
        self._create_user_config(config_manager, user_config)

        template = config_manager.load_user_template()
        merged = merge_template_defaults(template, user_config)

        # 验证：深层嵌套保留
        assert merged["skills"]["execution"]["timeout_sec"] == 600
        assert (
            merged["skills"]["execution"]["nested"]["level3"]["level4"] == "deep_value"
        )
        print("  ✓ 深层嵌套结构正确处理")

    def test_type_mismatch_handling(self, config_manager):
        """测试: 类型不匹配处理"""
        print("\n【场景 5.3】类型不匹配")

        # 用户配置中字段类型错误
        user_config = {
            "version": "1.0.0",
            "app": "should_be_object",  # 错误：应该是对象
        }
        self._create_user_config(config_manager, user_config)

        # 应该抛出验证错误
        with pytest.raises(Exception) as exc_info:
            config_manager.load()

        print(f"  ✓ 类型错误被正确捕获: {exc_info.value}")

    def test_null_values_preservation(self, config_manager):
        """测试: null 值保留"""
        print("\n【场景 5.4】null 值保留")

        user_config = {"version": "1.0.0", "env": {"NULL_VAR": None, "EMPTY_VAR": ""}}
        self._create_user_config(config_manager, user_config)

        template = config_manager.load_user_template()
        merged = merge_template_defaults(template, user_config)

        # null 值应该保留
        assert merged["env"]["NULL_VAR"] is None, "null 值应保留"
        assert merged["env"]["EMPTY_VAR"] == "", "空字符串应保留"
        print("  ✓ null 和空字符串值被保留")

    def test_array_in_config(self, config_manager):
        """测试: 数组类型处理（extra_headers 是字典不是数组）"""
        print("\n【场景 5.5】字典类型处理")

        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark"},
            "llm": {
                "profiles": {
                    "test": {
                        "model": "test/model",
                        "api_key": "sk-test",
                        "base_url": "https://test.com",
                        "extra_headers": {"Authorization": "Bearer token"},  # 字典
                    }
                }
            },
            "env": {},
        }
        self._create_user_config(config_manager, user_config)

        config = config_manager.load()
        # 验证字典被保留
        raw = config_manager.get_raw_user_config()
        assert isinstance(raw["llm"]["profiles"]["test"]["extra_headers"], dict)
        assert (
            raw["llm"]["profiles"]["test"]["extra_headers"]["Authorization"]
            == "Bearer token"
        )
        print("  ✓ 字典类型正确处理")

    def test_unicode_and_special_chars(self, config_manager):
        """测试: Unicode 和特殊字符"""
        print("\n【场景 5.6】Unicode 和特殊字符")

        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark", "language": "zh-CN"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {"UNICODE_VAR": "中文测试 🎉", "SPECIAL": "!@#$%^&*()"},
        }
        self._create_user_config(config_manager, user_config)

        config = config_manager.load()
        raw = config_manager.get_raw_user_config()
        assert raw["env"]["UNICODE_VAR"] == "中文测试 🎉"
        assert raw["env"]["SPECIAL"] == "!@#$%^&*()"
        print("  ✓ Unicode 和特殊字符正确处理")

    # ========== 场景 6: 配置变更检测 ==========

    def test_change_detection(self, config_manager):
        """测试: 配置变更检测"""
        print("\n【场景 6.1】配置变更检测")

        # 初始配置（完整结构）
        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark", "language": "zh-CN"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {},
        }
        self._create_user_config(config_manager, user_config)

        # 修改配置
        config_manager.load()
        config_manager.set("app.theme", "light", save=True)

        # 验证变更
        raw = config_manager.get_raw_user_config()
        assert raw["app"]["theme"] == "light", "变更应被保存"
        print("  ✓ 配置变更被正确检测和保存")

    def test_multiple_changes_batch_save(self, config_manager):
        """测试: 批量修改后保存"""
        print("\n【场景 6.2】批量修改后保存")

        user_config = {
            "version": "1.0.0",
            "app": {"theme": "dark", "language": "zh-CN"},
            "llm": {"active_profile": "", "profiles": {}},
            "env": {},
        }
        self._create_user_config(config_manager, user_config)

        config_manager.load()

        # 批量修改（不保存）
        config_manager.set("llm.active_profile", "model1", save=False)
        config_manager.set("env.BATCH_VAR1", "value1", save=False)
        config_manager.set("env.BATCH_VAR2", "value2", save=False)

        # 手动保存
        config_manager.save()

        raw = config_manager.get_raw_user_config()
        assert raw["llm"]["active_profile"] == "model1"
        assert raw["env"]["BATCH_VAR1"] == "value1"
        assert raw["env"]["BATCH_VAR2"] == "value2"
        print("  ✓ 批量修改后保存成功")


def run_migration_tests():
    """运行所有迁移测试并输出报告"""
    print("=" * 80)
    print("配置迁移综合测试报告")
    print("=" * 80)

    import subprocess

    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    print("=" * 80)
    print(f"返回码: {result.returncode}")
    print("=" * 80)


if __name__ == "__main__":
    run_migration_tests()
