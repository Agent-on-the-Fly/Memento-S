#!/usr/bin/env python3
"""
测试配置迁移系统

使用方法:
    .venv/bin/python tests/test_config_migration.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from middleware.config.migrations import (
    ConfigMigrator,
    MigrationResult,
    is_newer_version,
    merge_configs,
    merge_template_defaults,
    parse_semver,
)


def test_semver_parsing():
    """测试语义版本解析"""
    print("\n【测试 1: 语义版本解析】")

    # 有效版本
    assert parse_semver("1.2.3") == (1, 2, 3)
    assert parse_semver("0.0.1") == (0, 0, 1)
    assert parse_semver("10.20.30") == (10, 20, 30)
    print("  ✓ 有效版本解析正确")

    # 无效版本
    assert parse_semver("") is None
    assert parse_semver("invalid") is None
    assert parse_semver("1.2.invalid") is None
    print("  ✓ 无效版本返回 None")


def test_version_comparison():
    """测试版本比较"""
    print("\n【测试 2: 版本比较】")

    # 新版本 > 旧版本
    assert is_newer_version("1.1.0", "1.0.0") is True
    assert is_newer_version("2.0.0", "1.9.9") is True
    assert is_newer_version("1.0.1", "1.0.0") is True
    print("  ✓ 新版本检测正确")

    # 相同版本
    assert is_newer_version("1.0.0", "1.0.0") is False
    print("  ✓ 相同版本不触发升级")

    # 旧版本 < 新版本
    assert is_newer_version("1.0.0", "1.1.0") is False
    print("  ✓ 旧版本不触发升级")

    # 无效版本处理
    assert is_newer_version("invalid", "1.0.0") is False
    assert is_newer_version("1.0.0", "invalid") is True
    print("  ✓ 无效版本处理正确")


def test_config_merge():
    """测试配置合并"""
    print("\n【测试 3: 配置合并】")

    template = {
        "version": "1.0.0",
        "app": {"name": "App", "theme": "dark"},
        "new_key": "new_value",
    }

    old = {
        "version": "0.9.0",
        "app": {"name": "MyApp", "language": "zh"},
        "old_key": "old_value",
    }

    merged = merge_configs(template, old)

    # 旧值应该保留
    assert merged["app"]["name"] == "MyApp"
    print("  ✓ 旧值被保留")

    # 模板中的新键应该添加
    assert merged["new_key"] == "new_value"
    print("  ✓ 新键被添加")

    # 旧键不在模板中，但 merge_configs 会保留用户特有的键（如 llm.profiles）
    # 这是设计上的行为，以保留用户的动态配置
    assert "old_key" in merged
    assert merged["old_key"] == "old_value"
    print("  ✓ 旧键被保留（用户特有的键不会被移除）")

    # 嵌套 dict 应该递归合并
    assert merged["app"]["theme"] == "dark"
    assert "language" in merged["app"]  # language 被保留（用户特有的键）
    assert merged["app"]["language"] == "zh"
    print("  ✓ 嵌套 dict 递归合并，保留用户特有的键")


def test_migration():
    """测试完整迁移流程"""
    print("\n【测试 4: 完整迁移流程】")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建旧配置
        config_file = Path(tmpdir) / "config.json"
        old_config = {
            "version": "1.0.0",
            "app": {"name": "MyApp", "theme": "light"},
            "old_feature": True,
        }
        with open(config_file, "w") as f:
            json.dump(old_config, f)

        # 创建模板（新版本）
        template_file = Path(tmpdir) / "template.json"
        template = {
            "version": "1.1.0",
            "app": {"name": "App", "theme": "dark", "language": "en"},
            "new_feature": False,
        }
        with open(template_file, "w") as f:
            json.dump(template, f)

        # 创建 migrator
        migrator = ConfigMigrator(
            config_file=config_file,
            template_file=template_file,
        )

        # 检查是否需要迁移
        needs, old_ver, new_ver = migrator.needs_migration()
        assert needs is True
        assert old_ver == "1.0.0"
        assert new_ver == "1.1.0"
        print("  ✓ 正确检测需要迁移")

        # 执行迁移
        result = migrator.migrate()
        assert result.migrated is True
        assert result.old_version == "1.0.0"
        assert result.new_version == "1.1.0"
        assert result.backup_path is not None
        assert result.backup_path.exists()
        print("  ✓ 迁移成功，创建备份")

        # 验证迁移后的配置
        with open(config_file) as f:
            migrated = json.load(f)

        assert migrated["version"] == "1.1.0"
        assert migrated["app"]["name"] == "MyApp"  # 旧值保留
        assert migrated["app"]["theme"] == "light"  # 旧值保留
        assert migrated["new_feature"] is False  # 新键添加
        # merge_configs 会保留用户特有的键（如 llm.profiles）
        # old_feature 虽然是旧键，但被视为用户特有的键而被保留
        assert "old_feature" in migrated
        assert migrated["old_feature"] is True
        print("  ✓ 配置合并正确（保留用户特有的键）")

        # 再次检查（应该不需要迁移）
        needs2, _, _ = migrator.needs_migration()
        assert needs2 is False
        print("  ✓ 重复检查不需要迁移")


def test_no_migration_needed():
    """测试不需要迁移的情况"""
    print("\n【测试 5: 不需要迁移】")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建当前版本配置
        config_file = Path(tmpdir) / "config.json"
        config = {"version": "1.0.0", "app": {"name": "Test"}}
        with open(config_file, "w") as f:
            json.dump(config, f)

        # 创建相同版本模板
        template_file = Path(tmpdir) / "template.json"
        template = {"version": "1.0.0", "app": {"name": "Test"}}
        with open(template_file, "w") as f:
            json.dump(template, f)

        migrator = ConfigMigrator(
            config_file=config_file,
            template_file=template_file,
        )

        result = migrator.migrate()
        assert result.migrated is False
        print("  ✓ 正确返回不需要迁移")


def test_template_default_merge():
    """测试模板默认值合并（不覆盖用户配置）"""
    print("\n【测试 6: 模板默认值合并】")

    template = {
        "app": {
            "name": "App",
            "theme": "dark",
            "flags": {"beta": False, "new_ui": True},
        },
        "llm": {
            "profiles": {"default": {"model": "gpt"}},
        },
        "new_key": "new_value",
    }

    user = {
        "app": {
            "name": "UserApp",
            "flags": {"beta": True},
        },
        "llm": {
            "profiles": {"custom": {"model": "kimi"}},
        },
        "extra": 123,
    }

    merged = merge_template_defaults(template, user)

    # 用户值不应被覆盖
    assert merged["app"]["name"] == "UserApp"
    assert merged["app"]["flags"]["beta"] is True

    # 模板缺失字段应补全
    assert merged["app"]["theme"] == "dark"
    assert merged["app"]["flags"]["new_ui"] is True

    # 模板新增字段应补全
    assert merged["new_key"] == "new_value"

    # 用户独有字段应保留
    assert merged["extra"] == 123

    # 映射类字段应保留用户 key
    assert "custom" in merged["llm"]["profiles"]
    assert "default" in merged["llm"]["profiles"]

    print("  ✓ 模板默认值合并正确")


if __name__ == "__main__":
    print("=" * 70)
    print("测试配置迁移系统")
    print("=" * 70)

    test_semver_parsing()
    test_version_comparison()
    test_config_merge()
    test_migration()
    test_no_migration_needed()
    test_template_default_merge()

    print("\n" + "=" * 70)
    print("✓ 所有测试通过！")
    print("=" * 70)

    print("\n【新架构说明】")
    print("1. middleware/config/migrations/")
    print("   - migrator.py: 迁移核心逻辑")
    print("   - __init__.py: 模块导出")
    print("")
    print("2. 从 bootstrap.py 分离的好处：")
    print("   - 职责单一：bootstrap 只负责流程，迁移逻辑独立")
    print("   - 可测试：迁移逻辑可以独立单元测试")
    print("   - 可扩展：支持自定义模板加载器")
    print("   - 可复用：其他地方可以直接使用 migrator")
