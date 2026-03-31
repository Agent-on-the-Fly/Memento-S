#!/usr/bin/env python3
"""
数据库迁移测试

演示如何使用 Alembic 进行数据库迁移管理：
1. 创建迁移脚本（revision）
2. 应用迁移（upgrade）
3. 查看迁移历史（history）
4. 回滚迁移（downgrade）
5. 检查当前版本（current）

使用方法:
    # 1. 首先创建迁移脚本
    .venv/bin/python tests/test_db_migration.py --create "add user table"

    # 2. 应用所有迁移
    .venv/bin/python tests/test_db_migration.py --upgrade

    # 3. 查看迁移历史
    .venv/bin/python tests/test_db_migration.py --history

    # 4. 查看当前版本
    .venv/bin/python tests/test_db_migration.py --current

    # 5. 回滚一个版本
    .venv/bin/python tests/test_db_migration.py --downgrade

    # 6. 运行完整测试
    .venv/bin/python tests/test_db_migration.py --test
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alembic import command
from alembic.config import Config
from middleware.config.config_manager import ConfigManager
from utils.logger import setup_logger


def get_alembic_config(db_url: str | None = None) -> Config:
    """获取 Alembic 配置。

    Args:
        db_url: 数据库 URL，如果为 None 则从配置管理器获取

    Returns:
        Alembic Config 对象
    """
    root = Path(__file__).parent.parent
    alembic_ini = root / "middleware" / "storage" / "migrations" / "alembic.ini"
    script_location = root / "middleware" / "storage" / "migrations"

    if not alembic_ini.exists():
        raise FileNotFoundError(f"alembic.ini not found: {alembic_ini}")

    if db_url is None:
        manager = ConfigManager()
        db_url = manager.get_db_url()

    alembic_cfg = Config(str(alembic_ini))
    alembic_cfg.set_main_option("script_location", str(script_location))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    return alembic_cfg


def create_migration(message: str, db_url: str | None = None) -> None:
    """创建新的迁移脚本。

    Args:
        message: 迁移描述信息
        db_url: 数据库 URL
    """
    print(f"\n【创建迁移脚本】")
    print(f"描述: {message}")

    alembic_cfg = get_alembic_config(db_url)

    # 创建自动迁移（基于模型变化）
    command.revision(
        alembic_cfg,
        autogenerate=True,
        message=message,
    )

    print("✓ 迁移脚本创建成功")
    print("\n提示: 请检查生成的迁移脚本，确保变更正确")


def upgrade_migration(db_url: str | None = None, revision: str = "head") -> None:
    """应用迁移。

    Args:
        db_url: 数据库 URL
        revision: 目标版本，默认 "head"（最新）
    """
    print(f"\n【应用迁移】")
    print(f"目标版本: {revision}")

    alembic_cfg = get_alembic_config(db_url)

    # 应用迁移
    command.upgrade(alembic_cfg, revision)

    print(f"✓ 迁移应用成功，已升级到 {revision}")


def downgrade_migration(db_url: str | None = None, revision: str = "-1") -> None:
    """回滚迁移。

    Args:
        db_url: 数据库 URL
        revision: 目标版本，默认 "-1"（回滚一个版本）
    """
    print(f"\n【回滚迁移】")
    print(f"目标版本: {revision}")

    alembic_cfg = get_alembic_config(db_url)

    # 回滚迁移
    command.downgrade(alembic_cfg, revision)

    print(f"✓ 回滚成功，已降级到 {revision}")


def show_history(db_url: str | None = None) -> None:
    """显示迁移历史。"""
    print(f"\n【迁移历史】")

    alembic_cfg = get_alembic_config(db_url)

    # 显示历史
    command.history(alembic_cfg, verbose=True)


def show_current(db_url: str | None = None) -> None:
    """显示当前版本。"""
    print(f"\n【当前版本】")

    alembic_cfg = get_alembic_config(db_url)

    # 显示当前版本
    command.current(alembic_cfg, verbose=True)


def show_branches(db_url: str | None = None) -> None:
    """显示所有分支。"""
    print(f"\n【分支信息】")

    alembic_cfg = get_alembic_config(db_url)

    # 显示分支
    command.branches(alembic_cfg, verbose=True)


def stamp_version(revision: str, db_url: str | None = None) -> None:
    """标记数据库版本（不执行迁移）。

    用于将已有数据库纳入 alembic 管理。

    Args:
        revision: 版本号
        db_url: 数据库 URL
    """
    print(f"\n【标记版本】")
    print(f"版本: {revision}")

    alembic_cfg = get_alembic_config(db_url)

    # 标记版本
    command.stamp(alembic_cfg, revision)

    print(f"✓ 已标记为版本 {revision}")


def test_migration_workflow() -> None:
    """测试完整的迁移流程。"""
    print("=" * 70)
    print("测试数据库迁移工作流")
    print("=" * 70)

    setup_logger()

    manager = ConfigManager()
    db_url = manager.get_db_url()

    print(f"\n数据库 URL: {db_url}")

    # 1. 检查当前版本
    print("\n【步骤 1】检查当前版本")
    try:
        show_current(db_url)
    except Exception as e:
        print(f"  未找到版本信息（可能是新数据库）: {e}")

    # 2. 查看历史
    print("\n【步骤 2】查看迁移历史")
    try:
        show_history(db_url)
    except Exception as e:
        print(f"  无迁移历史: {e}")

    # 3. 检查是否有未应用的迁移
    print("\n【步骤 3】检查未应用的迁移")
    # 这里可以通过比较数据库版本和最新脚本来检查

    # 4. 应用迁移
    print("\n【步骤 4】应用所有迁移")
    try:
        upgrade_migration(db_url, "head")
    except Exception as e:
        print(f"  应用迁移失败: {e}")
        print("  提示: 如果是新数据库，可能需要先 stamp 初始版本")

    # 5. 验证当前版本
    print("\n【步骤 5】验证当前版本")
    show_current(db_url)

    print("\n" + "=" * 70)
    print("✓ 迁移测试完成")
    print("=" * 70)

    print("\n【建议】")
    print("1. 开发环境: 使用 autogenerate 自动生成迁移脚本")
    print("2. 生产环境: 手动编写迁移脚本并仔细审查")
    print("3. 迁移前备份: 始终备份数据库后再执行迁移")
    print("4. 测试迁移: 在测试环境验证后再应用到生产")


def main():
    parser = argparse.ArgumentParser(description="数据库迁移工具")
    parser.add_argument(
        "--create",
        metavar="MESSAGE",
        help="创建新的迁移脚本（基于模型自动检测变更）",
    )
    parser.add_argument(
        "--upgrade",
        nargs="?",
        const="head",
        metavar="REVISION",
        help="应用迁移（默认到 head）",
    )
    parser.add_argument(
        "--downgrade",
        nargs="?",
        const="-1",
        metavar="REVISION",
        help="回滚迁移（默认回滚一个版本）",
    )
    parser.add_argument("--history", action="store_true", help="显示迁移历史")
    parser.add_argument("--current", action="store_true", help="显示当前版本")
    parser.add_argument("--branches", action="store_true", help="显示分支信息")
    parser.add_argument(
        "--stamp",
        metavar="REVISION",
        help="标记数据库版本（不执行 SQL）",
    )
    parser.add_argument("--test", action="store_true", help="运行完整测试")
    parser.add_argument(
        "--db-url",
        help="指定数据库 URL（默认从配置读取）",
    )

    args = parser.parse_args()

    if args.test:
        test_migration_workflow()
    elif args.create:
        create_migration(args.create, args.db_url)
    elif args.upgrade is not None:
        upgrade_migration(args.db_url, args.upgrade)
    elif args.downgrade is not None:
        downgrade_migration(args.db_url, args.downgrade)
    elif args.history:
        show_history(args.db_url)
    elif args.current:
        show_current(args.db_url)
    elif args.branches:
        show_branches(args.db_url)
    elif args.stamp:
        stamp_version(args.stamp, args.db_url)
    else:
        parser.print_help()
        print("\n【示例】")
        print("  1. 创建迁移脚本:")
        print(
            "     .venv/bin/python tests/test_db_migration.py --create 'add user table'"
        )
        print("\n  2. 应用所有迁移:")
        print("     .venv/bin/python tests/test_db_migration.py --upgrade")
        print("\n  3. 回滚一个版本:")
        print("     .venv/bin/python tests/test_db_migration.py --downgrade")
        print("\n  4. 运行测试:")
        print("     .venv/bin/python tests/test_db_migration.py --test")


if __name__ == "__main__":
    main()
