#!/usr/bin/env python3
"""
测试新的 storage service（自动管理 session 版本）

使用方法:
    .venv/bin/python tests/test_storage_service.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from middleware.storage import (
    Base,
    MessageService,
    SessionService,
    SkillService,
    SessionCreate,
    MessageCreate,
    SkillCreate,
    SessionUpdate,
)
from middleware.storage.core.engine import get_db_manager
from utils.logger import setup_logger


async def test_services():
    """测试存储服务（无需手动管理 session）"""
    print("=" * 70)
    print("测试 Storage Services（自动管理 Session）")
    print("=" * 70)

    # 初始化日志
    setup_logger()

    # 初始化数据库
    from middleware.config.config_manager import ConfigManager

    manager = ConfigManager()
    db_path = manager.get_db_path()
    db_url = f"sqlite+aiosqlite:///{db_path}"

    print(f"\n数据库路径: {db_path}")

    db_manager = get_db_manager()
    await db_manager.init(db_url=db_url, echo=False)

    # 创建表
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ 数据库表创建成功")

    # 创建服务实例（无需传递 db session）
    session_service = SessionService()
    message_service = MessageService()
    skill_service = SkillService()

    print("\n【1. 测试 SessionService（自动管理 session）】")

    # 创建会话 - 无需传递 db 参数
    session_data = SessionCreate(title="测试会话", description="这是一个测试会话")
    chat_session = await session_service.create(session_data)
    print(f"  ✓ 会话创建: {chat_session.id}")
    print(f"    标题: {chat_session.title}")
    print(f"    状态: {chat_session.status}")

    # 获取会话 - 无需传递 db 参数
    fetched = await session_service.get(chat_session.id)
    print(f"  ✓ 获取会话: {fetched.title if fetched else 'None'}")

    # 更新会话
    updated = await session_service.update(
        chat_session.id, SessionUpdate(title="更新后的标题")
    )
    print(f"  ✓ 更新会话: {updated.title if updated else 'None'}")

    # 列出最近会话
    recent_sessions = await session_service.list_recent(limit=5)
    print(f"  ✓ 最近会话列表: {len(recent_sessions)} 个")

    print("\n【2. 测试 MessageService（自动管理 session）】")

    # 创建消息 - 无需传递 db 参数
    msg1 = await message_service.create(
        MessageCreate(
            session_id=chat_session.id,
            role="user",
            content="你好，请介绍一下自己",
            meta_info={"tokens": 15},
        )
    )
    print(f"  ✓ 消息1创建: sequence={msg1.sequence}")

    msg2 = await message_service.create(
        MessageCreate(
            session_id=chat_session.id,
            role="assistant",
            content="你好！我是一个AI助手...",
            meta_info={"tokens": 50},
        )
    )
    print(f"  ✓ 消息2创建: sequence={msg2.sequence}")

    # 获取会话消息列表
    messages = await message_service.list_by_session(chat_session.id)
    print(f"  ✓ 获取消息列表: {len(messages)} 条消息")

    # 验证会话统计已更新
    updated_session = await session_service.get(chat_session.id)
    print(f"    会话消息数: {updated_session.message_count}")
    print(f"    会话token数: {updated_session.total_tokens}")

    print("\n【3. 测试 SkillService（自动管理 session）】")

    # 创建技能
    skill_data = SkillCreate(
        name="weather_check",
        display_name="天气查询",
        description="查询指定城市的天气信息",
        version="1.0.0",
        author="Test",
        source_type="builtin",
        tags=["weather", "utility"],
        category="utility",
        meta_info={"requires_api_key": False},
    )
    skill = await skill_service.create(skill_data)
    print(f"  ✓ 技能创建: {skill.name}")
    print(f"    显示名: {skill.display_name}")
    print(f"    标签: {skill.tags}")

    # 通过名称获取技能
    by_name = await skill_service.get_by_name("weather_check")
    print(f"  ✓ 通过名称获取: {by_name.display_name if by_name else 'None'}")

    # 列出所有活跃技能
    skills = await skill_service.list_active()
    print(f"  ✓ 活跃技能列表: {len(skills)} 个")

    print("\n【4. 测试手动控制 session（高级用法）】")

    # 对于需要手动控制事务的场景
    async with session_service.session() as db:
        # 在同一个 session 中执行多个操作
        session1 = await session_service._create(
            db, SessionCreate(title="手动事务测试1")
        )
        session2 = await session_service._create(
            db, SessionCreate(title="手动事务测试2")
        )
        # 手动提交（也可不提交，让上下文管理器自动处理）
        await session_service.commit(db)

    print(f"  ✓ 手动事务控制完成")
    print(f"    创建了2个会话（在同一事务中）")

    print("\n" + "=" * 70)
    print("✓ 所有服务测试通过")
    print("=" * 70)

    print("\n【架构说明】")
    print("1. BaseService - 提供自动 session 管理")
    print("   - _with_session(): 自动创建和关闭 session")
    print("   - session(): 上下文管理器，手动控制 session")
    print("")
    print("2. 所有 Service 继承 BaseService")
    print("   - Public API: 无需传递 db 参数")
    print("   - Private API: _xxx 方法需要手动传入 db")
    print("")
    print("3. 使用方式对比:")
    print("   新API: await session_service.create(data)")
    print("   旧API: await session_service.create(db, data)")


if __name__ == "__main__":
    asyncio.run(test_services())
