#!/usr/bin/env python3
"""
测试新的存储模型设计

使用方法:
    .venv/bin/python tests/test_storage_models.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.ext.asyncio import AsyncSession

from middleware.storage.core.engine import get_db_manager
from middleware.storage.models import (
    Base,
    Session,
    SessionStatus,
    Message,
    MessageRole,
    MessageType,
    Skill,
    SkillStatus,
    SkillSourceType,
)
from utils.logger import setup_logger


async def test_models():
    """测试新的存储模型"""
    print("=" * 70)
    print("测试新的存储模型")
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
    db_manager.init(db_url=db_url, echo=False)

    # 创建表
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ 数据库表创建成功")

    # 测试数据
    async with db_manager.session_factory() as session:
        print("\n【1. 创建会话】")
        chat_session = Session(
            title="测试会话",
            description="这是一个测试会话",
            status=SessionStatus.ACTIVE.value,
            meta_info={
                "tags": ["test", "demo"],
                "category": "general",
                "priority": "high",
            },
        )
        session.add(chat_session)
        await session.commit()
        await session.refresh(chat_session)
        print(f"  ✓ 会话创建: {chat_session.id}")
        print(f"    标题: {chat_session.title}")
        print(f"    状态: {chat_session.status}")
        print(f"    元数据: {chat_session.meta_info}")

        print("\n【2. 创建消息】")
        msg1 = Message(
            session_id=chat_session.id,
            sequence=1,
            role=MessageRole.USER.value,
            message_type=MessageType.TEXT.value,
            content="你好，请介绍一下自己",
            meta_info={"tokens": 15, "model": "gpt-4"},
        )
        msg2 = Message(
            session_id=chat_session.id,
            sequence=2,
            role=MessageRole.ASSISTANT.value,
            message_type=MessageType.TEXT.value,
            content="你好！我是一个AI助手...",
            meta_info={"tokens": 50, "model": "gpt-4"},
        )
        session.add_all([msg1, msg2])

        # 更新会话统计
        chat_session.message_count = 2
        chat_session.total_tokens = 65
        await session.commit()
        print(f"  ✓ 创建了 2 条消息")

        print("\n【3. 创建技能】")
        test_skill = Skill(
            name="weather_check",
            display_name="天气查询",
            description="查询指定城市的天气信息",
            version="1.0.0",
            author="Memento Team",
            status=SkillStatus.ACTIVE.value,
            source_type=SkillSourceType.BUILTIN.value,
            source_url="https://github.com/...",
            local_path="/skills/weather_check",
            tags=["weather", "utility"],
            category="utility",
            meta_info={"requires_api_key": False},
        )
        session.add(test_skill)
        await session.commit()
        await session.refresh(test_skill)
        print(f"  ✓ 技能创建: {test_skill.name}")
        print(f"    显示名: {test_skill.display_name}")
        print(f"    来源: {test_skill.source_type}")
        print(f"    标签: {test_skill.tags}")

        print("\n【4. 查询验证】")
        # 查询会话及其消息
        result = await session.get(Session, chat_session.id)
        print(f"  ✓ 查询会话: {result.title}")
        print(f"    消息数: {result.message_count}")
        print(f"    总token: {result.total_tokens}")

        # 查询技能
        skill_result = await session.get(Skill, test_skill.id)
        print(f"  ✓ 查询技能: {skill_result.display_name}")

    print("\n" + "=" * 70)
    print("✓ 所有测试通过")
    print("=" * 70)

    print("\n【模型结构总结】")
    print("1. Session - 会话表")
    print("   - 基础信息: id, title, description, status")
    print("   - 元数据: meta_info (JSON)")
    print("   - 统计: message_count, total_tokens")
    print("   - 关联: messages (一对多)")
    print("")
    print("2. Message - 消息表")
    print("   - 基础: id, session_id, sequence, role, content")
    print("   - 类型: message_type (text/tool_call/system/error)")
    print("   - 工具: tool_calls, tool_call_id")
    print("   - 元数据: meta_info (JSON)")
    print("")
    print("3. Skill - 技能表")
    print("   - 基础: id, name, display_name, description, version, author")
    print("   - 来源: source_type, source_url, local_path, checksum")
    print("   - 向量: embedding, embedding_model (用于语义匹配)")
    print("   - 标签: tags, category")
    print("   - 元数据: meta_info (JSON)")


if __name__ == "__main__":
    asyncio.run(test_models())
