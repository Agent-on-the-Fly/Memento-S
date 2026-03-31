#!/usr/bin/env python3
"""
GUI 集成测试 - 验证新架构

测试内容：
1. Session 管理
2. Conversation 创建和显示（新架构：Conversation 直接存储内容）
3. 侧边栏刷新
4. 消息发送流程

新架构：
- Session: 顶层容器
- Conversation: 直接存储完整消息内容（role + content）
- 不再使用 Message 层
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from middleware.storage import (
    Base,
    SessionService,
    ConversationService,
    SessionCreate,
    ConversationCreate,
)
from middleware.storage.core.engine import get_db_manager
from utils.logger import setup_logger


async def test_gui_integration():
    """模拟 GUI 完整流程"""
    print("=" * 70)
    print("GUI 集成测试 - 新架构（Session -> Conversation）")
    print("=" * 70)

    setup_logger()

    # 初始化数据库
    from middleware.config.config_manager import ConfigManager

    manager = ConfigManager()
    db_path = manager.get_db_path()
    db_url = f"sqlite+aiosqlite:///{db_path}"

    db_manager = get_db_manager()
    await db_manager.init(db_url=db_url, echo=False)
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 初始化服务（模拟 GUI 中的 service）
    session_service = SessionService()
    conversation_service = ConversationService()

    # ========== 场景 1: 应用启动 ==========
    print("\n【场景 1】应用启动")

    # 查找最近的 Session
    sessions = await session_service.list_recent(limit=1)
    if sessions:
        current_session_id = sessions[0].id
        print(f"✓ 恢复 Session: {current_session_id}")
    else:
        # 创建新 Session
        session = await session_service.create(SessionCreate(title="新会话"))
        current_session_id = session.id
        print(f"✓ 创建 Session: {current_session_id}")

    # 加载该 Session 的 Conversations（模拟侧边栏）
    conversations = await conversation_service.list_by_session(current_session_id)
    print(f"✓ 加载 {len(conversations)} 个 Conversations 到侧边栏")

    # ========== 场景 2: 用户发送第一条消息 ==========
    print("\n【场景 2】用户发送第一条消息")

    user_content = "你好，请介绍一下自己"

    # 创建 User Conversation（新架构：直接存储内容）
    user_conv = await conversation_service.create(
        ConversationCreate(
            session_id=current_session_id,
            role="user",
            title=user_content[:30],
            content=user_content,
            meta_info={"timestamp": datetime.now().isoformat()},
        )
    )

    print(f"✓ 创建 User Conversation: {user_conv.id}")

    # 模拟 AI 回复
    ai_content = "你好！我是 AI 助手，有什么可以帮助你的？"

    ai_conv = await conversation_service.create(
        ConversationCreate(
            session_id=current_session_id,
            role="assistant",
            title=ai_content[:30],
            content=ai_content,
            meta_info={
                "timestamp": datetime.now().isoformat(),
                "reply_to": user_conv.id,
            },
        )
    )

    print(f"✓ 创建 AI Conversation: {ai_conv.id}")

    # 刷新侧边栏（模拟）
    conversations = await conversation_service.list_by_session(current_session_id)
    print(f"✓ 侧边栏刷新: 现在显示 {len(conversations)} 个 Conversations")

    # ========== 场景 3: 用户发送第二条消息 ==========
    print("\n【场景 3】用户发送第二条消息（同一会话）")

    user_content2 = "今天天气怎么样？"

    user_conv2 = await conversation_service.create(
        ConversationCreate(
            session_id=current_session_id,
            role="user",
            title=user_content2[:30],
            content=user_content2,
            meta_info={"timestamp": datetime.now().isoformat()},
        )
    )

    ai_content2 = "我无法获取实时天气信息。"

    ai_conv2 = await conversation_service.create(
        ConversationCreate(
            session_id=current_session_id,
            role="assistant",
            title=ai_content2[:30],
            content=ai_content2,
            meta_info={
                "timestamp": datetime.now().isoformat(),
                "reply_to": user_conv2.id,
            },
        )
    )

    print(f"✓ 创建第二个对话轮次的 Conversations")

    # ========== 场景 4: 点击"新对话"按钮 ==========
    print("\n【场景 4】点击'新对话'按钮")

    # 模拟：清空 UI 显示的消息，但保留 Session
    ui_messages = []  # 清空
    print(f"✓ UI 消息已清空")
    print(f"✓ Session 保持不变: {current_session_id}")

    # 用户在新"对话"中发送消息
    user_content3 = "新话题：如何学习 Python？"

    user_conv3 = await conversation_service.create(
        ConversationCreate(
            session_id=current_session_id,
            role="user",
            title=user_content3[:30],
            content=user_content3,
            meta_info={"timestamp": datetime.now().isoformat()},
        )
    )

    print(f"✓ 在新对话中创建 Conversation: {user_conv3.id}")

    # ========== 场景 5: 侧边栏显示所有 Conversations ==========
    print("\n【场景 5】侧边栏显示所有 Conversations")

    all_conversations = await conversation_service.list_by_session(current_session_id)
    print(f"\n侧边栏列表（共 {len(all_conversations)} 个）:")

    for i, conv in enumerate(all_conversations, 1):
        role_icon = "👤" if conv.role == "user" else "🤖"
        print(f"  {i}. {role_icon} {conv.title}")

    # ========== 场景 6: 点击侧边栏 Conversation 加载 ==========
    print("\n【场景 6】点击侧边栏加载特定 Conversation")

    # 加载第一个 user conversation
    target_conv = all_conversations[0]
    # 新架构：直接读取 conversation.content
    print(f"✓ 加载 Conversation: {target_conv.id}")
    print(f"✓ 角色: {target_conv.role}")
    print(f"✓ 内容: {target_conv.content[:50]}...")

    # ========== 验证 ==========
    print("\n【最终验证】")

    stats = {
        "session_id": current_session_id,
        "total_conversations": len(all_conversations),
        "user_conversations": len([c for c in all_conversations if c.role == "user"]),
        "ai_conversations": len(
            [c for c in all_conversations if c.role == "assistant"]
        ),
    }

    print(f"\n统计:")
    print(f"  Session ID: {stats['session_id']}")
    print(f"  总 Conversations: {stats['total_conversations']}")
    print(f"  User Conversations: {stats['user_conversations']}")
    print(f"  AI Conversations: {stats['ai_conversations']}")

    # 验证：User 和 AI 数量应该基本成对（最后一轮可能未完成）
    assert abs(stats["user_conversations"] - stats["ai_conversations"]) <= 1
    assert stats["total_conversations"] == 5, f"应该有 5 个 Conversations"

    print("\n✓ 所有验证通过！")

    # 清理
    print("\n【清理】")
    await session_service.delete(current_session_id)
    print("✓ 删除测试数据")

    print("\n" + "=" * 70)
    print("✓ GUI 集成测试通过！")
    print("=" * 70)
    print("\n新架构验证完成：")
    print("1. Session 持续存在")
    print("2. 每次消息交互创建 2 个 Conversations（user + assistant）")
    print("3. Conversation 直接存储 role 和 content")
    print("4. 侧边栏显示所有 Conversations")
    print("5. '新对话'只清空 UI，不创建新 Session")
    print("6. 不再使用 Message 层")


if __name__ == "__main__":
    asyncio.run(test_gui_integration())
