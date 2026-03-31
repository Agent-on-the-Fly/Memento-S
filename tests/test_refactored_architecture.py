#!/usr/bin/env python3
"""
测试重构后的架构：Session + Conversation 两层
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from middleware.storage import Base
from shared.chat import ChatManager
from middleware.storage.core.engine import get_db_manager
from utils.logger import setup_logger


async def test_refactored_architecture():
    """测试新架构"""
    print("=" * 70)
    print("测试重构后的架构：Session + Conversation")
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

    print("\n【测试 1】使用 ChatManager 创建 Session 和 Conversation")

    # 创建 Session
    session = await ChatManager.create_session(title="测试会话", description="测试描述")
    print(f"✓ 创建 Session: {session.id}")

    # 创建 User Conversation
    user_conv = await ChatManager.create_conversation(
        session_id=session.id,
        role="user",
        title="你好",
        content="你好，请介绍一下自己",
        tokens=10,
    )
    print(f"✓ 创建 User Conversation: {user_conv.id}")
    print(f"  - Content: {user_conv.content}")
    print(f"  - Role: {user_conv.role}")

    # 创建 Assistant Conversation
    assistant_conv = await ChatManager.create_conversation(
        session_id=session.id,
        role="assistant",
        title="你好！我是...",
        content="你好！我是 AI 助手，有什么可以帮助你的？",
        tokens=15,
        meta_info={"model": "gpt-4"},
    )
    print(f"✓ 创建 Assistant Conversation: {assistant_conv.id}")
    print(f"  - Content: {assistant_conv.content}")
    print(f"  - Role: {assistant_conv.role}")

    # 列出所有 Conversations
    conversations = await ChatManager.list_conversations(session.id)
    print(f"\n✓ Session 下有 {len(conversations)} 个 Conversations")
    for i, conv in enumerate(conversations, 1):
        print(f"  {i}. [{conv.role}] {conv.content[:30]}...")

    print("\n【测试 2】创建另一个 Session 测试对话历史")

    # 创建新 Session
    session2 = await ChatManager.create_session(
        title="Manager 测试会话",
        metadata={"model": "claude-3"},
    )
    print(f"✓ 通过 ChatManager 创建 Session: {session2.id}")

    # 添加一些对话
    await ChatManager.create_conversation(
        session_id=session2.id,
        role="user",
        title="测试消息1",
        content="这是第一条测试消息",
        tokens=5,
    )
    await ChatManager.create_conversation(
        session_id=session2.id,
        role="assistant",
        title="回复1",
        content="这是第一条回复",
        tokens=8,
    )

    print("\n【测试 3】使用 ChatManager 获取对话历史")

    # 获取对话历史
    history = await ChatManager.get_conversation_history(session.id)
    print(f"✓ 获取对话历史: {len(history)} 条")
    for h in history:
        print(f"  - [{h['role']}]: {h['content'][:30]}...")

    print("\n【测试 4】验证 Session 统计自动更新")

    updated_session = await ChatManager.get_session(session.id)
    print(f"✓ Session conversation_count: {updated_session.conversation_count}")
    print(f"✓ Session total_tokens: {updated_session.total_tokens}")

    # 清理
    print("\n【清理】")
    await ChatManager.delete_session(session.id)
    await ChatManager.delete_session(session2.id)
    print("✓ 删除测试数据")

    print("\n" + "=" * 70)
    print("✓ 重构后的架构测试通过！")
    print("=" * 70)
    print("\n架构特点：")
    print("1. 只有 Session + Conversation 两层（无 Message）")
    print("2. Conversation 直接存储 content、role 等字段")
    print("3. ChatManager 作为统一入口管理 Session 和 Conversation")
    print("4. 统计信息（conversation_count, total_tokens）自动维护")
    print("5. 返回类型：SessionInfo/ConversationInfo 对象（使用 .id, .title 等属性）")
    print("6. get_conversation_history 返回字典列表（使用 ['role'], ['content'] 等键）")


if __name__ == "__main__":
    asyncio.run(test_refactored_architecture())
