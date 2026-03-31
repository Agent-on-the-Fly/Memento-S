"""Shared Chat — 统一的 Session 和 Conversation 管理.

提供对 GUI、Agent、CLI、IM 等各层的统一数据访问接口.

所有 Session 和 Conversation 操作都通过 ChatManager 完成：
    from shared.chat import ChatManager

    # Session 操作
    session = await ChatManager.create_session(title="新会话")
    sessions = await ChatManager.list_sessions(limit=20)

    # Conversation 操作
    conv = await ChatManager.create_conversation(
        session_id=session.id,
        role="user",
        title="用户消息",
        content="你好"
    )
    history = await ChatManager.get_conversation_history(session.id)

Note: SessionManager 和 ConversationManager 只在 shared/chat 内部使用，
外部请直接使用 ChatManager 的类方法。
"""

from .chat_manager import ChatManager
from .session_manager import generate_session_id
from .types import SessionInfo, ConversationInfo

__all__ = [
    # 主要入口
    "ChatManager",
    # 数据类型
    "SessionInfo",
    "ConversationInfo",
    # 工具函数
    "generate_session_id",
]
