"""Core managers for Memento-S.

此模块包含仅在 core/ 内部使用的管理器.
Note: SessionManager 和 ConversationManager 已移动到 shared.chat.
"""

from .session_context import ActionRecord, EnvironmentSnapshot, SessionContext

__all__ = [
    # Session 和 Conversation 管理器已移动到 shared.chat
    # 请使用: from shared.chat import ChatManager
    # 仅 core/ 内部使用的执行上下文
    "SessionContext",
    "EnvironmentSnapshot",
    "ActionRecord",
]
