"""Chat Manager — 统一的 Chat 管理入口.

提供类方法级别的 Session 和 Conversation 操作，是 GUI、Agent、CLI、IM 的统一入口。
"""

from __future__ import annotations

from typing import Any

from middleware.storage.core.engine import DatabaseManager
from middleware.llm.embedding_client import EmbeddingClient
from middleware.storage.vector_storage import VectorStorage

from .session_manager import SessionManager, generate_session_id
from .conversation_manager import ConversationManager
from .types import SessionInfo, ConversationInfo


class ChatManager:
    """统一的 Chat 管理器.

    使用单例模式管理 SessionManager 和 ConversationManager 实例.
    所有操作都通过类方法完成.

    示例:
        # Session 操作
        session = await ChatManager.create_session(title="新会话")
        session = await ChatManager.get_session(session_id)

        # Conversation 操作
        conv = await ChatManager.create_conversation(
            session_id=session.id,
            role="user",
            title="用户消息",
            content="你好"
        )
    """

    _instance: ChatManager | None = None
    _session_manager: SessionManager | None = None
    _conversation_manager: ConversationManager | None = None

    @classmethod
    def _get_session_manager(cls) -> SessionManager:
        """获取 SessionManager 实例."""
        if cls._session_manager is None:
            cls._session_manager = SessionManager()
        return cls._session_manager

    @classmethod
    def _get_conversation_manager(cls) -> ConversationManager:
        """获取 ConversationManager 实例."""
        if cls._conversation_manager is None:
            cls._conversation_manager = ConversationManager()
        return cls._conversation_manager

    @classmethod
    def initialize(
        cls,
        db_manager: DatabaseManager | None = None,
        embedding_client: EmbeddingClient | None = None,
        vector_storage: VectorStorage | None = None,
    ) -> None:
        """初始化 ChatManager.

        Args:
            db_manager: 数据库管理器（可选，默认自动获取）
            embedding_client: Embedding 客户端（可选）
            vector_storage: 向量存储（可选）
        """
        cls._session_manager = SessionManager(db_manager)
        cls._conversation_manager = ConversationManager(db_manager)

        if embedding_client and vector_storage:
            cls._conversation_manager.set_embedding(embedding_client, vector_storage)

    # =====================================================================
    # Session 操作
    # =====================================================================

    @classmethod
    async def create_session(
        cls,
        title: str = "New Session",
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionInfo:
        """创建新 Session."""
        return await cls._get_session_manager().create(
            title=title,
            description=description,
            metadata=metadata,
        )

    @classmethod
    async def get_session(cls, session_id: str) -> SessionInfo | None:
        """获取 Session."""
        return await cls._get_session_manager().get(session_id)

    @classmethod
    async def update_session(
        cls,
        session_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionInfo | None:
        """更新 Session."""
        return await cls._get_session_manager().update(
            session_id,
            title=title,
            description=description,
            status=status,
            metadata=metadata,
        )

    @classmethod
    async def delete_session(cls, session_id: str) -> bool:
        """删除 Session."""
        return await cls._get_session_manager().delete(session_id)

    @classmethod
    async def list_sessions(cls, limit: int = 20) -> list[SessionInfo]:
        """列出 Sessions."""
        return await cls._get_session_manager().list_recent(limit=limit)

    @classmethod
    async def session_exists(cls, session_id: str) -> bool:
        """检查 Session 是否存在."""
        return await cls._get_session_manager().exists(session_id)

    @classmethod
    def generate_session_id(cls, existing_ids: set[str] | None = None) -> str:
        """生成唯一 Session ID."""
        return generate_session_id(existing_ids)

    @classmethod
    async def ensure_session(cls, session_id: str | None = None) -> str:
        """确保 Session 存在，不存在则创建.

        Args:
            session_id: 期望的 Session ID

        Returns:
            Session ID（传入的或新创建的）
        """
        if session_id and await cls.session_exists(session_id):
            return session_id

        session = await cls.create_session()
        return session.id

    # =====================================================================
    # Conversation 操作
    # =====================================================================

    @classmethod
    async def create_conversation(
        cls,
        session_id: str,
        role: str,
        title: str,
        content: str,
        *,
        meta_info: dict[str, Any] | None = None,
        tokens: int = 0,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
    ) -> ConversationInfo:
        """创建新 Conversation."""
        return await cls._get_conversation_manager().create(
            session_id=session_id,
            role=role,
            title=title,
            content=content,
            meta_info=meta_info,
            tokens=tokens,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )

    @classmethod
    async def get_conversation(cls, conversation_id: str) -> ConversationInfo | None:
        """获取 Conversation."""
        return await cls._get_conversation_manager().get(conversation_id)

    @classmethod
    async def update_conversation(
        cls,
        conversation_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        meta_info: dict[str, Any] | None = None,
    ) -> ConversationInfo | None:
        """更新 Conversation."""
        return await cls._get_conversation_manager().update(
            conversation_id,
            title=title,
            content=content,
            meta_info=meta_info,
        )

    @classmethod
    async def delete_conversation(cls, conversation_id: str) -> bool:
        """删除 Conversation."""
        return await cls._get_conversation_manager().delete(conversation_id)

    @classmethod
    async def list_conversations(
        cls,
        session_id: str,
        limit: int = 1000,
        *,
        exclude_tool: bool = True,
    ) -> list[ConversationInfo]:
        """列出 Conversations."""
        return await cls._get_conversation_manager().list_by_session(
            session_id, limit, exclude_tool=exclude_tool
        )

    @classmethod
    async def get_conversation_history(
        cls,
        session_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """获取对话历史."""
        return await cls._get_conversation_manager().get_history(session_id, limit)
