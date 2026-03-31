"""Conversation Manager — 多轮对话管理.

管理 Session 内的 Conversation 创建、查询、更新，以及对话历史.
"""

from __future__ import annotations

import asyncio
from typing import Any

from middleware.llm.embedding_client import EmbeddingClient
from middleware.storage.core.engine import DatabaseManager
from middleware.storage.schemas import ConversationCreate, ConversationUpdate
from middleware.storage.services import ConversationService
from middleware.storage.vector_storage import VectorStorage
from utils.logger import get_logger

from .manager import _ServiceManager
from .types import ConversationInfo

logger = get_logger(__name__)

_EMBEDDABLE_ROLES = {"user", "assistant"}


class ConversationManager(_ServiceManager[ConversationService]):
    """Conversation 管理器."""

    def __init__(self, db_manager: DatabaseManager | None = None) -> None:
        super().__init__(db_manager, lambda db: ConversationService(db))
        self._embedding_client: EmbeddingClient | None = None
        self._vector_storage: VectorStorage | None = None

    def set_embedding(
        self,
        client: EmbeddingClient,
        vector_storage: VectorStorage,
    ) -> None:
        """设置 Embedding 依赖，启用后自动保存向量."""
        self._embedding_client = client
        self._vector_storage = vector_storage

    async def create(
        self,
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
        """创建新 Conversation.

        Args:
            session_id: 所属 Session ID
            role: 角色（user/assistant/system/tool）
            title: 标题（内容预览）
            content: 内容
            meta_info: 元数据
            tokens: Token 数量
            tool_calls: 工具调用列表
            tool_call_id: 关联的工具调用 ID

        Returns:
            创建的 Conversation 信息
        """
        data = ConversationCreate(
            session_id=session_id,
            role=role,
            title=title,
            content=content,
            meta_info=meta_info or {},
            tokens=tokens,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )

        result = await self._get_service().create(data)
        logger.debug(f"Created conversation: {result.id} (role={role})")

        # 异步保存 embedding（fire-and-forget）
        if (
            role in _EMBEDDABLE_ROLES
            and content
            and self._embedding_client
            and self._vector_storage
        ):
            asyncio.create_task(self._embed_conversation(result.id, content))

        return ConversationInfo.from_orm(result)

    async def get(self, conversation_id: str) -> ConversationInfo | None:
        """获取 Conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation 信息，不存在则返回 None
        """
        result = await self._get_service().get(conversation_id)
        if result is None:
            return None
        return ConversationInfo.from_orm(result)

    async def update(
        self,
        conversation_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        meta_info: dict[str, Any] | None = None,
    ) -> ConversationInfo | None:
        """更新 Conversation.

        Args:
            conversation_id: Conversation ID
            title: 新标题
            content: 新内容
            meta_info: 新元数据（覆盖原有）

        Returns:
            更新后的 Conversation 信息，不存在则返回 None
        """
        data = ConversationUpdate()
        if title is not None:
            data.title = title
        if content is not None:
            data.content = content
        if meta_info is not None:
            data.meta_info = meta_info

        result = await self._get_service().update(conversation_id, data)
        if result is None:
            return None
        return ConversationInfo.from_orm(result)

    async def delete(self, conversation_id: str) -> bool:
        """删除 Conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            是否删除成功
        """
        return await self._get_service().delete(conversation_id)

    async def list_by_session(
        self,
        session_id: str,
        limit: int = 1000,
        *,
        exclude_tool: bool = True,
    ) -> list[ConversationInfo]:
        """列出 Session 下的所有 Conversations.

        Args:
            session_id: Session ID
            limit: 最大返回数量
            exclude_tool: 是否排除 tool 角色的消息

        Returns:
            Conversation 列表
        """
        results = await self._get_service().list_by_session(
            session_id, limit, exclude_tool=exclude_tool
        )
        return [ConversationInfo.from_orm(r) for r in results]

    async def get_history(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """获取对话历史（用于 LLM 上下文）.

        Args:
            session_id: Session ID
            limit: 最大返回消息数

        Returns:
            消息字典列表，格式:
            [{"role": "...", "content": "...", "conversation_id": "..."}, ...]
        """
        conversations = await self.list_by_session(
            session_id, limit, exclude_tool=False
        )

        history: list[dict[str, Any]] = []
        for conv in conversations:
            msg: dict[str, Any] = {
                "role": conv.role,
                "content": conv.content,
                "conversation_id": conv.id,
                "tokens": conv.tokens,
            }
            if conv.tool_call_id:
                msg["tool_call_id"] = conv.tool_call_id
            if conv.tool_calls:
                msg["tool_calls"] = conv.tool_calls
            history.append(msg)

        return history

    async def _embed_conversation(self, conv_id: str, content: str) -> None:
        """异步生成并保存 embedding."""
        try:
            vector = await self._embedding_client.embed_query(content)
            await self._vector_storage.save(conv_id, vector)
            logger.debug(f"Embedded conversation: {conv_id}")
        except Exception as e:
            logger.warning(f"Failed to embed conversation {conv_id}: {e}")
