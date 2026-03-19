"""Conversation Manager - manages multi-turn conversations within a session.

This manager provides high-level APIs for:
1. Creating and managing conversations within a session
2. Tracking conversation history
3. Loading/saving conversation context
"""

from __future__ import annotations

from typing import Any

from utils.logger import get_logger
from middleware.storage import ConversationService, ConversationCreate, ConversationRead
from middleware.storage.schemas import ConversationUpdate

logger = get_logger(__name__)


class ConversationManager:
    """Manager for handling multi-turn conversations within a session."""

    def __init__(self):
        self._conversation_service = ConversationService()

    async def create_conversation(
        self, session_id: str, role: str, title: str, content: str, **kwargs
    ) -> ConversationRead:
        """Create a new conversation in the session.

        Args:
            session_id: Parent session ID
            role: 'user', 'assistant', or 'system'
            title: Conversation title (preview)
            content: Message content
            **kwargs: Additional fields (tokens, meta_info, etc.)

        Returns:
            Created conversation
        """
        data = ConversationCreate(
            session_id=session_id,
            role=role,
            title=title,
            content=content,
            meta_info=kwargs.get("meta_info", {}),
            tokens=kwargs.get("tokens") or 0,
        )

        conversation = await self._conversation_service.create(data)
        logger.debug(f"Created conversation: {conversation.id} (role={role})")
        return conversation

    async def get_conversation(self, conversation_id: str) -> ConversationRead | None:
        """Get a conversation by ID."""
        return await self._conversation_service.get(conversation_id)

    async def list_session_conversations(
        self, session_id: str, limit: int = 1000
    ) -> list[ConversationRead]:
        """List all conversations in a session, ordered by sequence."""
        return await self._conversation_service.list_by_session(session_id, limit)

    async def get_conversation_history(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get conversation history in a format suitable for LLM context.

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        conversations = await self.list_session_conversations(session_id, limit)

        history = []
        for conv in conversations:
            history.append(
                {
                    "role": conv.role,
                    "content": conv.content,
                    "conversation_id": conv.id,
                }
            )

        return history

    async def update_conversation(
        self, conversation_id: str, **updates
    ) -> ConversationRead | None:
        """Update a conversation."""
        data = ConversationUpdate(**updates)
        return await self._conversation_service.update(conversation_id, data)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        return await self._conversation_service.delete(conversation_id)
