"""Conversation service for managing conversations within sessions."""

from __future__ import annotations

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from middleware.storage.models import Conversation, Session
from middleware.storage.schemas import (
    ConversationCreate,
    ConversationRead,
    ConversationUpdate,
)
from middleware.storage.utils import get_east_8_time
from .base_service import BaseService


class ConversationService(BaseService):
    """Service for conversation CRUD operations."""

    async def create(self, data: ConversationCreate) -> ConversationRead:
        """Create a new conversation. Auto-assigns sequence if not provided."""
        return await self._with_session(lambda db: self._create(db, data))

    async def get(self, conversation_id: str) -> ConversationRead | None:
        """Get conversation by ID."""
        return await self._with_session(lambda db: self._get(db, conversation_id))

    async def update(
        self, conversation_id: str, data: ConversationUpdate
    ) -> ConversationRead | None:
        """Update conversation."""
        return await self._with_session(
            lambda db: self._update(db, conversation_id, data)
        )

    async def delete(self, conversation_id: str) -> bool:
        """Delete conversation."""
        return await self._with_session(lambda db: self._delete(db, conversation_id))

    async def list_by_session(
        self, session_id: str, limit: int = 1000
    ) -> list[ConversationRead]:
        """List conversations by session, ordered by sequence."""
        return await self._with_session(
            lambda db: self._list_by_session(db, session_id, limit)
        )

    async def get_next_sequence(self, session_id: str) -> int:
        """Get next sequence number for a session."""
        return await self._with_session(
            lambda db: self._get_next_sequence(db, session_id)
        )

    async def _create(
        self, db: AsyncSession, data: ConversationCreate
    ) -> ConversationRead:
        """Create conversation implementation."""
        # Auto-assign sequence if not provided
        sequence = data.sequence
        if sequence is None:
            sequence = await self._get_next_sequence(db, data.session_id)

        obj = Conversation(
            session_id=data.session_id,
            sequence=sequence,
            role=data.role,
            title=data.title,
            content=data.content,
            content_detail=data.content_detail,
            tool_calls=data.tool_calls,
            tool_call_id=data.tool_call_id,
            meta_info=data.meta_info,
            tokens=data.tokens,
            created_at=get_east_8_time(),
            updated_at=get_east_8_time(),
        )
        db.add(obj)

        # Flush first so aggregate queries include this new conversation
        await db.flush()

        # Update session stats
        await self._update_session_stats(db, data.session_id)

        await db.commit()
        await db.refresh(obj)
        return ConversationRead.model_validate(obj)

    async def _get(
        self, db: AsyncSession, conversation_id: str
    ) -> ConversationRead | None:
        """Get conversation implementation."""
        obj = await db.get(Conversation, conversation_id)
        if obj is None:
            return None
        return ConversationRead.model_validate(obj)

    async def _update(
        self, db: AsyncSession, conversation_id: str, data: ConversationUpdate
    ) -> ConversationRead | None:
        """Update conversation implementation."""
        obj = await db.get(Conversation, conversation_id)
        if obj is None:
            return None

        if data.title is not None:
            obj.title = data.title
        if data.content is not None:
            obj.content = data.content
        if data.meta_info is not None:
            obj.meta_info = data.meta_info

        # Update timestamp to East 8 time
        obj.updated_at = get_east_8_time()

        db.add(obj)
        await db.commit()
        await db.refresh(obj)
        return ConversationRead.model_validate(obj)

    async def _delete(self, db: AsyncSession, conversation_id: str) -> bool:
        """Delete conversation implementation."""
        obj = await db.get(Conversation, conversation_id)
        if obj is None:
            return False

        session_id = obj.session_id
        await db.delete(obj)

        # Update session stats
        await self._update_session_stats(db, session_id)

        await db.commit()
        return True

    async def _list_by_session(
        self, db: AsyncSession, session_id: str, limit: int
    ) -> list[ConversationRead]:
        """List by session implementation."""
        stmt = (
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .order_by(Conversation.sequence.asc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()
        return [ConversationRead.model_validate(x) for x in rows]

    async def _get_next_sequence(self, db: AsyncSession, session_id: str) -> int:
        """Get next sequence number for a session."""
        stmt = select(func.max(Conversation.sequence)).where(
            Conversation.session_id == session_id
        )
        result = await db.execute(stmt)
        max_seq = result.scalar()
        return (max_seq or 0) + 1

    async def _update_session_stats(self, db: AsyncSession, session_id: str) -> None:
        """Update session conversation count and total tokens."""
        # Get stats
        stmt = select(
            func.count(Conversation.id).label("count"),
            func.coalesce(func.sum(Conversation.tokens), 0).label("tokens"),
        ).where(Conversation.session_id == session_id)
        result = await db.execute(stmt)
        stats = result.one()

        # Update session
        session = await db.get(Session, session_id)
        if session:
            session.conversation_count = stats.count
            session.total_tokens = stats.tokens
            db.add(session)
