"""Session service for managing chat sessions."""

from __future__ import annotations

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from middleware.storage.models import Session, SessionStatus
from middleware.storage.schemas import SessionCreate, SessionRead, SessionUpdate
from middleware.storage.utils import get_east_8_time
from .base_service import BaseService


class SessionService(BaseService):
    """Service for managing chat sessions with automatic database session management."""

    # Public API - auto-managed session

    async def create(self, data: SessionCreate) -> SessionRead:
        """Create a new session.

        Args:
            data: Session creation data

        Returns:
            Created session
        """
        return await self._with_session(lambda db: self._create(db, data))

    async def get(self, session_id: str) -> SessionRead | None:
        """Get session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session or None if not found
        """
        return await self._with_session(lambda db: self._get(db, session_id))

    async def update(self, session_id: str, data: SessionUpdate) -> SessionRead | None:
        """Update session.

        Args:
            session_id: Session ID
            data: Update data

        Returns:
            Updated session or None if not found
        """
        return await self._with_session(lambda db: self._update(db, session_id, data))

    async def delete(self, session_id: str) -> bool:
        """Delete session.

        Args:
            session_id: Session ID

        Returns:
            True if deleted, False if not found
        """
        return await self._with_session(lambda db: self._delete(db, session_id))

    async def list_recent(self, limit: int = 20) -> list[SessionRead]:
        """Get recent sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of sessions ordered by updated_at desc
        """
        return await self._with_session(lambda db: self._list_recent(db, limit))

    async def list_by_status(self, status: str, limit: int = 100) -> list[SessionRead]:
        """Get sessions by status.

        Args:
            status: Session status
            limit: Maximum number of sessions to return

        Returns:
            List of sessions
        """
        return await self._with_session(
            lambda db: self._list_by_status(db, status, limit)
        )

    # Private implementation - requires manual session

    async def _create(self, db: AsyncSession, data: SessionCreate) -> SessionRead:
        """Create session implementation."""
        obj = Session(
            title=data.title,
            description=data.description,
            meta_info=data.meta_info,
            status=SessionStatus.ACTIVE.value,
            created_at=get_east_8_time(),
            updated_at=get_east_8_time(),
        )
        db.add(obj)
        await db.commit()
        await db.refresh(obj)
        return SessionRead.model_validate(obj)

    async def _get(self, db: AsyncSession, session_id: str) -> SessionRead | None:
        """Get session implementation."""
        obj = await db.get(Session, session_id)
        if obj is None:
            return None
        return SessionRead.model_validate(obj)

    async def _update(
        self, db: AsyncSession, session_id: str, data: SessionUpdate
    ) -> SessionRead | None:
        """Update session implementation."""
        obj = await db.get(Session, session_id)
        if obj is None:
            return None

        if data.title is not None:
            obj.title = data.title
        if data.description is not None:
            obj.description = data.description
        if data.status is not None:
            obj.status = data.status
        if data.meta_info is not None:
            obj.meta_info = data.meta_info

        # Update timestamp to East 8 time
        obj.updated_at = get_east_8_time()

        db.add(obj)
        await db.commit()
        await db.refresh(obj)
        return SessionRead.model_validate(obj)

    async def _delete(self, db: AsyncSession, session_id: str) -> bool:
        """Delete session implementation."""
        obj = await db.get(Session, session_id)
        if obj is None:
            return False
        await db.delete(obj)
        await db.commit()
        return True

    async def _list_recent(self, db: AsyncSession, limit: int) -> list[SessionRead]:
        """List recent sessions implementation."""
        stmt = select(Session).order_by(desc(Session.updated_at)).limit(limit)
        result = await db.execute(stmt)
        rows = result.scalars().all()
        return [SessionRead.model_validate(x) for x in rows]

    async def _list_by_status(
        self, db: AsyncSession, status: str, limit: int
    ) -> list[SessionRead]:
        """List by status implementation."""
        stmt = (
            select(Session)
            .where(Session.status == status)
            .order_by(desc(Session.updated_at))
            .limit(limit)
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()
        return [SessionRead.model_validate(x) for x in rows]
