"""SessionManager — entry point for session management.

Simplified two-layer architecture:
- Session (top-level workspace)
- Conversation (stores complete message content directly)

Note: This manager only handles Session operations.
For Conversation operations, use ConversationManager directly.
"""

from __future__ import annotations

import secrets
import string
from typing import Any

from middleware.storage.schemas import (
    SessionCreate,
    SessionUpdate,
)
from middleware.storage.services import SessionService

_ID_CHARS: str = string.ascii_lowercase + string.digits
_ID_LENGTH: int = 8


def generate_session_id(existing_ids: set[str] | None = None) -> str:
    """Generate a unique session ID."""
    for _ in range(100):
        candidate = "".join(secrets.choice(_ID_CHARS) for _ in range(_ID_LENGTH))
        if existing_ids is None or candidate not in existing_ids:
            return candidate
    raise RuntimeError("Failed to generate unique session ID after 100 attempts")


class SessionManager:
    """Manager for session lifecycle.

    This is the entry point for session-related operations only.
    For conversation management, use ConversationManager directly.
    """

    def __init__(self) -> None:
        self._session_service = SessionService()

    async def create_session(
        self, title: str = "New Session", metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create a new session."""
        create = SessionCreate(
            title=title,
            meta_info=metadata or {},
        )
        created = await self._session_service.create(create)
        return {
            "id": created.id,
            "title": created.title,
            "created_at": created.created_at.isoformat()
            if created.created_at
            else None,
            "updated_at": created.updated_at.isoformat()
            if created.updated_at
            else None,
        }

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session by ID."""
        session = await self._session_service.get(session_id)
        if session is None:
            return None

        return {
            "id": session.id,
            "title": session.title,
            "description": session.description,
            "status": session.status,
            "created_at": session.created_at.isoformat()
            if session.created_at
            else None,
            "updated_at": session.updated_at.isoformat()
            if session.updated_at
            else None,
            "conversation_count": session.conversation_count,
            "total_tokens": session.total_tokens,
            "model": session.meta_info.get("model", ""),
            "metadata": session.meta_info,
        }

    async def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update session properties."""
        update_data = SessionUpdate()

        if title is not None:
            update_data.title = title
        if description is not None:
            update_data.description = description
        if status is not None:
            update_data.status = status
        if metadata is not None:
            session = await self._session_service.get(session_id)
            if session:
                current_meta = dict(session.meta_info)
                current_meta.update(metadata)
                update_data.meta_info = current_meta

        updated = await self._session_service.update(session_id, update_data)
        return updated is not None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its conversations."""
        return await self._session_service.delete(session_id)

    async def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent sessions."""
        sessions = await self._session_service.list_recent(limit=limit)
        return [
            {
                "id": s.id,
                "title": s.title,
                "description": s.description,
                "status": s.status,
                "conversation_count": s.conversation_count,
                "total_tokens": s.total_tokens,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            }
            for s in sessions
        ]

    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        session = await self._session_service.get(session_id)
        return session is not None
