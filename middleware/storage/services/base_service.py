"""Base service with automatic database session management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from middleware.storage.core.engine import get_db_manager

T = TypeVar("T")


class BaseService:
    """Base service with automatic database session management.

    Concurrency Safety:
    - Each service method creates its own isolated database session
    - Sessions are not shared between concurrent calls
    - Safe for concurrent use in async/await environment
    - NOT thread-safe (use in single-threaded async context)

    Usage:
        # Method 1: Use auto-managed session (recommended)
        service = SessionService()
        session = await service.create(title="test")

        # Method 2: Use manual session for transaction control
        async with service.session() as db:
            session = await service._create(db, title="test")
            await service.commit(db)
    """

    def __init__(self, db_manager=None):
        """Initialize service with optional db manager.

        Args:
            db_manager: Database manager instance. If None, uses global singleton.
        """
        self._db_manager = db_manager or get_db_manager()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session context manager.

        Yields:
            AsyncSession: Database session

        Example:
            async with service.session() as db:
                result = await service.get_with_session(db, id)
        """
        async with self._db_manager.session_factory() as db:
            yield db

    async def _with_session(self, operation) -> T:
        """Execute operation with auto-managed session.

        Args:
            operation: Async function that takes db session as first argument

        Returns:
            Operation result
        """
        async with self.session() as db:
            return await operation(db)

    @staticmethod
    async def commit(db: AsyncSession) -> None:
        """Commit transaction."""
        await db.commit()

    @staticmethod
    async def rollback(db: AsyncSession) -> None:
        """Rollback transaction."""
        await db.rollback()

    @staticmethod
    async def refresh(db: AsyncSession, obj) -> None:
        """Refresh object from database."""
        await db.refresh(obj)
