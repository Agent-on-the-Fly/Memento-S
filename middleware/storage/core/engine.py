"""Async database engine with SQLite support."""

from __future__ import annotations

import asyncio
import platform
import sys
from pathlib import Path
from threading import Lock
from typing import Optional

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class DatabaseManager:
    """Singleton-style async database manager.

    Responsibilities:
    - Build AsyncEngine and AsyncSession factory
    - Set up SQLite pragmas for optimal performance

    Thread-safety:
    - instance() is thread-safe using double-checked locking
    - init() should be called once during app startup (not thread-safe)
    - session_factory is stateless and safe for concurrent use
    - Each session is independent and bound to async context
    """

    _instance: Optional["DatabaseManager"] = None
    _lock: Lock = Lock()
    _init_lock: asyncio.Lock = asyncio.Lock()
    _initialized: bool = False

    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._db_url: str | None = None

    @classmethod
    def instance(cls) -> "DatabaseManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("DatabaseManager is not initialized. Call init() first.")
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError("DatabaseManager is not initialized. Call init() first.")
        return self._session_factory

    async def init(
        self,
        db_url: str,
        echo: bool = False,
    ) -> None:
        """Initialize the database manager (async and coroutine-safe).

        This method is safe to call multiple times. Only the first call
        will perform initialization.

        Args:
            db_url: Database URL (e.g., sqlite+aiosqlite:///path/to/db.sqlite)
            echo: Whether to echo SQL statements
        """
        if self._initialized:
            return

        async with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return

            self._db_url = db_url

            # Create async engine
            self._engine = create_async_engine(
                db_url,
                echo=echo,
                future=True,
                pool_pre_ping=True,
            )

            # Set up connection event for SQLite pragmas
            @event.listens_for(self._engine.sync_engine, "connect")
            def _on_connect(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute("PRAGMA foreign_keys = ON;")
                    cursor.execute("PRAGMA journal_mode = WAL;")
                    cursor.execute("PRAGMA synchronous = NORMAL;")
                finally:
                    cursor.close()

            self._session_factory = async_sessionmaker(
                bind=self._engine,
                expire_on_commit=False,
                class_=AsyncSession,
                autoflush=False,
                autocommit=False,
            )

            self._initialized = True

    async def dispose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()


def get_db_manager() -> DatabaseManager:
    return DatabaseManager.instance()
