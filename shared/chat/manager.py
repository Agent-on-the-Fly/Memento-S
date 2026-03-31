"""Shared Chat Manager — 内部基类.

提供延迟初始化的服务管理器基类。
"""

from __future__ import annotations

from typing import TypeVar, Generic, Callable

from middleware.storage.core.engine import DatabaseManager
from middleware.storage.services.base_service import BaseService


T = TypeVar("T", bound=BaseService)


class _ServiceManager(Generic[T]):
    """服务管理器基类 — 提供延迟初始化."""

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        service_factory: Callable[[DatabaseManager], T] | None = None,
    ) -> None:
        self._db_manager = db_manager
        self._service_factory = service_factory
        self._service: T | None = None

    def _get_service(self) -> T:
        """延迟获取服务实例."""
        if self._service is None:
            if self._db_manager is None:
                from middleware.storage.core.engine import get_db_manager

                self._db_manager = get_db_manager()
                if not getattr(self._db_manager, "_initialized", False):
                    raise RuntimeError(
                        "DatabaseManager is not initialized. "
                        "Call bootstrap() before using ChatManager."
                    )
            if self._service_factory:
                self._service = self._service_factory(self._db_manager)
        return self._service
