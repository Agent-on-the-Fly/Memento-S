"""Session Manager — Session 生命周期管理.

管理 Session 的创建、查询、更新和删除.
"""

from __future__ import annotations

import secrets
import string
from typing import Any

from middleware.storage.core.engine import DatabaseManager
from middleware.storage.schemas import SessionCreate, SessionUpdate
from middleware.storage.services import SessionService

from .manager import _ServiceManager
from .types import SessionInfo

_ID_CHARS: str = string.ascii_lowercase + string.digits
_ID_LENGTH: int = 8


def generate_session_id(existing_ids: set[str] | None = None) -> str:
    """生成唯一的 Session ID."""
    for _ in range(100):
        candidate = "".join(secrets.choice(_ID_CHARS) for _ in range(_ID_LENGTH))
        if existing_ids is None or candidate not in existing_ids:
            return candidate
    raise RuntimeError("生成唯一 Session ID 失败，已尝试100次")


class SessionManager(_ServiceManager[SessionService]):
    """Session 管理器."""

    def __init__(self, db_manager: DatabaseManager | None = None) -> None:
        super().__init__(db_manager, lambda db: SessionService(db))

    async def create(
        self,
        title: str = "New Session",
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionInfo:
        """创建新 Session.

        Args:
            title: Session 标题
            description: Session 描述
            metadata: 元数据字典

        Returns:
            创建的 Session 信息
        """
        data = SessionCreate(
            title=title,
            description=description,
            meta_info=metadata or {},
        )
        result = await self._get_service().create(data)
        return SessionInfo.from_orm(result)

    async def get(self, session_id: str) -> SessionInfo | None:
        """获取 Session 信息.

        Args:
            session_id: Session ID

        Returns:
            Session 信息，不存在则返回 None
        """
        result = await self._get_service().get(session_id)
        if result is None:
            return None
        return SessionInfo.from_orm(result)

    async def update(
        self,
        session_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionInfo | None:
        """更新 Session.

        Args:
            session_id: Session ID
            title: 新标题
            description: 新描述
            status: 新状态
            metadata: 元数据（会与现有合并）

        Returns:
            更新后的 Session 信息，不存在则返回 None
        """
        # 如果需要合并 metadata，先查询当前值
        current_meta: dict[str, Any] = {}
        if metadata is not None:
            session = await self._get_service().get(session_id)
            if session:
                current_meta = dict(session.meta_info or {})
                current_meta.update(metadata)

        update_data = SessionUpdate()
        if title is not None:
            update_data.title = title
        if description is not None:
            update_data.description = description
        if status is not None:
            update_data.status = status
        if metadata is not None:
            update_data.meta_info = current_meta

        result = await self._get_service().update(session_id, update_data)
        if result is None:
            return None
        return SessionInfo.from_orm(result)

    async def delete(self, session_id: str) -> bool:
        """删除 Session（级联删除所有 Conversations）.

        Args:
            session_id: Session ID

        Returns:
            是否删除成功
        """
        return await self._get_service().delete(session_id)

    async def list_recent(self, limit: int = 20) -> list[SessionInfo]:
        """列出最近的 Sessions.

        Args:
            limit: 最大返回数量

        Returns:
            Session 列表
        """
        results = await self._get_service().list_recent(limit=limit)
        return [SessionInfo.from_orm(r) for r in results]

    async def exists(self, session_id: str) -> bool:
        """检查 Session 是否存在.

        Args:
            session_id: Session ID

        Returns:
            是否存在
        """
        result = await self._get_service().get(session_id)
        return result is not None
