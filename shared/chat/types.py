"""Shared Chat Types — 统一的数据类型定义.

提供 Session 和 Conversation 的数据类，用于类型安全和清晰的接口.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class SessionInfo:
    """Session 信息数据类."""

    id: str
    title: str
    description: str | None
    status: str
    created_at: datetime | None
    updated_at: datetime | None
    conversation_count: int
    total_tokens: int
    metadata: dict[str, Any]

    @property
    def model(self) -> str:
        """从 metadata 中获取模型名称."""
        return self.metadata.get("model", "")

    @classmethod
    def from_orm(cls, obj: Any) -> "SessionInfo":
        """从 ORM 对象创建 SessionInfo."""
        return cls(
            id=obj.id,
            title=obj.title,
            description=obj.description,
            status=obj.status,
            created_at=obj.created_at,
            updated_at=obj.updated_at,
            conversation_count=getattr(obj, "conversation_count", 0),
            total_tokens=getattr(obj, "total_tokens", 0),
            metadata=getattr(obj, "meta_info", {}) or {},
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionInfo":
        """从字典创建 SessionInfo."""
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")

        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description"),
            status=data.get("status", "active"),
            created_at=created_at,
            updated_at=updated_at,
            conversation_count=data.get("conversation_count", 0),
            total_tokens=data.get("total_tokens", 0),
            metadata=data.get("metadata", data.get("meta_info", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "conversation_count": self.conversation_count,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "metadata": self.metadata,
        }


@dataclass
class ConversationInfo:
    """Conversation 信息数据类."""

    id: str
    session_id: str
    sequence: int
    role: str
    title: str
    content: str | None
    content_detail: dict[str, Any] | None
    tool_calls: list[dict] | None
    tool_call_id: str | None
    meta_info: dict[str, Any]
    tokens: int
    created_at: datetime | None
    updated_at: datetime | None

    @property
    def model_name(self) -> str:
        """从 meta_info 中获取模型名称."""
        return self.meta_info.get("model_name", "")

    @property
    def duration_seconds(self) -> float | None:
        """从 meta_info 中获取响应时长(秒)."""
        return self.meta_info.get("duration_seconds")

    @property
    def steps(self) -> int | None:
        """从 meta_info 中获取步骤数."""
        return self.meta_info.get("steps")

    @classmethod
    def from_orm(cls, obj: Any) -> "ConversationInfo":
        """从 ORM 对象创建 ConversationInfo."""
        return cls(
            id=obj.id,
            session_id=obj.session_id,
            sequence=obj.sequence,
            role=obj.role,
            title=obj.title,
            content=obj.content,
            content_detail=obj.content_detail,
            tool_calls=obj.tool_calls,
            tool_call_id=obj.tool_call_id,
            meta_info=getattr(obj, "meta_info", {}) or {},
            tokens=getattr(obj, "tokens", 0) or 0,
            created_at=obj.created_at,
            updated_at=obj.updated_at,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationInfo":
        """从字典创建 ConversationInfo."""
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")

        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            id=data["id"],
            session_id=data["session_id"],
            sequence=data["sequence"],
            role=data["role"],
            title=data["title"],
            content=data.get("content"),
            content_detail=data.get("content_detail"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            meta_info=data.get("meta_info", {}),
            tokens=data.get("tokens", 0),
            created_at=created_at,
            updated_at=updated_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "role": self.role,
            "title": self.title,
            "content": self.content,
            "content_detail": self.content_detail,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "meta_info": self.meta_info,
            "model_name": self.model_name,
            "duration_seconds": self.duration_seconds,
            "steps": self.steps,
            "tokens": self.tokens,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
