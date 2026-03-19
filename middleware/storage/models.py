"""
Memento-S 存储模型 - 简化版（Session + Conversation 两层架构）

核心实体：
- Session: 顶层会话，管理整个对话生命周期
- Conversation: 单轮对话，直接存储消息内容
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy 基础类"""

    pass


# ============================================================================
# Session 模型（顶层会话）
# ============================================================================


class SessionStatus(str, Enum):
    """会话状态"""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class Session(Base):
    """
    会话表 - 存储完整的会话信息

    包含：
    - 基础信息（标题、描述）
    - 状态管理（活跃、暂停、完成等）
    - 元数据（标签、分类、优先级等）
    - 统计信息（对话轮数、token数等）
    """

    __tablename__ = "sessions"

    # 主键
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # 基础信息
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 状态管理
    status: Mapped[str] = mapped_column(
        String(32), default=SessionStatus.ACTIVE.value, nullable=False
    )

    # 元数据
    meta_info: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        comment="存储模型配置、标签等",
    )

    # 统计信息
    conversation_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="对话轮数"
    )
    total_tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="总token消耗"
    )

    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # 关联关系
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="Conversation.sequence.asc()",
    )

    # 索引
    __table_args__ = (
        Index("idx_session_status", "status"),
        Index("idx_session_created", "created_at"),
        Index("idx_session_updated", "updated_at"),
    )


# ============================================================================
# Conversation 模型（单轮对话）
# ============================================================================


class ConversationRole(str, Enum):
    """对话角色"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Conversation(Base):
    """
    对话表 - 存储单轮对话的完整内容

    包含原来 Message 的所有字段：
    - 基础信息（角色、内容）
    - 内容详情（支持多模态、工具调用等）
    - 序列管理（在 Session 中的顺序）
    - 元数据（token数、耗时、模型信息等）
    """

    __tablename__ = "conversations"

    # 主键
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # 外键关联
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )

    # 序列号（在 Session 中的顺序，从1开始）
    sequence: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="在 Session 中的序号"
    )

    # 角色（user/assistant/system）
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="user/assistant/system"
    )

    # 标题（内容预览，用于侧边栏显示）
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    # 内容（支持长文本）
    content: Mapped[str | None] = mapped_column(Text, nullable=True, comment="文本内容")

    # 内容详情（JSON格式，支持复杂结构）
    content_detail: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, comment="详细内容结构，如工具调用参数、多模态内容等"
    )

    # 工具调用相关
    tool_calls: Mapped[list[dict] | None] = mapped_column(
        JSON, nullable=True, comment="工具调用列表（OpenAI格式）"
    )
    tool_call_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="关联的工具调用ID"
    )

    # 元数据（token数、模型名称、耗时、成本等）
    meta_info: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False, comment="token数、模型名称、耗时、成本等"
    )

    # Token 统计
    tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="本对话的token数"
    )

    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # 关联关系
    session: Mapped[Session] = relationship(back_populates="conversations")

    # 索引
    __table_args__ = (
        Index("idx_conversation_session", "session_id"),
        Index("idx_conversation_sequence", "session_id", "sequence"),
        Index("idx_conversation_role", "role"),
        Index("idx_conversation_updated", "updated_at"),
    )


# ============================================================================
# Skill 模型（保持不变）
# ============================================================================


class SkillStatus(str, Enum):
    """技能状态"""

    ACTIVE = "active"
    DISABLED = "disabled"
    DEPRECATED = "deprecated"


class SkillSourceType(str, Enum):
    """技能来源类型"""

    BUILTIN = "builtin"
    LOCAL = "local"
    GITHUB = "github"
    MARKETPLACE = "marketplace"
    CUSTOM = "custom"


class Skill(Base):
    """技能表"""

    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # 基础信息
    name: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, comment="技能唯一标识名"
    )
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(32), default="0.1.0", nullable=False)
    author: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # 状态
    status: Mapped[str] = mapped_column(
        String(32), default=SkillStatus.ACTIVE.value, nullable=False
    )

    # 来源信息
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    local_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # 向量信息
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    # 元数据
    tags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    meta_info: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # 索引
    __table_args__ = (
        Index("idx_skill_name", "name"),
        Index("idx_skill_status", "status"),
        Index("idx_skill_source", "source_type"),
        Index("idx_skill_category", "category"),
        Index("idx_skill_created", "created_at"),
    )
