from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Session Schemas
# ============================================================================


class SessionCreate(BaseModel):
    """创建会话请求"""

    title: str
    description: str | None = None
    meta_info: dict[str, Any] = Field(default_factory=dict)


class SessionUpdate(BaseModel):
    """更新会话请求"""

    title: str | None = None
    description: str | None = None
    status: str | None = None
    meta_info: dict[str, Any] | None = None


class SessionRead(BaseModel):
    """会话响应"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    description: str | None
    status: str
    meta_info: dict[str, Any]
    conversation_count: int
    total_tokens: int
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Conversation Schemas
# ============================================================================


class ConversationCreate(BaseModel):
    """创建对话请求"""

    session_id: str
    role: str  # user/assistant/system
    title: str
    content: str
    sequence: int | None = None  # 如果为None，自动分配
    content_detail: dict[str, Any] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    meta_info: dict[str, Any] = Field(default_factory=dict)
    tokens: int = 0


class ConversationUpdate(BaseModel):
    """更新对话请求"""

    title: str | None = None
    content: str | None = None
    meta_info: dict[str, Any] | None = None


class ConversationRead(BaseModel):
    """对话响应"""

    model_config = ConfigDict(from_attributes=True)

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
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Skill Schemas
# ============================================================================


class SkillCreate(BaseModel):
    """创建技能请求"""

    name: str
    display_name: str | None = None
    description: str | None = None
    version: str = "0.1.0"
    author: str | None = None
    source_type: str
    source_url: str | None = None
    local_path: str | None = None
    embedding: bytes | None = None
    tags: list[str] = Field(default_factory=list)
    category: str | None = None
    meta_info: dict[str, Any] = Field(default_factory=dict)


class SkillUpdate(BaseModel):
    """更新技能请求"""

    display_name: str | None = None
    description: str | None = None
    version: str | None = None
    status: str | None = None
    tags: list[str] | None = None
    category: str | None = None
    meta_info: dict[str, Any] | None = None


class SkillRead(BaseModel):
    """技能响应"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    display_name: str | None
    description: str | None
    version: str
    author: str | None
    status: str
    source_type: str
    source_url: str | None
    local_path: str | None
    tags: list[str]
    category: str | None
    meta_info: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class SkillWithEmbedding(SkillRead):
    """包含向量嵌入的技能响应"""

    embedding: bytes | None = None
    embedding_model: str | None = None
