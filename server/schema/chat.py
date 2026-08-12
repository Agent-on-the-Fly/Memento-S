# SPDX-License-Identifier: Apache-2.0
"""
server/schema/chat.py
聊天相关 Pydantic 模型
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str  # "user" or "assistant"
    content: str
    created_at: Optional[datetime] = None


class ChatSession(BaseModel):
    """聊天会话"""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class StreamEvent(BaseModel):
    """流式事件"""
    type: str  # "text", "done", "error", "step"
    content: Optional[str] = None
    step: Optional[int] = None
