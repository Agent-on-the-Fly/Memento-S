"""core.context — 统一的上下文管理模块。

ContextManager 是 agent 唯一的上下文入口，负责:
  - prompt 构建与历史管理
  - scratchpad（tool result 持久化）
  - compress（单条超长消息 LLM 摘要）
  - compact（消息列表 LLM 全量合并）
"""
from .manager import ContextManager

__all__ = [
    "ContextManager",
]
