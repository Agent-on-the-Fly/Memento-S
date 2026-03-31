"""retrieval/schema.py — 检索层数据模型

定义召回相关的数据结构和契约。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class RecallCandidate:
    """召回候选 — 统一的检索结果数据结构

    用于本地检索、远程检索和多路召回合并后的统一表示。

    Attributes:
        name: skill 名称
        description: skill 描述
        source: 来源类型（local/remote）
        score: 相似度分数 (0-1)
        match_type: 匹配类型标记（如 "embedding", "fulltext", "exact"）
        skill: 本地 skill 对象（本地召回时有值，远程为 None）
        metadata: 额外元数据（远程召回时的云端信息等）
    """

    name: str
    description: str = ""
    source: Literal["local", "remote"] = "local"
    score: float = 0.0
    match_type: str = ""
    skill: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
