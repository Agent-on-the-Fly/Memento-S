# SPDX-License-Identifier: Apache-2.0
"""shared/schema/skill_search.py — 检索结果命中（跨 retrieval 层和 UI）。

替代旧的 RecallCandidate（core/skill/retrieval/schema.py）。
与 RecallCandidate 的关键区别：
- 独立字段（name, description, source），不依赖 SkillManifest
- 避免远程召回时缺少完整 Skill 对象的兼容问题
- 统一 local/cloud 的检索结果格式
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class SkillSearchResult:
    """检索结果命中 — 统一的跨层检索契约。

    Attributes:
        name: skill 名称
        description: skill 描述
        source: 来源类型（"local" / "cloud"）
        score: 相似度分数 (0-1)
        match_type: 匹配类型（"keyword", "embedding", "remote"）
        metadata: 额外信息（远程召回时的云端信息等）
    """

    name: str
    description: str = ""
    source: Literal["local", "cloud"] = "local"
    score: float = 0.0
    match_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["SkillSearchResult"]
