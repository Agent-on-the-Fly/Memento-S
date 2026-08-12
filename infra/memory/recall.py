# SPDX-License-Identifier: Apache-2.0
"""Topic recall — pure keyword matching, no embedding.

Matches query tokens against topic slug, title, hook, and content.
Reuses the scoring strategy from core/skill/retrieval/local_recall.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

CONTENT_PREVIEW_CHARS = 300


@dataclass
class TopicMatch:
    """A single matched topic result."""
    slug: str
    title: str
    hook: str
    content_preview: str
    score: float


def _match_score(
    query: str,
    slug: str,
    title: str,
    hook: str,
    content: str,
) -> float:
    """Keyword matching score for a topic.

    | 匹配类型                     | 分数  |
    |-----------------------------|-------|
    | token 完全匹配 slug/title   | 1.0   |
    | token 完全匹配 hook         | 0.9   |
    | token 子串匹配 slug/title   | 0.7   |
    | token 出现在 content       | 0.5   |
    | fuzzy match (ratio >= 0.6) | 0.3 * similarity |
    | 无匹配                     | 0.0   |

    Returns:
        Highest match score (0.0 ~ 1.0)
    """
    query_lower = query.lower().strip()
    if not query_lower:
        return 0.0

    tokens = query_lower.split()
    slug_lower = slug.lower()
    title_lower = title.lower()
    hook_lower = hook.lower()
    content_lower = content.lower()

    max_score = 0.0

    for token in tokens:
        # 1. token 全匹配 slug/title（前缀视为 1.0）
        if token == slug_lower or slug_lower.startswith(token):
            return 1.0
        if token == title_lower or title_lower.startswith(token):
            return 1.0

        # 2. token 全匹配 hook
        if token == hook_lower:
            max_score = max(max_score, 0.9)
            continue

        # 3. token 子串匹配 slug/title
        if token in slug_lower or token in title_lower:
            max_score = max(max_score, 0.7)
            continue

        # 4. token 出现在 content
        if token in content_lower:
            max_score = max(max_score, 0.5)
            continue

        # 5. fuzzy matching（编辑距离）
        for target in [slug_lower, title_lower, hook_lower]:
            if not target:
                continue
            fuzzy = SequenceMatcher(None, token, target).ratio()
            if fuzzy >= 0.6:
                max_score = max(max_score, 0.3 * fuzzy)
                break

        # 6. query 整体作为子串出现
        if query_lower in slug_lower or query_lower in title_lower:
            max_score = max(max_score, 0.3)

    return max(0.0, min(1.0, max_score))


def recall_topics(
    topics_dir: Path,
    query: str,
    k: int = 5,
) -> list[TopicMatch]:
    """关键词检索 topics/ 目录下匹配的 topic 文件。

    Args:
        topics_dir: memory/topics/ 目录路径
        query: 搜索查询
        k: 返回的最大结果数

    Returns:
        按匹配分数降序排列的 TopicMatch 列表
    """
    if not topics_dir.exists() or not query.strip():
        return []

    candidates: list[TopicMatch] = []

    for path in sorted(topics_dir.glob("*.md")):
        slug = path.stem
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue

        lines = content.split("\n")
        title = lines[0].strip().lstrip("# ") if lines else slug
        # hook 是第一行之后第一个不以 # 开头的内容行，或整个文件的第二行
        hook = ""
        for line in lines[1:]:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                hook = stripped
                break

        score = _match_score(query, slug, title, hook, content)
        if score <= 0.0:
            continue

        candidates.append(TopicMatch(
            slug=slug,
            title=title,
            hook=hook,
            content_preview=content[:CONTENT_PREVIEW_CHARS],
            score=score,
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:k]


def recall_topics_prompt(
    topics_dir: Path,
    query: str,
    k: int = 3,
) -> str:
    """关键词检索并拼装成 prompt 段落。

    Args:
        topics_dir: memory/topics/ 目录路径
        query: 搜索查询
        k: 返回的最大结果数

    Returns:
        拼装好的 markdown 段落，无匹配时返回空字符串
    """
    matches = recall_topics(topics_dir, query, k=k)
    if not matches:
        return ""

    lines = ["## Long-term Memory (相关记忆)"]
    for m in matches:
        lines.append(f"### {m.title} (`{m.slug}`)")
        lines.append(m.content_preview)
        lines.append("")

    return "\n".join(lines)
