# SPDX-License-Identifier: Apache-2.0
"""RecallEngine — unified recall interface across L1, L2, and artifact sources.

Three public modes:
  - recall_session_memory(): L1 keyword search on current session's summary.md
  - recall_long_term_memory(): L2 topic recall via LongTermMemory.recall_topics_prompt()
  - recall_all(): aggregate L1 + L2 + artifacts + cross-session L1 (future extension)

Path arguments are injected directly from InfraService; no .parent/.parent derivation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

# Defaults
DEFAULT_MAX_SESSIONS = 10
DEFAULT_MAX_ARTIFACTS = 50
DEFAULT_CONTENT_PREVIEW = 500

# Paths relative to a session directory
ARTIFACTS_DIR = "artifacts"
MEMORY_DIR = "memory"
SUMMARY_FILE = "summary.md"


# ── Data classes ─────────────────────────────────────────────────────────────────

@dataclass
class RecallEntry:
    """A single recall result entry."""

    source: str  # "L1:current" | "L1:{session_id}" | "L2:{slug}" | "Artifact:{id}"
    title: str
    content: str
    score: float = 0.0


@dataclass
class RecallResult:
    """Aggregated recall results with optional query for display formatting."""

    entries: list[RecallEntry] = field(default_factory=list)

    def to_display(self, query: str = "") -> str:
        """Format entries as readable text; returns "no context found" hint if empty."""
        if not self.entries:
            return f"No context found for: {query}"
        return "\n\n---\n\n".join(
            f"[{e.source}] {e.title}\n{e.content[:DEFAULT_CONTENT_PREVIEW]}"
            for e in self.entries
        )


# ── Scoring ──────────────────────────────────────────────────────────────────────

def _match_score(
    query: str,
    slug: str,
    title: str,
    hook: str,
    content: str,
) -> float:
    """Keyword matching score for a topic.

    | 匹配类型                  | 分数            |
    |--------------------------|-----------------|
    | token 完全匹配 slug/title | 1.0             |
    | token 完全匹配 hook       | 0.9             |
    | token 子串匹配 slug/title | 0.7             |
    | token 出现在 content     | 0.5             |
    | fuzzy match (ratio>=0.6) | 0.3 * similarity |
    | 无匹配                   | 0.0             |
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
        # 1. token 全匹配 slug/title
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

        # 5. fuzzy matching
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


# ── RecallEngine ─────────────────────────────────────────────────────────────────

class RecallEngine:
    """Unified recall engine — three public modes with structured output."""

    def __init__(
        self,
        session_memory: Any,  # SessionMemory instance
        long_term_memory: Any,  # LongTermMemory instance
        artifact_provider: Any,  # ArtifactProvider instance
        context_dir: Path,
        current_session_dir: Path,
    ) -> None:
        self._session_memory = session_memory
        self._long_term_memory = long_term_memory
        self._artifact_provider = artifact_provider
        self._context_dir = context_dir
        self._current_session_dir = current_session_dir

    # ── Public interfaces ────────────────────────────────────────────────────────

    def recall_session_memory(self, query: str) -> RecallResult:
        """L1: Keyword search on current session's summary.md.

        Wraps SessionMemory.recall() into RecallResult format.
        """
        if self._session_memory is None:
            return RecallResult()
        raw = self._session_memory.recall(query)
        if not raw or "No matches found" in raw:
            return RecallResult()

        sections = raw.split("\n# ")
        entries = []
        for section in sections:
            stripped = section.strip()
            if not stripped:
                continue
            # Extract section header as title
            lines = stripped.split("\n", 1)
            title = lines[0].strip().lstrip("# ")
            content = lines[1].strip() if len(lines) > 1 else stripped
            entries.append(RecallEntry(
                source="L1:current",
                title=title,
                content=content[:DEFAULT_CONTENT_PREVIEW],
                score=1.0,
            ))

        return RecallResult(entries=entries)

    def recall_long_term_memory(self, query: str) -> RecallResult:
        """L2: Topic recall via LongTermMemory.recall_topics_prompt().

        Converts structured TopicMatch results into RecallResult format.
        """
        if self._long_term_memory is None:
            return RecallResult()
        from infra.memory.recall import recall_topics

        topics_dir = self._long_term_memory.topics_dir
        matches = recall_topics(topics_dir, query, k=5)

        entries = [
            RecallEntry(
                source=f"L2:{m.slug}",
                title=m.title,
                content=m.content_preview[:DEFAULT_CONTENT_PREVIEW],
                score=m.score,
            )
            for m in matches
        ]

        return RecallResult(entries=entries)

    async def recall_all(self, query: str) -> RecallResult:
        """Aggregate: L1 + L2 + artifacts + cross-session L1.

        Runs all recall paths concurrently, merges by score descending.
        """
        l1_result = self.recall_session_memory(query)
        l2_result = self.recall_long_term_memory(query)

        artifact_result, cross_result = await asyncio.gather(
            self._recall_artifact(query),
            self._recall_l1_other_sessions(query),
        )

        all_entries = (
            l1_result.entries
            + l2_result.entries
            + artifact_result.entries
            + cross_result.entries
        )
        all_entries.sort(key=lambda e: e.score, reverse=True)

        return RecallResult(entries=all_entries)

    # ── Internal: Artifact keyword search ──────────────────────────────────────

    async def _recall_artifact(self, query: str) -> RecallResult:
        """Keyword search across artifact files in current session.

        Matches query tokens against filename and file content.
        """
        if self._artifact_provider is None:
            return RecallResult()
        artifacts_dir = self._current_session_dir / ARTIFACTS_DIR
        if not artifacts_dir.exists():
            return RecallResult()

        entries: list[RecallEntry] = []
        count = 0

        for path in sorted(artifacts_dir.glob("*.txt")):
            if count >= DEFAULT_MAX_ARTIFACTS:
                break
            count += 1

            try:
                content = await asyncio.to_thread(path.read_text, encoding="utf-8")
            except OSError:
                continue

            slug = path.stem  # tool_call_id
            score = _match_score(query, slug, slug, "", content)
            if score <= 0.0:
                continue

            entries.append(RecallEntry(
                source=f"Artifact:{slug}",
                title=slug,
                content=content[:DEFAULT_CONTENT_PREVIEW],
                score=score,
            ))

        entries.sort(key=lambda e: e.score, reverse=True)
        return RecallResult(entries=entries)

    # ── Internal: Cross-session L1 ───────────────────────────────────────────────

    async def _recall_l1_other_sessions(self, query: str) -> RecallResult:
        """Scan other sessions' summary.md files for keyword matches.

        Searches {context_dir}/sessions/*/memory/summary.md (excluding current session).
        """
        sessions_root = self._context_dir / "sessions"
        if not sessions_root.exists():
            return RecallResult()

        entries: list[RecallEntry] = []
        count = 0
        current_session_id = self._current_session_dir.name

        for session_path in sorted(sessions_root.iterdir()):
            if not session_path.is_dir():
                continue
            if session_path.name == current_session_id:
                continue
            if count >= DEFAULT_MAX_SESSIONS:
                break
            count += 1

            summary_path = session_path / MEMORY_DIR / SUMMARY_FILE
            if not summary_path.exists():
                continue

            try:
                content = await asyncio.to_thread(summary_path.read_text, encoding="utf-8")
            except OSError:
                continue

            # Extract title from first heading
            lines = content.split("\n")
            title = lines[0].strip().lstrip("# ") if lines else session_path.name

            # Score against session id as slug
            score = _match_score(
                query,
                slug=session_path.name,
                title=title,
                hook="",
                content=content,
            )
            if score <= 0.0:
                continue

            entries.append(RecallEntry(
                source=f"L1:{session_path.name}",
                title=title,
                content=content[:DEFAULT_CONTENT_PREVIEW],
                score=score,
            ))

        entries.sort(key=lambda e: e.score, reverse=True)
        return RecallResult(entries=entries)
