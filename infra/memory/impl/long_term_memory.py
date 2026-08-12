# SPDX-License-Identifier: Apache-2.0
"""LongTermMemory — indexed topic files.

MEMORY.md serves as the entrypoint index.
Individual topic-slug.md files are stored in the topics/ subdirectory.
_staging.md accumulates session knowledge for consolidation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
TOPICS_DIR = "topics"
INDEX_NAME = "MEMORY.md"
STAGING_NAME = "_staging.md"
STAGING_LOCK_NAME = ".staging.lock"

MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000


@dataclass
class EntrypointTruncation:
    content: str
    was_truncated: bool
    truncation_reason: str


def truncate_entrypoint_content(
    raw: str,
    max_lines: int = MAX_ENTRYPOINT_LINES,
    max_bytes: int = MAX_ENTRYPOINT_BYTES,
) -> EntrypointTruncation:
    """行/字节上限截断。"""
    lines = raw.strip().split("\n")
    was_truncated = False
    reason = ""

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        was_truncated = True
        reason = "line_limit"

    content = "\n".join(lines)
    if len(content.encode("utf-8")) > max_bytes:
        content = content.encode("utf-8")[:max_bytes].decode(errors="ignore")
        last_nl = content.rfind("\n")
        if last_nl > 0:
            content = content[:last_nl]
        was_truncated = True
        reason = "byte_limit"

    return EntrypointTruncation(
        content=content,
        was_truncated=was_truncated,
        truncation_reason=reason,
    )


class LongTermMemory:
    """Indexed topic files for durable knowledge storage.

    MEMORY.md serves as the entrypoint index.
    Individual topic-slug.md files are stored in the topics/ subdirectory.
    _staging.md accumulates session knowledge for consolidation.
    """

    def __init__(self, memory_dir: Path, model: str = "") -> None:
        self._dir = memory_dir
        self._topics_dir: Path = memory_dir / TOPICS_DIR
        self._index_path: Path = memory_dir / INDEX_NAME
        self._staging_path: Path = memory_dir / STAGING_NAME
        self._model = model
        self._dir.mkdir(parents=True, exist_ok=True)
        self._topics_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载 ----

    def load_memory_prompt(self) -> str:
        """加载 MEMORY.md 索引 (截断到 200行/25KB)。"""
        if not self._index_path.exists():
            return ""
        try:
            content = self._index_path.read_text(encoding="utf-8")
        except OSError:
            return ""

        if not content.strip():
            return ""

        truncated = truncate_entrypoint_content(content)
        return f"## Long-term Memory Index\n{truncated.content}"

    def read_topic(self, slug: str) -> str:
        """读取指定主题文件全文。"""
        path = self._topics_dir / f"{slug}.md"
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def list_topics(self) -> list[dict[str, Any]]:
        """列出所有主题 (slug, title, size)。"""
        topics: list[dict[str, Any]] = []
        for path in self._topics_dir.glob("*.md"):
            slug = path.stem
            try:
                content = path.read_text(encoding="utf-8")
                first_line = content.split("\n")[0].strip().lstrip("# ")
                topics.append({
                    "slug": slug,
                    "title": first_line or slug,
                    "size": len(content),
                })
            except OSError:
                continue
        return topics

    def list_topics_with_content(self) -> list[dict[str, str]]:
        """列出所有主题 + 全文 (供 consolidation 使用)。"""
        result: list[dict[str, str]] = []
        for path in self._topics_dir.glob("*.md"):
            try:
                content = path.read_text(encoding="utf-8")
                result.append({"slug": path.stem, "content": content})
            except OSError:
                continue
        return result

    # ---- 写入 ----

    async def accumulate_session(self, session_memory_content: str) -> None:
        """Session 结束时追加到 staging 区 (零 LLM)。"""
        if not session_memory_content.strip():
            return

        try:
            with open(self._staging_path, "a", encoding="utf-8") as f:
                f.write(f"\n---\n## Session {datetime.now():%Y-%m-%d %H:%M}\n")
                f.write(session_memory_content)
                f.write("\n")
            logger.debug("LongTermMemory: accumulated session to _staging.md")
        except OSError:
            logger.opt(exception=True).warning("Failed to write to _staging.md")

    # ---- 兼容别名 (旧的调用方) ----

    async def absorb_session(self, session_memory_content: str) -> None:
        """[Deprecated] 使用 accumulate_session 代替。"""
        await self.accumulate_session(session_memory_content)

    async def write_topic(
        self, slug: str, title: str, content: str, hook: str
    ) -> None:
        """写入/更新主题文件 (索引更新由 ResultApplier 统一负责)。"""
        topic_path = self._topics_dir / f"{slug}.md"
        try:
            topic_path.write_text(f"# {title}\n\n{content}", encoding="utf-8")
        except OSError:
            logger.opt(exception=True).warning("Failed to write topic {}", slug)

    # ---- Dream 整合入口 ----

    def get_index_content(self) -> str:
        """读取 MEMORY.md 全文。"""
        if not self._index_path.exists():
            return ""
        try:
            return self._index_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def get_staging_content(self) -> str:
        """读取 _staging.md 全文。"""
        if not self._staging_path.exists():
            return ""
        try:
            return self._staging_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    # ---- 兼容别名 ----
    get_pending_content = get_staging_content
    """[Deprecated] 使用 get_staging_content 代替。"""

    def apply_consolidation_result(self, result: dict[str, Any]) -> None:
        """应用整合结果 (更新 topics + index)。

        注意：此方法仅在 engine 外部调用时使用。
        engine 内部通过 ResultApplier 写入。
        """
        for topic in result.get("updated_topics", []):
            slug = topic.get("slug", "")
            if not slug:
                continue
            path = self._topics_dir / f"{slug}.md"
            try:
                path.write_text(topic.get("content", ""), encoding="utf-8")
            except OSError:
                logger.warning("Failed to write updated topic {}", slug)

        for topic in result.get("new_topics", []):
            slug = topic.get("slug", "")
            if not slug:
                continue
            path = self._topics_dir / f"{slug}.md"
            try:
                title = topic.get("title", slug)
                path.write_text(f"# {title}\n\n{topic.get('content', '')}", encoding="utf-8")
            except OSError:
                logger.warning("Failed to write new topic {}", slug)

        for slug in result.get("deleted_topics", []):
            path = self._topics_dir / f"{slug}.md"
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    logger.warning("Failed to delete topic {}", slug)

        index_content = result.get("index_content", "")
        if index_content:
            try:
                self._index_path.write_text(index_content, encoding="utf-8")
            except OSError:
                logger.warning("Failed to update MEMORY.md index")

    def clear_staging(self) -> None:
        """清空 _staging.md (整合完成后调用)。"""
        try:
            if self._staging_path.exists():
                self._staging_path.write_text("", encoding="utf-8")
        except OSError:
            logger.opt(exception=True).warning("Failed to clear _staging.md")

    # ---- 兼容别名 ----
    clear_pending = clear_staging
    """[Deprecated] 使用 clear_staging 代替。"""

    @property
    def topics_dir(self) -> Path:
        """返回 topics/ 子目录路径。"""
        return self._topics_dir

    @property
    def staging_path(self) -> Path:
        """[Deprecated] 兼容别名，指向 _staging.md。"""
        return self._staging_path

    def recall_topics_prompt(self, query: str, k: int = 3) -> str:
        """关键词检索相关 topic，拼装成 prompt 段落。

        由 TopicRecallEngine 提供实现，延迟导入避免循环依赖。
        """
        from infra.memory.recall import recall_topics_prompt as _recall
        return _recall(self._topics_dir, query, k=k)
