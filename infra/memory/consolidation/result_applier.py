# SPDX-License-Identifier: Apache-2.0
"""Apply consolidation results to topic files and index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


class ResultApplier:
    """Apply consolidation result to LongTermMemory storage."""

    def __init__(self, memory_dir: Path) -> None:
        self._dir = memory_dir
        self._topics_dir: Path = memory_dir / "topics"
        self._topics_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = memory_dir / "MEMORY.md"

    def apply(self, result: dict[str, Any]) -> int:
        """Apply consolidation result, return number of topics changed."""
        changed = 0

        for topic in result.get("updated_topics", []):
            slug = topic.get("slug", "")
            if not slug:
                continue
            path = self._topics_dir / f"{slug}.md"
            try:
                path.write_text(topic.get("content", ""), encoding="utf-8")
                changed += 1
                logger.debug("ResultApplier: updated topic {}", slug)
            except OSError:
                logger.warning("ResultApplier: failed to write updated topic {}", slug)

        for topic in result.get("new_topics", []):
            slug = topic.get("slug", "")
            if not slug:
                continue
            path = self._topics_dir / f"{slug}.md"
            if path.exists():
                continue
            try:
                title = topic.get("title", slug)
                path.write_text(f"# {title}\n\n{topic.get('content', '')}", encoding="utf-8")
                changed += 1
                logger.debug("ResultApplier: created new topic {}", slug)
            except OSError:
                logger.warning("ResultApplier: failed to write new topic {}", slug)

        for slug in result.get("deleted_topics", []):
            path = self._topics_dir / f"{slug}.md"
            if path.exists():
                try:
                    path.unlink()
                    changed += 1
                    logger.debug("ResultApplier: deleted topic {}", slug)
                except OSError:
                    logger.warning("ResultApplier: failed to delete topic {}", slug)

        index_content = result.get("index_content", "")
        if index_content:
            self._update_index_from_result(result)

        return changed

    def _update_index_from_result(self, result: dict[str, Any]) -> None:
        """Update MEMORY.md index from consolidation result."""
        index_content = result.get("index_content", "")
        if not index_content:
            return

        try:
            self._index_path.write_text(index_content, encoding="utf-8")
            logger.debug("ResultApplier: updated MEMORY.md index")
        except OSError:
            logger.warning("ResultApplier: failed to update MEMORY.md index")

    def update_index_entry(self, slug: str, title: str, hook: str) -> None:
        """Update a single entry in MEMORY.md index."""
        entry = f"- [{title}]({slug}.md) — {hook}"

        if self._index_path.exists():
            try:
                content = self._index_path.read_text(encoding="utf-8")
            except OSError:
                content = "# Project Memory Index\n"
        else:
            content = "# Project Memory Index\n"

        lines = content.split("\n")
        updated = False
        for i, line in enumerate(lines):
            if f"({slug}.md)" in line:
                lines[i] = entry
                updated = True
                break

        if not updated:
            lines.append(entry)

        try:
            self._index_path.write_text("\n".join(lines), encoding="utf-8")
        except OSError:
            logger.opt(exception=True).warning("ResultApplier: failed to update index entry {}", slug)
