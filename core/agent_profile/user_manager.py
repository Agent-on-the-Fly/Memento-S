# SPDX-License-Identifier: Apache-2.0
"""UserManager — USER.md 文件 I/O + TTL 缓存。

对称于 SoulManager，管理 USER.md 的读写和追加。
热路径同步模块，供 agent 实时查询。
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from core.agent_profile.constants import CACHE_TTL_SECONDS, USER_FILE
from core.agent_profile.defaults import _DEFAULT_USER_TEMPLATE


class UserManager:
    """USER.md I/O + profile building. 对称于 SoulManager。"""

    def __init__(self) -> None:
        self._user_context_cache: str | None = None
        self._user_cache_at: float = 0
        self._profile_dir: Path | None = None

    # ── path ───────────────────────────────────────────────────────────

    def _get_profile_dir(self) -> Path:
        if self._profile_dir is not None:
            return self._profile_dir
        from middleware.config import g_config

        base = g_config.paths.data_dir
        if not base:
            base = Path.home() / "memento_s"
        self._profile_dir = Path(base) / "agent_profile"
        return self._profile_dir

    # ── internal helpers ───────────────────────────────────────────────

    def _backup_before_write(self) -> str | None:
        """Backup USER.md to backup/ before overwriting. Returns backup path or None."""
        try:
            from daemon.agent_profile.backup import backup_file
            return backup_file(USER_FILE)
        except Exception:
            return None

    def _read_file(self, filename: str) -> str | None:
        path = self._get_profile_dir() / filename
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _write_file_atomic(self, filename: str, content: str) -> None:
        dir_ = self._get_profile_dir()
        path = dir_ / filename
        dir_.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(dir_), prefix="._", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, str(path))
        except BaseException:
            os.unlink(tmp)
            raise

    # ── USER.md ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_sections(content: str) -> dict[str, list[str]]:
        """Parse USER.md into {section_name: [fact, ...]}. Strips placeholder lines."""
        sections: dict[str, list[str]] = {}
        section_re = re.compile(r"^## (.+)$", re.MULTILINE)
        matches = list(section_re.finditer(content))
        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[start:end]
            facts = []
            for line in body.splitlines():
                stripped = line.strip()
                if stripped:
                    facts.append(stripped)
            sections[title] = facts
        return sections

    def load_context(self) -> str:
        """Load USER.md with 15-minute TTL cache. Returns sectioned context for LLM."""
        now = time.monotonic()
        if (self._user_context_cache is not None
                and (now - self._user_cache_at) < CACHE_TTL_SECONDS):
            return self._user_context_cache

        raw = self._read_file(USER_FILE)
        if raw:
            sections = self._parse_sections(raw)
            has_any = any(sections.values())
            if has_any:
                lines = ["\n## User Context"]
                for title, facts in sections.items():
                    lines.append(f"\n### {title}")
                    for f in facts:
                        lines.append(f"- {f}")
                self._user_context_cache = "\n".join(lines)
            else:
                self._backup_before_write()
                self._write_file_atomic(USER_FILE, _DEFAULT_USER_TEMPLATE)
                self._user_context_cache = ""
        else:
            self._backup_before_write()
            self._write_file_atomic(USER_FILE, _DEFAULT_USER_TEMPLATE)
            self._user_context_cache = ""
        self._user_cache_at = now
        return self._user_context_cache

    def append_facts(self, facts_by_section: dict[str, list[str]]) -> bool:
        """Merge facts into USER.md by section. Returns True if new facts were added."""
        if not facts_by_section:
            return False

        path = self._get_profile_dir() / USER_FILE
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if not raw:
            return False

        sections = self._parse_sections(raw)
        new_added = False

        for section_title, new_facts in facts_by_section.items():
            existing = set(sections.get(section_title, []))
            to_add = [f for f in new_facts if f.strip() and f.strip() not in existing]
            if to_add:
                sections[section_title] = existing | set(to_add)
                new_added = True

        if not new_added:
            return False

        lines = []
        section_order = [
            "Identity & Preferences",
            "Communication Style",
            "Expertise & Background",
            "Current Goals & Context",
            "Agreements & Corrections",
        ]
        for title in section_order:
            facts = sections.get(title, [])
            lines.append(f"## {title}")
            for f in facts:
                lines.append(f)
            lines.append("")
        content = "\n".join(lines).strip() + "\n"
        self._backup_before_write()
        self._write_file_atomic(USER_FILE, content)
        self._user_context_cache = None
        self._user_cache_at = 0
        return True

    # ── lifecycle ─────────────────────────────────────────────────────

    def ensure_exists(self) -> None:
        """Ensure USER.md exists with default template if missing."""
        user_path = self._get_profile_dir() / USER_FILE
        if not user_path.exists():
            self._write_file_atomic(USER_FILE, _DEFAULT_USER_TEMPLATE)
