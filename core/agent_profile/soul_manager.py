# SPDX-License-Identifier: Apache-2.0
"""SoulManager — SOUL.md 文件 I/O + TTL 缓存。

对称于 UserManager，管理 SOUL.md 的读写和增量更新。
热路径同步模块，供 agent 实时加载身份定义。
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from core.agent_profile.constants import CACHE_TTL_SECONDS, SOUL_FILE
from core.agent_profile.defaults import _DEFAULT_SOUL_TEMPLATE
from core.agent_profile.utils import _format_soul, _parse_soul
from utils.logger import get_logger

logger = get_logger(__name__)


class SoulManager:
    """SOUL.md read/write with TTL cache. 对称于 UserManager。"""

    def __init__(self) -> None:
        self._cache: dict | None = None
        self._cache_at: float = 0
        self._profile_dir: Path | None = None

    def _get_profile_dir(self) -> Path:
        if self._profile_dir is not None:
            return self._profile_dir
        from middleware.config import g_config

        base = g_config.paths.data_dir
        if not base:
            base = Path.home() / "memento_s"
        self._profile_dir = Path(base) / "agent_profile"
        return self._profile_dir

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

    def backup_before_write(self, filename: str) -> str | None:
        """Backup a profile file before it is overwritten. Delegates to daemon backup."""
        try:
            from daemon.agent_profile.backup import backup_file
            return backup_file(filename)
        except Exception:
            return None

    def load(self) -> dict:
        """Load SOUL.md with TTL cache. Falls back to default template on parse failure."""
        now = time.monotonic()
        if self._cache is not None and (now - self._cache_at) < CACHE_TTL_SECONDS:
            return self._cache

        raw = self._read_file(SOUL_FILE)
        self._cache = _parse_soul(raw or "")
        self._cache_at = now

        if not raw:
            self._write_file_atomic(SOUL_FILE, _format_soul(self._cache))
        return self._cache

    def save(self, data: dict) -> None:
        """Write SOUL.md atomically and update cache."""
        self.backup_before_write(SOUL_FILE)
        self._write_file_atomic(SOUL_FILE, _format_soul(data))
        self._cache = data
        self._cache_at = time.monotonic()

    def merge_update(self, updates: dict) -> bool:
        """Merge partial field updates into SOUL.md. Returns True if anything changed."""
        current = self.load()
        changed = False

        for key, new_values in updates.items():
            if key not in current:
                continue

            if isinstance(new_values, list) and isinstance(current[key], list):
                existing_set = {t.strip() for t in current[key] if t.strip()}
                for item in new_values:
                    stripped = item.strip()
                    if stripped and stripped not in existing_set:
                        current[key].append(stripped)
                        changed = True

            elif key in ("vibe", "role") and isinstance(new_values, str):
                if new_values.strip() and new_values.strip() != current.get(key, "").strip():
                    current[key] = new_values.strip()
                    changed = True

        if changed:
            bp = self.backup_before_write(SOUL_FILE)
            if bp:
                logger.debug("[SoulManager] backed up SOUL.md -> {}", bp)
            self.save(current)
        return changed

    def ensure_exists(self) -> None:
        """Ensure SOUL.md exists with default template if missing."""
        soul_path = self._get_profile_dir() / SOUL_FILE
        if not soul_path.exists():
            self.save(_parse_soul(""))

    def format_current(self) -> str:
        """Return current SOUL.md as formatted string (for LLM prompts)."""
        return _format_soul(self.load())
