# SPDX-License-Identifier: Apache-2.0
"""Backup utilities for agent profile files.

Backup strategy:
  - SOUL.md and USER.md are backed up to backup/ before any write operation
  - Backups are timestamped: soul_YYYY-MM-DDTDD-HH-MM-SS.md or user_YYYY-MM-DDTDD-HH-MM-SS.md
  - Backup directory lives under agent_profile/ (same level as SOUL.md / USER.md)
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

_BACKUP_PREFIXES = {"SOUL.md": "soul", "USER.md": "user"}


def get_profile_dir() -> Path:
    """Return the agent_profile directory path."""
    from middleware.config import g_config

    base = g_config.paths.data_dir
    if not base:
        base = Path.home() / "memento_s"
    return Path(base) / "agent_profile"


def backup_file(filename: str) -> str | None:
    """Backup a SOUL.md or USER.md file before it is overwritten.

    Returns the backup path on success, None on failure.
    """
    if filename not in _BACKUP_PREFIXES:
        return None

    prefix = _BACKUP_PREFIXES[filename]
    dir_ = get_profile_dir()
    src = dir_ / filename
    if not src.exists():
        return None

    backup_dir = dir_ / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    backup_name = f"{prefix}_{ts}.md"
    backup_path = backup_dir / backup_name
    shutil.copy2(src, backup_path)
    logger.debug("[backup] {} -> {}", filename, backup_path)
    return str(backup_path)
