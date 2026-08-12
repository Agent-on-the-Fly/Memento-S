# SPDX-License-Identifier: Apache-2.0
"""SessionContext — unified session identity and path container.

Scope: core/ module only. Callers outside core/ (server/, cli/, gui/) pass
session_id strings; SessionContext is created *inside* core/memento_s/agent.py
and flows through core/context/ and core/memento_s/ layers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SessionContext:
    """Immutable session identity and path container.

    Created once by ``create()`` at the entry of ``reply_stream`` and passed
    downstream.  No field may be mutated after creation.
    """

    session_id: str
    session_dir: Path
    date_dir: Path
    run_dir: Path
    user_id: Optional[str] = None
    channel: Optional[str] = None

    @classmethod
    def create(
        cls,
        session_id: str,
        base_dir: Path,
        *,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> SessionContext:
        """Factory — constructs paths from ``session_id`` and ``base_dir``.

        Uses sessions-directory convention: ``{base_dir}/sessions/{session_id}``.
        """
        date_str = datetime.now().strftime("%Y%m%d")
        date_dir = base_dir / date_str
        run_hash = hashlib.md5(session_id.encode()).hexdigest()[:8]
        run_dir = date_dir / run_hash
        session_dir = base_dir / "sessions" / session_id
        return cls(
            session_id=session_id,
            session_dir=session_dir,
            date_dir=date_dir,
            run_dir=run_dir,
            user_id=user_id,
            channel=channel,
        )

    @classmethod
    def from_parent(
        cls,
        session_dir: Path,
        date_dir: Path,
        run_dir: Path,
        session_id: str,
        *,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> SessionContext:
        """Factory — for cases where the parent layer (e.g. InfraService)
        already computed the session/date/run directories.

        All paths must be pre-validated (mkdir done by the caller).
        """
        return cls(
            session_id=session_id,
            session_dir=session_dir,
            date_dir=date_dir,
            run_dir=run_dir,
            user_id=user_id,
            channel=channel,
        )

    @property
    def run_hash(self) -> str:
        """Short hash suffix of run_dir (MD5[:8] of session_id)."""
        return self.run_dir.name
