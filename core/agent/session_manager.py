
from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import g_settings
from core.config.logging import get_logger

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _approx_tokens_from_content(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = ""
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                text += part.get("text", "")
            elif isinstance(part, str):
                text += part
        if not text:
            return 0
    else:
        return 0
    if not text:
        return 0
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f")
    other = len(text) - cjk
    return max(0, int(other / 4 + cjk / 1.5))


def _total_tokens_from_messages(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        total += _approx_tokens_from_content(m.get("content"))
    return total


_ID_CHARS: str = string.ascii_lowercase + string.digits
_ID_LENGTH: int = 8
_LEGACY_ID_PATTERN: re.Pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_")


def generate_session_id(existing_ids: set[str] | None = None) -> str:
    for _ in range(100):
        candidate = "".join(secrets.choice(_ID_CHARS) for _ in range(_ID_LENGTH))
        if existing_ids is None or candidate not in existing_ids:
            return candidate
    raise RuntimeError("Failed to generate unique session ID after 100 attempts")


def _default_session(session_id: str, title: str = "New chat") -> dict[str, Any]:
    now = _now_iso()
    return {
        "id": session_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "total_tokens": 0,
        "model": "",
        "metadata": {},
    }


class SessionManager:

    def __init__(
        self,
        workspace: Path | None = None,
        conversations_dir: str | None = 'conversations',
    ) -> None:
        self.workspace = workspace or g_settings.workspace_path
        if conversations_dir is not None:
            self._dir = self.workspace / conversations_dir
        else:
            self._dir = g_settings.conversations_path
        self._dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_sessions()


    def _migrate_legacy_sessions(self) -> None:
        existing_ids: set[str] = set()
        legacy_files: list[Path] = []

        for path in self._dir.glob("*.json"):
            stem = path.stem
            if _LEGACY_ID_PATTERN.match(stem):
                legacy_files.append(path)
            else:
                existing_ids.add(stem)

        for path in legacy_files:
            old_id = path.stem
            try:
                new_id = generate_session_id(existing_ids=existing_ids)
                existing_ids.add(new_id)

                data = json.loads(path.read_text(encoding="utf-8"))
                data["id"] = new_id
                new_path = self._dir / f"{new_id}.json"
                new_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                path.unlink()
                logger.info("Migrated session %s -> %s", old_id, new_id)
            except Exception as exc:
                logger.warning("Failed to migrate session %s: %s", old_id, exc)

    def _path(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self._dir / f"{safe or 'default'}.json"

    def _load_session(self, session_id: str) -> dict[str, Any] | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "messages" not in data:
                logger.warning("Invalid session structure in %s", path)
                return None
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load session %s: %s", path, e)
            return None

    def _save_session(self, session_id: str, session: dict[str, Any]) -> None:
        path = self._path(session_id)
        path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self._load_session(session_id)

    async def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        def _do() -> list[dict[str, Any]]:
            session = self._load_session(session_id)
            if session is None:
                return []
            return session.get("messages", [])

        return await asyncio.to_thread(_do)

    def append_message(self, session_id: str, message: dict[str, Any]) -> None:
        session = self._load_session(session_id)
        if session is None:
            title = "New chat"
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                title = (message["content"].strip() or title)[:80]
            session = _default_session(session_id, title=title)
        session.setdefault("messages", []).append(message)
        session["total_tokens"] = _total_tokens_from_messages(session["messages"])
        session["updated_at"] = _now_iso()
        self._save_session(session_id, session)

    async def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        def _do() -> None:
            session = self._load_session(session_id)
            if session is None:
                session = _default_session(session_id)
            session["messages"] = messages
            session["total_tokens"] = _total_tokens_from_messages(messages)
            session["updated_at"] = _now_iso()
            self._save_session(session_id, session)

        await asyncio.to_thread(_do)

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        model: str | None = None,
        total_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        session = self._load_session(session_id)
        if session is None:
            return False
        if title is not None:
            session["title"] = title
        if model is not None:
            session["model"] = model
        if total_tokens is not None:
            session["total_tokens"] = total_tokens
        if metadata is not None:
            session["metadata"] = metadata
        session["updated_at"] = _now_iso()
        self._save_session(session_id, session)
        return True

    def list_sessions(self) -> list[str]:
        if not self._dir.exists():
            return []
        return [p.stem for p in self._dir.glob("*.json")]

    def delete_session(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def delete_sessions(
        self, session_ids: list[str]
    ) -> dict[str, bool]:
        if len(session_ids) > 100:
            raise ValueError(
                f" 100  {len(session_ids)} "
            )
        results: dict[str, bool] = {}
        for sid in session_ids:
            try:
                results[sid] = self.delete_session(sid)
            except Exception as exc:
                logger.warning("Failed to delete session %s: %s", sid, exc)
                results[sid] = False
        return results
