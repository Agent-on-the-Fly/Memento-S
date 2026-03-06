
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from core.agent.session_manager import SessionManager, generate_session_id

logger = logging.getLogger(__name__)



@dataclass
class Conversation:

    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    model: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def is_empty(self) -> bool:
        return len(self.messages) == 0


@dataclass
class ConversationSummary:

    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    model: str = ""

    @property
    def display_date(self) -> str:
        try:
            dt = datetime.fromisoformat(self.updated_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError, AttributeError):
            return self.updated_at[:19] if self.updated_at else "(unknown)"

    @property
    def updated_datetime(self) -> datetime:
        try:
            dt = datetime.fromisoformat(self.updated_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone()
        except (ValueError, TypeError, AttributeError):
            return datetime.min.replace(tzinfo=timezone.utc).astimezone()



class ConversationStore:

    def __init__(self, session_manager: SessionManager) -> None:
        self._sm = session_manager
        self._lock = threading.Lock()


    def create(self, *, model: str = "", title: str = "") -> Conversation:
        session_id = generate_session_id()
        title = title or "New Conversation"
        now = self._utc_now_iso()
        session_data = {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "total_tokens": 0,
            "model": model,
            "metadata": {},
        }
        with self._lock:
            self._sm._save_session(session_id, session_data)
        return Conversation(
            id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            model=model,
        )

    def save(self, conversation: Conversation) -> None:
        with self._lock:
            session = self._sm.get_session(conversation.id)
            if session is None:
                now = self._utc_now_iso()
                session = {
                    "id": conversation.id,
                    "title": conversation.title,
                    "created_at": conversation.created_at or now,
                    "updated_at": now,
                    "messages": conversation.messages,
                    "total_tokens": conversation.total_tokens,
                    "model": conversation.model,
                    "metadata": conversation.metadata,
                }
            else:
                session["messages"] = conversation.messages
                session["total_tokens"] = conversation.total_tokens
                session["model"] = conversation.model
                session["title"] = conversation.title
                session["metadata"] = conversation.metadata
                session["updated_at"] = self._utc_now_iso()
            self._sm._save_session(conversation.id, session)

    def load(self, conversation_id: str) -> Optional[Conversation]:
        with self._lock:
            session = self._sm.get_session(conversation_id)
        if session is None:
            logger.warning("Conversation not found: %s", conversation_id)
            return None
        return self._session_to_conversation(session)


    def list_all(self) -> list[ConversationSummary]:
        with self._lock:
            session_ids = self._sm.list_sessions()
        summaries: list[ConversationSummary] = []
        for sid in session_ids:
            session = self._sm.get_session(sid)
            if session:
                summaries.append(ConversationSummary(
                    id=session.get("id", sid),
                    title=session.get("title", "Untitled"),
                    created_at=session.get("created_at", ""),
                    updated_at=session.get("updated_at", ""),
                    message_count=len(session.get("messages", [])),
                    model=session.get("model", ""),
                ))
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries

    @staticmethod
    def group_by_time_period(
        summaries: list[ConversationSummary],
    ) -> list[tuple[str, list[ConversationSummary]]]:
        today = datetime.now().astimezone().date()
        yesterday = today - timedelta(days=1)
        week_lower = today - timedelta(days=6)
        month_lower = today - timedelta(days=30)

        buckets: dict[str, list[ConversationSummary]] = {}
        month_keys: list[str] = []

        for s in summaries:
            d = s.updated_datetime.date()
            if d == today:
                label = ""
            elif d == yesterday:
                label = ""
            elif d >= week_lower:
                label = "7"
            elif d >= month_lower:
                label = "30"
            else:
                label = f"{d.year}-{d.month:02d}"
                if label not in buckets:
                    month_keys.append(label)

            buckets.setdefault(label, []).append(s)

        ordered_labels = ["", "", "7", "30"]
        month_keys.sort(reverse=True)
        ordered_labels.extend(month_keys)

        return [(lbl, buckets[lbl]) for lbl in ordered_labels if lbl in buckets]

    def get_latest_id(self) -> Optional[str]:
        summaries = self.list_all()
        return summaries[0].id if summaries else None

    def exists(self, conversation_id: str) -> bool:
        return self._sm.get_session(conversation_id) is not None

    @property
    def count(self) -> int:
        return len(self._sm.list_sessions())


    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            return self._sm.delete_session(conversation_id)

    def batch_delete(self, conversation_ids: list[str]) -> dict[str, bool]:
        with self._lock:
            return self._sm.delete_sessions(conversation_ids)

    def update_title(self, conversation_id: str, title: str) -> bool:
        with self._lock:
            return self._sm.update_session(conversation_id, title=title.strip() or None)


    @staticmethod
    def auto_title(messages: list[dict], max_length: int = 50) -> str:
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    text = content.strip().replace("\n", " ").strip()
                    if len(text) > max_length:
                        text = text[: max_length - 3].rstrip() + "..."
                    return text
        return "New Conversation"

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _session_to_conversation(session: dict[str, Any]) -> Conversation:
        return Conversation(
            id=session.get("id", ""),
            title=session.get("title", "Untitled"),
            created_at=session.get("created_at", ""),
            updated_at=session.get("updated_at", ""),
            messages=session.get("messages", []),
            total_tokens=session.get("total_tokens", 0),
            model=session.get("model", ""),
            metadata=session.get("metadata", {}),
        )
