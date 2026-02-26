"""Execution logging utilities extracted from agent.py.

Provides structured JSONL event logging for the Memento-S agent.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import EXEC_LOG_ENABLED, EXEC_LOG_DIR, EXEC_LOG_MAX_CHARS

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_EXEC_LOG_SESSION_ID: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
_EXEC_LOG_FILE: Path | None = None
_EXEC_LOG_FAILED: bool = False
_EXEC_LOG_LOCK: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate_for_log(value: str) -> str:
    """Truncate a string value for logging if EXEC_LOG_MAX_CHARS is set."""
    if EXEC_LOG_MAX_CHARS <= 0:
        return value
    if len(value) <= EXEC_LOG_MAX_CHARS:
        return value
    marker = f"...[truncated:{len(value)}]"
    keep = max(0, EXEC_LOG_MAX_CHARS - len(marker))
    return value[:keep] + marker


def _prepare_for_log(value: Any) -> Any:
    """Recursively prepare a value for JSON serialization in logs."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_for_log(value)
    if isinstance(value, bytes):
        return _truncate_for_log(value.decode("utf-8", errors="replace"))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _prepare_for_log(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_prepare_for_log(v) for v in value]
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        return _truncate_for_log(repr(value))


def _ensure_exec_log_file() -> Path | None:
    """Lazily initialize (and cache) the execution log file path."""
    global _EXEC_LOG_FILE, _EXEC_LOG_FAILED
    if not EXEC_LOG_ENABLED or _EXEC_LOG_FAILED:
        return None
    if _EXEC_LOG_FILE is not None:
        return _EXEC_LOG_FILE
    try:
        EXEC_LOG_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"agent-{_EXEC_LOG_SESSION_ID}-pid{os.getpid()}.jsonl"
        _EXEC_LOG_FILE = (EXEC_LOG_DIR / filename).resolve()
        return _EXEC_LOG_FILE
    except Exception as exc:
        _EXEC_LOG_FAILED = True
        print(f"[warn] failed to initialize execution log file: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_exec_log_path() -> str | None:
    """Return the path to the current session's execution log file, or None."""
    path = _ensure_exec_log_file()
    return str(path) if path else None


def log_event(event: str, **fields: Any) -> None:
    """Append a structured JSON event to the execution log.

    Each record includes a UTC timestamp, session ID, event name,
    and any additional keyword-argument fields (recursively sanitised
    via ``_prepare_for_log``).
    """
    if not EXEC_LOG_ENABLED:
        return
    path = _ensure_exec_log_file()
    if path is None:
        return
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": _EXEC_LOG_SESSION_ID,
        "event": str(event or "unknown"),
    }
    record.update({k: _prepare_for_log(v) for k, v in fields.items()})
    line = json.dumps(record, ensure_ascii=False)
    try:
        with _EXEC_LOG_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        global _EXEC_LOG_FAILED
        _EXEC_LOG_FAILED = True
        print(f"[warn] failed to write execution log: {exc}")
