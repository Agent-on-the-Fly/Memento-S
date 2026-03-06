
from __future__ import annotations

import json
import logging
import os
import platform
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_GLOBAL: Optional["InteractionLogger"] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _looks_sensitive_key(key: str) -> bool:
    k = (key or "").lower()
    if k.endswith("_key") or "api_key" in k or "apikey" in k:
        return True
    if any(marker in k for marker in ("password", "secret", "private", "bearer", "authorization")):
        return True
    if k in {"token", "access_token", "refresh_token", "id_token"}:
        return True
    if k.endswith("_token") or k.endswith("token"):
        return True
    if k in {"claude_api_key", "openrouter_api_key"}:
        return True
    return False


def _redact_str(value: str) -> str:
    s = value.strip()
    if not s:
        return s
    if len(s) <= 8:
        return "*" * len(s)
    return f"{s[:4]}…{s[-2:]}"


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if _looks_sensitive_key(str(k)):
                out[str(k)] = _redact_str(str(v)) if v is not None else None
            else:
                out[str(k)] = _redact(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_redact(x) for x in obj]
    if isinstance(obj, str):
        lowered = obj.lower()
        if lowered.startswith("sk-") or lowered.startswith("rk-") or "bearer " in lowered:
            return _redact_str(obj)
        return obj
    return obj


class _JsonlFormatter(logging.Formatter):
    _STANDARD_ATTRS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _utc_now_iso(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "pid": record.process,
            "thread": record.threadName,
            "module": record.module,
        }

        extra = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self._STANDARD_ATTRS and not k.startswith("_")
        }
        if extra:
            payload.update(_redact(extra))

        if record.exc_info:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info)).strip()

        return json.dumps(payload, ensure_ascii=False)


class _EnsureExtrasFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "event"):
            record.event = "log"
        if not hasattr(record, "session_id"):
            record.session_id = "unknown"
        return True


_RESERVED_EXTRA_KEYS = set(_JsonlFormatter._STANDARD_ATTRS) | {"message", "asctime"}


def _sanitize_extra_fields(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    key_map: dict[str, str] = {}

    for key, value in fields.items():
        if not isinstance(key, str) or not key:
            safe_key = f"field_{str(key)}"
        else:
            safe_key = key

        if safe_key in _RESERVED_EXTRA_KEYS:
            new_key = f"field_{safe_key}"
            while new_key in out or new_key in _RESERVED_EXTRA_KEYS:
                new_key = f"field_{new_key}"
            key_map[safe_key] = new_key
            safe_key = new_key

        out[safe_key] = value

    if key_map:
        out["extra_key_map"] = key_map

    return out


@dataclass(frozen=True)
class InteractionLogger:
    session_id: str
    log_dir: Path
    text_log_path: Path
    jsonl_log_path: Path
    logger: logging.Logger

    def event(self, event: str, **fields: Any) -> None:
        extra = _sanitize_extra_fields({"session_id": self.session_id, "event": event, **fields})
        self.logger.info(event, extra=extra)

    def exception(self, event: str, **fields: Any) -> None:
        extra = _sanitize_extra_fields({"session_id": self.session_id, "event": event, **fields})
        self.logger.exception(event, extra=extra)


def setup_interaction_logging(
    *,
    log_dir: Optional[Path] = None,
    app_name: str = "memento-s",
    level: int = logging.INFO,
) -> InteractionLogger:
    global _GLOBAL
    if _GLOBAL is not None:
        return _GLOBAL

    resolved_log_dir = Path(log_dir) if log_dir else (_repo_root() / "log")
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    session_id = f"{app_name}_{ts}_{pid}"

    text_log_path = resolved_log_dir / f"{session_id}.log"
    jsonl_log_path = resolved_log_dir / f"{session_id}.jsonl"

    logger = logging.getLogger("memento.interaction")
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(text_log_path) for h in logger.handlers):
        text_handler = logging.FileHandler(text_log_path, encoding="utf-8")
        text_handler.setLevel(level)
        text_handler.addFilter(_EnsureExtrasFilter())
        text_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s %(message)s %(event)s %(session_id)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(text_handler)

        jsonl_handler = logging.FileHandler(jsonl_log_path, encoding="utf-8")
        jsonl_handler.setLevel(level)
        jsonl_handler.addFilter(_EnsureExtrasFilter())
        jsonl_handler.setFormatter(_JsonlFormatter())
        logger.addHandler(jsonl_handler)

    interaction = InteractionLogger(
        session_id=session_id,
        log_dir=resolved_log_dir,
        text_log_path=text_log_path,
        jsonl_log_path=jsonl_log_path,
        logger=logger,
    )

    def _excepthook(exc_type, exc, tb) -> None:
        try:
            logger.error(
                "uncaught_exception",
                exc_info=(exc_type, exc, tb),
                extra={"event": "uncaught_exception", "session_id": session_id},
            )
        finally:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook

    interaction.event(
        "session_start",
        argv=list(sys.argv),
        python=sys.version.split()[0],
        platform=platform.platform(),
        cwd=str(Path.cwd()),
        time=time.time(),
        thread_count=threading.active_count(),
    )

    _GLOBAL = interaction
    return interaction


def get_interaction_logger() -> InteractionLogger:
    return setup_interaction_logging()


def try_get_interaction_logger() -> Optional[InteractionLogger]:
    return _GLOBAL
