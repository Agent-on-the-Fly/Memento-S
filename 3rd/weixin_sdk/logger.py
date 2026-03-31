"""
Structured logging module for weixin_sdk.

Provides structured JSON logging with context support, correlation IDs,
and sensitive field redaction. Inspired by the TypeScript logger implementation.
"""

import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, Optional
from logging.handlers import RotatingFileHandler


# Log level mapping matching TypeScript logger
LEVEL_IDS: Dict[str, int] = {
    "TRACE": 1,
    "DEBUG": 2,
    "INFO": 3,
    "WARN": 4,
    "ERROR": 5,
    "FATAL": 6,
}

DEFAULT_LOG_LEVEL = "INFO"

# Sensitive field names that should be redacted
SENSITIVE_FIELDS = {
    "token",
    "access_token",
    "refresh_token",
    "context_token",
    "bot_token",
    "authorization",
    "Authorization",
    "password",
    "secret",
    "api_key",
    "apikey",
    "private_key",
    "session_id",
}

# Default field name patterns for redaction (word boundary matching)
SENSITIVE_FIELD_PATTERNS = r"\b(token|access_token|refresh_token|context_token|bot_token|authorization|Authorization|password|secret|api_key|apikey|private_key|session_id)\b"


class SensitiveFieldRedactor:
    """Handles redaction of sensitive fields from log data."""

    def __init__(self, sensitive_fields: Optional[set] = None):
        self.sensitive_fields = sensitive_fields or SENSITIVE_FIELDS

    def redact_value(self, value: Any) -> Any:
        """Redact a single value."""
        if value is None:
            return None
        if isinstance(value, str):
            if len(value) <= 4:
                return "****"
            return f"{value[:4]}…(len={len(value)})"
        if isinstance(value, dict):
            return self.redact_dict(value)
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        return value

    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from a dictionary."""
        if not isinstance(data, dict):
            return data

        result = {}
        for key, value in data.items():
            if key.lower() in self.sensitive_fields or any(
                pattern in key.lower()
                for pattern in ["token", "password", "secret", "key"]
            ):
                result[key] = "<redacted>"
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.redact_value(item) if isinstance(item, (dict, str)) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def redact_string(self, text: str) -> str:
        """Redact sensitive patterns from a JSON string."""
        import re

        # Redact "key":"value" patterns for sensitive fields
        pattern = rf'("(?:{"|".join(self.sensitive_fields)})")\s*:\s*"[^"]*"'
        redacted = re.sub(pattern, r'\1:"<redacted>"', text, flags=re.IGNORECASE)
        return redacted


class LogContext:
    """Thread-local storage for log context and correlation IDs."""

    def __init__(self):
        self._storage = threading.local()

    @property
    def context(self) -> Dict[str, Any]:
        """Get the current context dictionary."""
        if not hasattr(self._storage, "context"):
            self._storage.context = {}
        return self._storage.context

    @context.setter
    def context(self, value: Dict[str, Any]):
        self._storage.context = value

    @property
    def correlation_id(self) -> Optional[str]:
        """Get the current correlation ID."""
        return getattr(self._storage, "correlation_id", None)

    @correlation_id.setter
    def correlation_id(self, value: Optional[str]):
        self._storage.correlation_id = value

    def clear(self):
        """Clear all context."""
        self._storage.context = {}
        self._storage.correlation_id = None


# Global context storage
_log_context = LogContext()


class StructuredLogRecord(logging.LogRecord):
    """Custom log record with structured data support."""

    def __init__(self, *args, extra: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra = extra or {}


class StructuredLogger(logging.Logger):
    """Structured logger with context support and JSON output."""

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        json_format: bool = True,
        output_file: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        redact_sensitive: bool = True,
    ):
        super().__init__(name, level)
        self.json_format = json_format
        self.redact_sensitive = redact_sensitive
        self.redactor = SensitiveFieldRedactor() if redact_sensitive else None

        # Set up handlers
        self.handlers = []
        self._setup_handlers(output_file, max_bytes, backup_count)

        # Ensure we don't propagate to root logger
        self.propagate = False

    def _setup_handlers(
        self, output_file: Optional[str], max_bytes: int, backup_count: int
    ):
        """Set up console and file handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_formatter())
        self.addHandler(console_handler)

        # File handler (optional)
        if output_file:
            # Ensure directory exists
            log_dir = os.path.dirname(output_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = RotatingFileHandler(
                output_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(self._get_formatter())
            self.addHandler(file_handler)

    def _get_formatter(self) -> logging.Formatter:
        """Get the appropriate formatter based on settings."""
        if self.json_format:
            # For JSON format, we handle formatting in _format_log_entry
            return logging.Formatter("%(message)s")
        return logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

    def _get_timestamp(self) -> str:
        """Get ISO 8601 formatted timestamp."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _format_log_entry(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, object]:
        """Format a structured log entry."""
        # Build base entry
        entry: Dict[str, object] = {
            "level": level,
            "message": message,
            "timestamp": self._get_timestamp(),
            "logger": self.name,
        }

        # Add correlation ID if present
        correlation_id = _log_context.correlation_id
        if correlation_id:
            entry["correlation_id"] = correlation_id

        # Add context fields
        context = _log_context.context
        if context:
            # Redact sensitive fields in context
            if self.redact_sensitive and self.redactor:
                context = self.redactor.redact_dict(context)
            entry.update(context)

        # Add extra fields
        if extra:
            # Redact sensitive fields in extra
            if self.redact_sensitive and self.redactor:
                extra = self.redactor.redact_dict(extra)
            entry.update(extra)

        # Add level ID for filtering (level is already a string here)
        entry["level_id"] = LEVEL_IDS.get(level, LEVEL_IDS["INFO"])

        return entry

    def _format_and_log(
        self,
        level: int,
        msg: str,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Format and log a message with structured data."""
        if extra is None:
            extra = {}

        # Get level name
        level_name = logging.getLevelName(level)
        if not level_name or level_name == "Level %s":
            level_name = "INFO"

        # Skip if below minimum level
        level_id = LEVEL_IDS.get(level_name, LEVEL_IDS["INFO"])
        min_level_id = LEVEL_IDS.get(
            logging.getLevelName(self.getEffectiveLevel()), LEVEL_IDS[DEFAULT_LOG_LEVEL]
        )

        if level_id < min_level_id:
            return

        # Format entry
        if self.json_format:
            entry = self._format_log_entry(level_name, msg, extra)
            log_message = json.dumps(entry, ensure_ascii=False)
        else:
            log_message = msg
            if extra:
                extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
                log_message = f"{msg} | {extra_str}"

        # Call parent _log
        super()._log(level, log_message, (), None, None, False, 1)

    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None, *args, **kwargs):
        """Log a debug message with optional extra fields."""
        self._format_and_log(logging.DEBUG, msg, extra)

    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None, *args, **kwargs):
        """Log an info message with optional extra fields."""
        self._format_and_log(logging.INFO, msg, extra)

    def warn(self, msg: str, extra: Optional[Dict[str, Any]] = None, *args, **kwargs):
        """Log a warning message with optional extra fields."""
        self._format_and_log(logging.WARN, msg, extra)

    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None, *args, **kwargs):
        """Log an error message with optional extra fields."""
        self._format_and_log(logging.ERROR, msg, extra)

    def fatal(self, msg: str, extra: Optional[Dict[str, Any]] = None, *args, **kwargs):
        """Log a fatal message with optional extra fields."""
        self._format_and_log(logging.CRITICAL, msg, extra)

    @contextmanager
    def context(self, **kwargs) -> Generator[None, None, None]:
        """Context manager to add fields to all log entries within the block."""
        old_context = _log_context.context.copy() if _log_context.context else {}
        _log_context.context.update(kwargs)
        try:
            yield
        finally:
            _log_context.context = old_context

    def with_context(self, **kwargs) -> "StructuredLogger":
        """Return a logger bound with additional context."""
        # Create a child logger with merged context
        child = StructuredLogger(
            self.name,
            level=self.level,
            json_format=self.json_format,
            redact_sensitive=self.redact_sensitive,
        )
        # Merge current context into child (but won't persist for subsequent calls)
        # This is a simple implementation - for full context binding, use context() manager
        return child


# Module-level configuration
_config: Dict[str, Any] = {
    "json_format": True,
    "output_file": None,
    "log_level": DEFAULT_LOG_LEVEL,
    "max_bytes": 10 * 1024 * 1024,
    "backup_count": 5,
    "redact_sensitive": True,
}

_loggers: Dict[str, StructuredLogger] = {}
_loggers_lock = threading.Lock()


def configure(
    json_format: bool = True,
    output_file: Optional[str] = None,
    log_level: str = DEFAULT_LOG_LEVEL,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    redact_sensitive: bool = True,
):
    """Configure the logging module globally."""
    _config.update(
        {
            "json_format": json_format,
            "output_file": output_file,
            "log_level": log_level.upper(),
            "max_bytes": max_bytes,
            "backup_count": backup_count,
            "redact_sensitive": redact_sensitive,
        }
    )

    # Clear cached loggers to apply new config
    with _loggers_lock:
        _loggers.clear()


def get_logger(
    name: str,
    json_format: Optional[bool] = None,
    output_file: Optional[str] = None,
    log_level: Optional[str] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
    redact_sensitive: Optional[bool] = None,
) -> StructuredLogger:
    """
    Get or create a structured logger.

    Args:
        name: Logger name (typically __name__)
        json_format: Whether to use JSON format (default: from config)
        output_file: Path to log file (optional)
        log_level: Minimum log level (default: from config)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        redact_sensitive: Whether to redact sensitive fields

    Returns:
        StructuredLogger instance
    """
    # Use config defaults if not specified
    _json_format = json_format if json_format is not None else _config["json_format"]
    _log_level = log_level if log_level is not None else _config["log_level"]
    _max_bytes = max_bytes if max_bytes is not None else _config["max_bytes"]
    _backup_count = (
        backup_count if backup_count is not None else _config["backup_count"]
    )
    _redact_sensitive = (
        redact_sensitive
        if redact_sensitive is not None
        else _config["redact_sensitive"]
    )
    _output_file = output_file if output_file is not None else _config["output_file"]

    # Create unique key for logger configuration
    cache_key = f"{name}:{_json_format}:{_output_file}:{_log_level}:{_redact_sensitive}"

    with _loggers_lock:
        if cache_key not in _loggers:
            level = LEVEL_IDS.get(_log_level.upper(), LEVEL_IDS[DEFAULT_LOG_LEVEL])
            logger = StructuredLogger(
                name=name,
                level=level,
                json_format=_json_format,
                output_file=_output_file,
                max_bytes=_max_bytes,
                backup_count=_backup_count,
                redact_sensitive=_redact_sensitive,
            )
            _loggers[cache_key] = logger

        return _loggers[cache_key]


def set_correlation_id(correlation_id: Optional[str]) -> None:
    """Set the correlation ID for request tracing."""
    _log_context.correlation_id = correlation_id


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _log_context.correlation_id


@contextmanager
def correlation_context(correlation_id: str) -> Generator[None, None, None]:
    """Context manager to set correlation ID for a block."""
    old_id = _log_context.correlation_id
    _log_context.correlation_id = correlation_id
    try:
        yield
    finally:
        _log_context.correlation_id = old_id


def clear_context() -> None:
    """Clear all context and correlation ID."""
    _log_context.clear()


def set_log_level(level: str) -> None:
    """
    Dynamically change the minimum log level.

    Args:
        level: Log level name (TRACE, DEBUG, INFO, WARN, ERROR, FATAL)
    """
    level_upper = level.upper()
    if level_upper not in LEVEL_IDS:
        raise ValueError(
            f"Invalid log level: {level}. Valid levels: {list(LEVEL_IDS.keys())}"
        )

    # Update all cached loggers
    level_id = LEVEL_IDS[level_upper]
    with _loggers_lock:
        for logger in _loggers.values():
            logger.setLevel(level_id)


# Convenience function for getting root logger
def get_root_logger(
    json_format: Optional[bool] = None,
    output_file: Optional[str] = None,
) -> StructuredLogger:
    """Get the root structured logger."""
    return get_logger("weixin_sdk", json_format=json_format, output_file=output_file)


# Example usage and testing
if __name__ == "__main__":
    # Demo usage
    logger = get_logger("weixin_sdk.demo")

    print("=== Basic logging ===")
    logger.info("Simple info message")
    logger.debug("Debug message")
    logger.warn("Warning message")
    logger.error("Error message")
    logger.fatal("Fatal message")

    print("\n=== With extra fields ===")
    logger.info("User action", extra={"user_id": "user_123", "action": "login"})

    print("\n=== With correlation ID ===")
    with correlation_context("req-abc-123"):
        logger.info("Processing request", extra={"endpoint": "/api/users"})

    print("\n=== With context manager ===")
    with logger.context(request_id="req-xyz-789", user_id="user_456"):
        logger.info("User performing action")
        logger.info("Another log in same context")

    print("\n=== With sensitive data (redacted) ===")
    logger.info(
        "API call",
        extra={
            "url": "https://api.example.com",
            "token": "secret-token-12345",
            "password": "mysecretpassword",
        },
    )

    print("\n=== Text format ===")
    text_logger = get_logger("weixin_sdk.text", json_format=False)
    text_logger.info("Text format message", extra={"key": "value"})
