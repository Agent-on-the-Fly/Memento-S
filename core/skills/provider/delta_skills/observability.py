
from __future__ import annotations

from typing import Callable

from core.config.logging import get_logger

logger = get_logger(__name__)

_handlers: list[Callable[..., None]] = []


def register_handler(handler: Callable[..., None]) -> None:
    _handlers.append(handler)


def _notify(event: str, **kwargs) -> None:
    for h in _handlers:
        try:
            h(event, **kwargs)
        except Exception:
            pass


def track_resolve(source: str, success: bool, latency_ms: float) -> None:
    logger.info(
        "[metric] resolve: source=%s, success=%s, latency=%.1fms",
        source, success, latency_ms,
    )
    _notify("resolve", source=source, success=success, latency_ms=latency_ms)


def track_retrieval(method: str, result_count: int, latency_ms: float) -> None:
    logger.info(
        "[metric] retrieval: method=%s, results=%d, latency=%.1fms",
        method, result_count, latency_ms,
    )
    _notify("retrieval", method=method, result_count=result_count, latency_ms=latency_ms)


def track_reflection(
    skill_name: str, attempts: int, diagnosis: str, success: bool, latency_ms: float,
) -> None:
    logger.info(
        "[metric] reflection: skill=%s, attempts=%d, diagnosis=%s, success=%s, latency=%.1fms",
        skill_name, attempts, diagnosis, success, latency_ms,
    )
    _notify(
        "reflection",
        skill_name=skill_name, attempts=attempts,
        diagnosis=diagnosis, success=success, latency_ms=latency_ms,
    )
