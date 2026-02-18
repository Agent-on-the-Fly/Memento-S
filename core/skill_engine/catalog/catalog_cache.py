"""Shared cache/config utilities for split catalog modules."""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from hashlib import sha1
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from core.config import (
    PROJECT_ROOT,
    SEMANTIC_ROUTER_EMBED_BATCH_SIZE,
    SEMANTIC_ROUTER_EMBED_CACHE_DIR,
    SEMANTIC_ROUTER_EMBED_MAX_LENGTH,
    SEMANTIC_ROUTER_EMBED_PREWARM,
    SEMANTIC_ROUTER_EMBED_QUERY_INSTRUCTION,
    SEMANTIC_ROUTER_METHOD,
)

_ROUTER_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "this",
    "these",
    "those",
    "need",
    "needs",
    "using",
    "use",
    "help",
    "please",
}


def _tokenize_for_semantic(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [tok for tok in raw if tok and tok not in _ROUTER_STOPWORDS and len(tok) > 1]


def _catalog_signature(skills: list[dict]) -> str:
    digest = sha1()
    for s in skills:
        name = str(s.get("name") or "").strip()
        desc = str(s.get("description") or "").strip()
        digest.update(name.encode("utf-8", errors="ignore"))
        digest.update(b"|")
        digest.update(desc.encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()


_SEMANTIC_INDEX_SIG: str | None = None
_SEMANTIC_INDEX: dict[str, Any] | None = None
_SEMANTIC_INDEX_SOURCE_ID: int | None = None
_BM25_INDEX_SIG: str | None = None
_BM25_INDEX: dict[str, Any] | None = None
_BM25_INDEX_SOURCE_ID: int | None = None
_LAST_VISIBLE_AGENTS_SIG: str | None = None
_JSONL_CATALOG_CACHE_MAX_ITEMS = 8
_JSONL_CATALOG_CACHE_TTL_SEC = 900
_EMBEDDING_CACHE_MAX_ITEMS = 6
_JSONL_CATALOG_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_JSONL_CATALOG_LOCK = threading.Lock()
_EMBEDDING_DOC_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_EMBEDDING_RUNTIME_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_EMBEDDING_LOCK = threading.Lock()
_EMBEDDING_PREWARM_LOCK = threading.Lock()
_EMBEDDING_PREWARM_IN_PROGRESS: set[str] = set()
_EMBEDDING_PREWARM_COMPLETED: set[str] = set()
_DOTENV_STATE_LOCK = threading.Lock()
_DOTENV_LAST_MTIME_NS: int | None = None
_DOTENV_PATH = (PROJECT_ROOT / ".env").resolve()


def _lru_get(cache: OrderedDict[str, dict[str, Any]], key: str) -> dict[str, Any] | None:
    value = cache.get(key)
    if isinstance(value, dict):
        cache.move_to_end(key)
        return value
    return None


def _lru_put(
    cache: OrderedDict[str, dict[str, Any]],
    key: str,
    value: dict[str, Any],
    *,
    max_items: int,
) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


def _refresh_dotenv_if_changed() -> None:
    global _DOTENV_LAST_MTIME_NS
    try:
        st = _DOTENV_PATH.stat()
    except Exception:
        return

    with _DOTENV_STATE_LOCK:
        if _DOTENV_LAST_MTIME_NS == st.st_mtime_ns:
            return
        load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
        _DOTENV_LAST_MTIME_NS = st.st_mtime_ns


def _env_str(name: str, fallback: str = "") -> str:
    _refresh_dotenv_if_changed()
    val = os.getenv(name)
    if val is None:
        return str(fallback or "").strip()
    return str(val).strip()


def _env_int(name: str, fallback: int) -> int:
    raw = _env_str(name, str(fallback))
    try:
        return int(raw)
    except Exception:
        return int(fallback)


def _env_flag(name: str, fallback: bool) -> bool:
    raw = _env_str(name, "1" if fallback else "0").lower()
    return raw not in {"0", "false", "no", "off"}


def _router_method() -> str:
    return (_env_str("SEMANTIC_ROUTER_METHOD", SEMANTIC_ROUTER_METHOD) or "bm25").lower()


def _router_embed_max_length() -> int:
    return max(256, _env_int("SEMANTIC_ROUTER_EMBED_MAX_LENGTH", SEMANTIC_ROUTER_EMBED_MAX_LENGTH))


def _router_embed_batch_size() -> int:
    return max(1, _env_int("SEMANTIC_ROUTER_EMBED_BATCH_SIZE", SEMANTIC_ROUTER_EMBED_BATCH_SIZE))


def _router_embed_query_instruction() -> str:
    return _env_str(
        "SEMANTIC_ROUTER_EMBED_QUERY_INSTRUCTION",
        SEMANTIC_ROUTER_EMBED_QUERY_INSTRUCTION,
    ) or "Given a user query, retrieve relevant skill descriptions that match the query"


def _router_embed_prewarm_enabled() -> bool:
    return _env_flag("SEMANTIC_ROUTER_EMBED_PREWARM", SEMANTIC_ROUTER_EMBED_PREWARM)


def _router_embed_cache_dir() -> Path:
    raw = _env_str("SEMANTIC_ROUTER_EMBED_CACHE_DIR", str(SEMANTIC_ROUTER_EMBED_CACHE_DIR))
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def get_last_visible_agents_sig() -> str | None:
    return _LAST_VISIBLE_AGENTS_SIG


def set_last_visible_agents_sig(signature: str | None) -> None:
    global _LAST_VISIBLE_AGENTS_SIG
    _LAST_VISIBLE_AGENTS_SIG = signature


def get_or_build_semantic_index(
    skills: list[dict[str, Any]],
    builder: Any,
) -> dict[str, Any]:
    global _SEMANTIC_INDEX_SIG, _SEMANTIC_INDEX, _SEMANTIC_INDEX_SOURCE_ID
    if _SEMANTIC_INDEX is not None and _SEMANTIC_INDEX_SOURCE_ID == id(skills):
        return _SEMANTIC_INDEX
    sig = _catalog_signature(skills)
    if _SEMANTIC_INDEX is not None and _SEMANTIC_INDEX_SIG == sig:
        _SEMANTIC_INDEX_SOURCE_ID = id(skills)
        return _SEMANTIC_INDEX
    _SEMANTIC_INDEX = builder(skills)
    _SEMANTIC_INDEX_SIG = sig
    _SEMANTIC_INDEX_SOURCE_ID = id(skills)
    return _SEMANTIC_INDEX


def get_or_build_bm25_index(
    skills: list[dict[str, Any]],
    builder: Any,
) -> dict[str, Any] | None:
    global _BM25_INDEX_SIG, _BM25_INDEX, _BM25_INDEX_SOURCE_ID
    if _BM25_INDEX is not None and _BM25_INDEX_SOURCE_ID == id(skills):
        return _BM25_INDEX
    sig = _catalog_signature(skills)
    if _BM25_INDEX is not None and _BM25_INDEX_SIG == sig:
        _BM25_INDEX_SOURCE_ID = id(skills)
        return _BM25_INDEX
    _BM25_INDEX = builder(skills)
    _BM25_INDEX_SIG = sig
    _BM25_INDEX_SOURCE_ID = id(skills)
    return _BM25_INDEX


def get_jsonl_catalog_cache(
    cache_key: str,
    *,
    mtime_ns: int,
    size: int,
    now_ts: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]] | None:
    with _JSONL_CATALOG_LOCK:
        cached = _lru_get(_JSONL_CATALOG_CACHE, cache_key)
        if not cached:
            return None
        if cached.get("mtime_ns") != mtime_ns or cached.get("size") != size:
            return None
        cached_at = int(cached.get("cached_at") or 0)
        if now_ts - cached_at > _JSONL_CATALOG_CACHE_TTL_SEC:
            return None
        skills = cached.get("skills")
        by_name = cached.get("by_name")
        if isinstance(skills, list) and isinstance(by_name, dict):
            return skills, by_name
    return None


def put_jsonl_catalog_cache(
    cache_key: str,
    *,
    now_ts: int,
    mtime_ns: int,
    size: int,
    skills: list[dict[str, Any]],
    by_name: dict[str, list[dict[str, Any]]],
) -> None:
    with _JSONL_CATALOG_LOCK:
        _lru_put(
            _JSONL_CATALOG_CACHE,
            cache_key,
            {
                "cached_at": int(now_ts),
                "mtime_ns": mtime_ns,
                "size": size,
                "skills": skills,
                "by_name": by_name,
            },
            max_items=_JSONL_CATALOG_CACHE_MAX_ITEMS,
        )


def get_embedding_runtime_cache(cache_key: str) -> dict[str, Any] | None:
    with _EMBEDDING_LOCK:
        cached = _lru_get(_EMBEDDING_RUNTIME_CACHE, cache_key)
        return cached if isinstance(cached, dict) else None


def put_embedding_runtime_cache(cache_key: str, runtime: dict[str, Any]) -> None:
    with _EMBEDDING_LOCK:
        _lru_put(
            _EMBEDDING_RUNTIME_CACHE,
            cache_key,
            runtime,
            max_items=_EMBEDDING_CACHE_MAX_ITEMS,
        )


def get_embedding_doc_cache(cache_key: str) -> dict[str, Any] | None:
    with _EMBEDDING_LOCK:
        cached = _lru_get(_EMBEDDING_DOC_CACHE, cache_key)
        return cached if isinstance(cached, dict) else None


def put_embedding_doc_cache(cache_key: str, payload: dict[str, Any]) -> None:
    with _EMBEDDING_LOCK:
        _lru_put(
            _EMBEDDING_DOC_CACHE,
            cache_key,
            payload,
            max_items=_EMBEDDING_CACHE_MAX_ITEMS,
        )


def begin_embedding_prewarm(prewarm_key: str) -> bool:
    with _EMBEDDING_PREWARM_LOCK:
        if (
            prewarm_key in _EMBEDDING_PREWARM_COMPLETED
            or prewarm_key in _EMBEDDING_PREWARM_IN_PROGRESS
        ):
            return False
        _EMBEDDING_PREWARM_IN_PROGRESS.add(prewarm_key)
        return True


def finish_embedding_prewarm(prewarm_key: str) -> None:
    with _EMBEDDING_PREWARM_LOCK:
        _EMBEDDING_PREWARM_IN_PROGRESS.discard(prewarm_key)
        _EMBEDDING_PREWARM_COMPLETED.add(prewarm_key)


def router_method() -> str:
    return _router_method()


def router_embed_max_length() -> int:
    return _router_embed_max_length()


def router_embed_batch_size() -> int:
    return _router_embed_batch_size()


def router_embed_query_instruction() -> str:
    return _router_embed_query_instruction()


def router_embed_prewarm_enabled() -> bool:
    return _router_embed_prewarm_enabled()


def router_embed_cache_dir() -> Path:
    return _router_embed_cache_dir()


def env_str(name: str, fallback: str = "") -> str:
    return _env_str(name, fallback)


def tokenize_for_semantic(text: str) -> list[str]:
    return _tokenize_for_semantic(text)


def catalog_signature(skills: list[dict]) -> str:
    return _catalog_signature(skills)
