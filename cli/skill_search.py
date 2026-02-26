"""Cloud skill catalog loading and query ranking for CLI `/skills` command."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from core.config import PROJECT_ROOT, SKILL_DYNAMIC_FETCH_CATALOG_JSONL
from core.skill_engine.catalog_jsonl import parse_catalog_jsonl_text

_DEFAULT_TIMEOUT_SEC = 12
_DEFAULT_CACHE_TTL_SEC = 600


def _build_catalog_meta(
    *,
    ok: bool,
    source: str,
    error: str,
    catalog_ref: str,
    cached: bool = False,
    stale: bool = False,
) -> dict[str, Any]:
    return {
        "ok": bool(ok),
        "source": str(source or "").strip() or "unknown",
        "error": str(error or "").strip(),
        "catalog_ref": str(catalog_ref or "").strip(),
        "cached": bool(cached),
        "stale": bool(stale),
    }


def _parse_int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _normalize_entry(obj: Any, *, line_no: int = 0) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("name") or "").strip()
    if not name:
        return None
    return {
        "name": name,
        "description": str(obj.get("description") or "").strip(),
        "githubUrl": str(obj.get("githubUrl") or obj.get("github_url") or "").strip(),
        "skillUrl": str(obj.get("skillUrl") or "").strip(),
        "id": str(obj.get("id") or "").strip(),
        "author": str(obj.get("author") or "").strip(),
        "stars": _parse_int_or_zero(obj.get("stars")),
        "updatedAt": _parse_int_or_zero(obj.get("updatedAt")),
        "_line": _parse_int_or_zero(line_no),
    }


def _choose_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    return max(
        entries,
        key=lambda e: (
            _parse_int_or_zero(e.get("stars")),
            _parse_int_or_zero(e.get("updatedAt")),
            len(str(e.get("description") or "")),
            -_parse_int_or_zero(e.get("_line")),
        ),
    )


def _parse_jsonl_text(text: str) -> list[dict[str, Any]]:
    _skills, by_name = parse_catalog_jsonl_text(text)
    out: list[dict[str, Any]] = []
    for name in sorted(str(k) for k in by_name.keys()):
        preferred = _choose_entry(by_name.get(name) or [])
        if preferred is not None:
            out.append(preferred)
    return out


def _is_url(ref: str) -> bool:
    value = str(ref or "").strip().lower()
    return value.startswith("http://") or value.startswith("https://")


def _resolve_local_path(ref: str) -> Path:
    p = Path(str(ref or "").strip()).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def _cache_path_for_ref(ref: str) -> Path:
    key = hashlib.sha1(str(ref or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
    return (PROJECT_ROOT / ".agent" / "cache" / f"skills-cloud-{key}.json").resolve()


def _load_cache(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _save_cache(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def load_cloud_skill_catalog(
    catalog_ref: str | None = None,
    *,
    timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    cache_ttl_sec: int = _DEFAULT_CACHE_TTL_SEC,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load skill catalog from URL or local JSONL file with cache fallback."""
    ref = str(catalog_ref or SKILL_DYNAMIC_FETCH_CATALOG_JSONL or "").strip()
    if not ref:
        return [], _build_catalog_meta(
            ok=False,
            source="none",
            error="empty catalog reference",
            catalog_ref=ref,
        )

    if not _is_url(ref):
        path = _resolve_local_path(ref)
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            return [], _build_catalog_meta(
                ok=False,
                source="local",
                error=f"failed to read catalog: {exc}",
                catalog_ref=str(path),
            )
        entries = _parse_jsonl_text(text)
        return entries, _build_catalog_meta(
            ok=bool(entries),
            source="local",
            error="" if entries else "catalog has no valid entries",
            catalog_ref=str(path),
        )

    cache_path = _cache_path_for_ref(ref)
    cached_payload = _load_cache(cache_path)

    try:
        req = urllib.request.Request(ref, headers={"User-Agent": "memento-s-cli/0.1"})
        with urllib.request.urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
            raw = resp.read()
        text = raw.decode("utf-8", errors="replace")
        entries = _parse_jsonl_text(text)
        now = int(time.time())
        _save_cache(
            cache_path,
            {
                "fetched_at": now,
                "catalog_ref": ref,
                "entries": entries,
            },
        )
        return entries, _build_catalog_meta(
            ok=bool(entries),
            source="remote",
            error="" if entries else "catalog has no valid entries",
            catalog_ref=ref,
        )
    except Exception as exc:
        if isinstance(cached_payload, dict):
            entries = cached_payload.get("entries")
            fetched_at = _parse_int_or_zero(cached_payload.get("fetched_at"))
            age_sec = max(0, int(time.time()) - fetched_at) if fetched_at else 10**9
            stale = age_sec > max(1, int(cache_ttl_sec))
            if isinstance(entries, list) and entries:
                return entries, _build_catalog_meta(
                    ok=True,
                    source="cache",
                    error=f"remote fetch failed: {exc}",
                    catalog_ref=ref,
                    cached=True,
                    stale=stale,
                )
        return [], _build_catalog_meta(
            ok=False,
            source="remote",
            error=f"remote fetch failed: {exc}",
            catalog_ref=ref,
        )


def _score_skill_entry(query: str, tokens: list[str], entry: dict[str, Any]) -> float:
    name = str(entry.get("name") or "").strip()
    desc = str(entry.get("description") or "").strip()
    author = str(entry.get("author") or "").strip()
    if not name:
        return 0.0

    name_l = name.lower()
    desc_l = desc.lower()
    author_l = author.lower()

    score = 0.0
    if name_l == query:
        score += 500
    if name_l.startswith(query):
        score += 280
    if query in name_l:
        score += 180
    if query in desc_l:
        score += 90

    for tok in tokens:
        if tok in name_l:
            score += 60
        if tok in desc_l:
            score += 20
        if tok in author_l:
            score += 10

    stars = _parse_int_or_zero(entry.get("stars"))
    updated_at = _parse_int_or_zero(entry.get("updatedAt"))
    score += min(stars, 5000) / 250
    if updated_at > 0:
        score += 2
    return score


def search_cloud_skills(
    query: str,
    entries: list[dict[str, Any]],
    *,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Rank cloud skills for a query and return top matches."""
    if not isinstance(entries, list) or not entries:
        return []

    q = str(query or "").strip().lower()
    top_k = max(1, min(int(top_k), 50))

    if not q:
        ranked = sorted(
            entries,
            key=lambda x: (
                _parse_int_or_zero(x.get("stars")),
                _parse_int_or_zero(x.get("updatedAt")),
                str(x.get("name") or "").lower(),
            ),
            reverse=True,
        )
        return ranked[:top_k]

    tokens = [t for t in re.split(r"[^a-z0-9]+", q) if t]
    scored: list[dict[str, Any]] = []

    for e in entries:
        score = _score_skill_entry(q, tokens, e)
        if score <= 0:
            continue

        item = dict(e)
        item["_score"] = score
        scored.append(item)

    scored.sort(
        key=lambda x: (
            float(x.get("_score") or 0),
            _parse_int_or_zero(x.get("stars")),
            _parse_int_or_zero(x.get("updatedAt")),
            str(x.get("name") or "").lower(),
        ),
        reverse=True,
    )
    return scored[:top_k]
