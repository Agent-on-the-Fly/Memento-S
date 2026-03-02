"""Fetch URL content using crawl4ai with loop/download guardrails."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from crawl4ai import AsyncWebCrawler

_DOWNLOAD_LIKE_EXTENSIONS = {
    ".7z",
    ".bz2",
    ".csv",
    ".doc",
    ".docx",
    ".gz",
    ".jsonl",
    ".parquet",
    ".pdf",
    ".ppt",
    ".pptx",
    ".rar",
    ".tar",
    ".tgz",
    ".tsv",
    ".xls",
    ".xlsx",
    ".xml",
    ".xz",
    ".zip",
}
_DEFAULT_FETCH_BLOCK_SUBSTRINGS = "arxiv.org/src/,/download?,download=1,download=true"


def _int_env(name: str, default: int, *, minimum: int = 1, maximum: int = 10000) -> int:
    try:
        value = int(str(os.getenv(name, str(default))).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _guard_state_path() -> Path:
    raw = str(os.getenv("WEB_FETCH_GUARD_STATE_FILE") or "").strip()
    if raw:
        p = Path(raw).expanduser()
    else:
        p = Path(".agent") / "web_fetch_guard_state.json"
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_state(path: Path) -> dict:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {"url": {}, "host": {}}


def _save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _normalize_url(raw_url: str) -> tuple[str, str, str] | None:
    text = str(raw_url or "").strip()
    if not text:
        return None
    try:
        parsed = urlsplit(text)
    except Exception:
        return None
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        return None
    host = (parsed.netloc or "").strip().lower()
    if not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or "/"
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    normalized = urlunsplit((scheme, host, path, query, ""))
    return normalized, path.lower(), host


def _is_download_like(normalized_url: str, path_lower: str) -> tuple[bool, str]:
    blocks = tuple(
        token.strip().lower()
        for token in str(os.getenv("WEB_FETCH_BLOCK_SUBSTRINGS") or _DEFAULT_FETCH_BLOCK_SUBSTRINGS).split(",")
        if token.strip()
    )
    low = normalized_url.lower()
    for token in blocks:
        if token and token in low:
            return True, f"blocked_pattern:{token}"
    if "/src/" in path_lower:
        return True, "download_source_path"
    for ext in _DOWNLOAD_LIKE_EXTENSIONS:
        if path_lower.endswith(ext):
            return True, f"download_extension:{ext}"
    try:
        for key, value in parse_qsl(urlsplit(normalized_url).query, keep_blank_values=True):
            k = key.lower()
            v = value.lower()
            if k in {"download", "attachment", "filename", "file", "format"}:
                if k == "format" and v in {"html", "htm", "md", "txt"}:
                    continue
                return True, f"download_query:{k}"
    except Exception:
        pass
    return False, ""


def _enforce_fetch_guard(raw_url: str) -> tuple[bool, str]:
    normalized = _normalize_url(raw_url)
    if normalized is None:
        return False, "invalid_or_unsupported_url"
    url_key, path_lower, host_key = normalized

    blocked, reason = _is_download_like(url_key, path_lower)
    if blocked:
        return False, f"blocked download-like URL ({reason}); use skill `download`"

    url_limit = _int_env("WEB_FETCH_MAX_REPEAT_PER_URL", 4, minimum=1, maximum=50)
    host_limit = _int_env("WEB_FETCH_MAX_PER_HOST", 10, minimum=2, maximum=200)

    state_path = _guard_state_path()
    state = _load_state(state_path)
    url_counts = state.get("url") if isinstance(state.get("url"), dict) else {}
    host_counts = state.get("host") if isinstance(state.get("host"), dict) else {}

    if len(url_counts) > 2048:
        url_counts = {}
    if len(host_counts) > 1024:
        host_counts = {}

    url_seen = int(url_counts.get(url_key, 0)) + 1
    host_seen = int(host_counts.get(host_key, 0)) + 1
    url_counts[url_key] = url_seen
    host_counts[host_key] = host_seen
    state["url"] = url_counts
    state["host"] = host_counts
    _save_state(state_path, state)

    if url_seen > url_limit:
        return (
            False,
            f"duplicate_fetch_blocked for {url_key} (seen={url_seen}, limit={url_limit})",
        )
    if host_seen > host_limit:
        return (
            False,
            f"host_fetch_limit_blocked for {host_key} (seen={host_seen}, limit={host_limit})",
        )
    return True, ""


async def _fetch_async(url: str, max_length: int = 50000, raw: bool = False) -> str:
    """Async fetch implementation using crawl4ai."""
    async with AsyncWebCrawler() as crawler:
        timeout_sec = _int_env("WEB_FETCH_TIMEOUT_SEC", 60, minimum=5, maximum=300)
        result = await asyncio.wait_for(crawler.arun(url=url), timeout=timeout_sec)
        if raw:
            content = result.html or ""
        else:
            content = result.markdown or ""
        return str(content)[:max_length]


def fetch(url: str, max_length: int = 50000, raw: bool = False) -> str:
    """Fetch page content using crawl4ai.

    Args:
        url: URL to fetch
        max_length: Maximum content length (default 50000)
        raw: If True, return HTML; if False, return markdown (default False)

    Returns:
        Markdown or HTML content string
    """
    allowed, reason = _enforce_fetch_guard(url)
    if not allowed:
        return f"ERR: {reason}"

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            import concurrent.futures

            timeout_sec = _int_env("WEB_FETCH_TIMEOUT_SEC", 60, minimum=5, maximum=300)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _fetch_async(url, max_length, raw))
                return future.result(timeout=timeout_sec + 5)
        return asyncio.run(_fetch_async(url, max_length, raw))
    except Exception as e:
        message = str(e or "")
        if "Download is starting" in message:
            return (
                "ERR: fetch blocked because this URL triggers a direct download "
                "(Page.goto: Download is starting). Use skill `download` instead."
            )
        return f"Error fetching {url}: {e}"


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    content = fetch(url)
    print(content[:1000])
