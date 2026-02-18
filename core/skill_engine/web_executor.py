"""Web-skill executor extracted from skill_executor.

Keeps web integration concerns (Serper/SerpAPI/fetch) isolated from other bridge
execution domains and provides structured, code-based error handling.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import requests

from core.utils.path_utils import _truncate_text


@dataclass(frozen=True)
class WebExecutionError(Exception):
    """Structured web integration error with stable code/provider fields."""

    code: str
    message: str
    provider: str | None = None
    retryable: bool = False

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message

    def as_report(self) -> str:
        provider = f" provider={self.provider}" if self.provider else ""
        retryable = " retryable=true" if self.retryable else ""
        return f"ERR[{self.code}]{provider}{retryable}: {self.message}"


_WEB_OP_ALIASES: dict[str, str] = {
    "google_search": "web_search",
    "search": "web_search",
    "fetch_url": "fetch",
    "fetch_markdown": "fetch",
}


def _canonicalize_web_op_type(raw_type: Any) -> str:
    op_type = str(raw_type or "").strip().lower()
    if not op_type:
        return ""
    return _WEB_OP_ALIASES.get(op_type, op_type)


def _parse_int(value: Any, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    parsed = default
    try:
        if isinstance(value, bool):
            raise ValueError("bool is not accepted as int")
        parsed = int(value)
    except Exception:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered:
            return default
        if lowered in {"1", "true", "yes", "on", "y", "t"}:
            return True
        if lowered in {"0", "false", "no", "off", "n", "f"}:
            return False
        return default
    return bool(value)


def _normalize_organic_results(items: list[Any], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(items[:limit]):
        row = item if isinstance(item, dict) else {}
        position_raw = row.get("position")
        try:
            position = int(position_raw)
        except Exception:
            position = idx + 1
        out.append(
            {
                "title": str(row.get("title") or "N/A"),
                "link": str(row.get("link") or "N/A"),
                "snippet": str(row.get("snippet") or ""),
                "position": position,
            }
        )
    return out


def _web_google_search(query: str, num_results: int = 10) -> list[dict[str, Any]]:
    q = str(query or "").strip()
    if not q:
        return []
    n = _parse_int(num_results, 10, minimum=1)

    serper_key = (os.getenv("SERPER_API_KEY") or os.getenv("SERPER_DEV_API_KEY") or "").strip()
    serpapi_key = (os.getenv("SERPAPI_API_KEY") or "").strip()

    if serper_key:
        try:
            endpoint = (os.getenv("SERPER_BASE_URL") or "https://google.serper.dev/search").strip()
            payload = {"q": q, "num": max(1, min(n, 20))}
            headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
            body_preview = str(resp.text or "")[:1000]
            if resp.status_code >= 400:
                raise WebExecutionError(
                    code="serper_http_error",
                    provider="serper",
                    message=f"HTTP {resp.status_code}: {body_preview}",
                )

            try:
                data = resp.json()
            except Exception as exc:
                raise WebExecutionError(
                    code="serper_invalid_json",
                    provider="serper",
                    message=f"non-JSON response: {body_preview}",
                ) from exc

            if not isinstance(data, dict):
                raise WebExecutionError(
                    code="serper_invalid_payload",
                    provider="serper",
                    message=f"expected dict, got {type(data).__name__}",
                )
            if data.get("error"):
                raise WebExecutionError(
                    code="serper_api_error",
                    provider="serper",
                    message=str(data.get("error")),
                )
            organic = data.get("organic")
            if organic is None:
                return []
            if not isinstance(organic, list):
                raise WebExecutionError(
                    code="serper_invalid_organic",
                    provider="serper",
                    message=f"organic field must be list, got {type(organic).__name__}",
                )
            return _normalize_organic_results(organic, limit=n)
        except requests.RequestException as exc:
            if not serpapi_key:
                raise WebExecutionError(
                    code="serper_request_failed",
                    provider="serper",
                    message=f"{type(exc).__name__}: {exc}",
                    retryable=True,
                ) from exc
        except WebExecutionError as serper_exc:
            if not serpapi_key:
                raise serper_exc

    if not serpapi_key:
        raise WebExecutionError(
            code="search_api_key_missing",
            message="set SERPER_API_KEY or SERPAPI_API_KEY",
        )

    try:
        from serpapi import GoogleSearch
    except Exception as exc:
        raise WebExecutionError(
            code="serpapi_import_error",
            provider="serpapi",
            message=f"failed importing serpapi: {exc}",
        ) from exc

    params = {"engine": "google", "q": q, "api_key": serpapi_key, "num": n}
    results = GoogleSearch(params).get_dict() or {}
    if not isinstance(results, dict):
        raise WebExecutionError(
            code="serpapi_invalid_payload",
            provider="serpapi",
            message=f"expected dict, got {type(results).__name__}",
        )
    if results.get("error"):
        raise WebExecutionError(
            code="serpapi_api_error",
            provider="serpapi",
            message=str(results.get("error")),
        )
    organic = results.get("organic_results")
    if organic is None:
        meta = results.get("search_metadata") if isinstance(results.get("search_metadata"), dict) else {}
        status = meta.get("status") or meta.get("api_status") or "unknown"
        raise WebExecutionError(
            code="serpapi_missing_organic",
            provider="serpapi",
            message=f"no organic_results (status={status})",
        )
    if not isinstance(organic, list):
        raise WebExecutionError(
            code="serpapi_invalid_organic",
            provider="serpapi",
            message=f"organic_results must be list, got {type(organic).__name__}",
        )
    return _normalize_organic_results(organic, limit=n)


async def _fetch_async(url: str, max_length: int = 50000, raw: bool = False) -> str:
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        content = result.html if raw else result.markdown
        return str(content or "")[:max_length]


def _web_fetch(url: str, max_length: int = 50000, raw: bool = False) -> str:
    try:
        try:
            asyncio.get_running_loop()
            has_loop = True
        except RuntimeError:
            has_loop = False

        if has_loop:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _fetch_async(url, max_length, raw))
                return future.result(timeout=60)
        return asyncio.run(_fetch_async(url, max_length, raw))
    except Exception as exc:
        raise WebExecutionError(
            code="fetch_failed",
            provider="crawl4ai",
            message=f"Error fetching {url}: {exc}",
            retryable=True,
        ) from exc


def execute_web_ops(plan: dict[str, Any]) -> str:
    """Execute web operations (search/fetch) with structured error reporting."""
    ops = plan.get("ops", [])
    if not isinstance(ops, list) or not ops:
        return "ERR[invalid_plan]: no tool_calls provided. Expected 'query' for search or 'url' for fetch."

    results: list[str] = []
    for op in ops:
        if not isinstance(op, dict):
            results.append("SKIP: op is not a dict")
            continue

        op_type = _canonicalize_web_op_type(op.get("type"))
        if op_type == "web_search":
            query = str(op.get("query", ""))
            num_results = _parse_int(op.get("num_results"), 10, minimum=1)
            try:
                search_results = _web_google_search(query, num_results=num_results)
                output_parts: list[str] = []
                for row in search_results:
                    output_parts.append(f"Title: {row.get('title', 'N/A')}")
                    output_parts.append(f"Link: {row.get('link', 'N/A')}")
                    output_parts.append(f"Snippet: {row.get('snippet', 'N/A')}")
                    output_parts.append("---")
                text = "\n".join(output_parts)
                results.append(f"[web_search]\n{text or 'No results found'}")
            except WebExecutionError as exc:
                results.append(f"[web_search]\n{exc.as_report()}")
            continue

        if op_type == "fetch":
            url = str(op.get("url", ""))
            max_length = _parse_int(op.get("max_length"), 50000, minimum=1)
            raw_flag = _parse_bool(op.get("raw"), False)
            try:
                content = _web_fetch(url, max_length=max_length, raw=raw_flag)
                results.append(f"[fetch]\n{_truncate_text(content, 50000) or 'No content fetched'}")
            except WebExecutionError as exc:
                results.append(f"[fetch]\n{exc.as_report()}")
            continue

        results.append(f"unknown op_type: {op_type}")

    return "\n\n".join(results) if results else "OK"
