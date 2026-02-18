"""Web executor split from skill_executor.py."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import requests

from core.utils.path_utils import _truncate_text

from ..executor_utils import canonicalize_op_type, parse_bool, parse_int


def _normalize_organic_results(items: list[Any], *, limit: int) -> list[dict]:
    out: list[dict] = []
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


def web_google_search(query: str, num_results: int = 10) -> list[dict]:
    q = str(query or "").strip()
    if not q:
        return []
    n = parse_int(num_results, 10, minimum=1)

    serper_key = (os.getenv("SERPER_API_KEY") or os.getenv("SERPER_DEV_API_KEY") or "").strip()
    serpapi_key = (os.getenv("SERPAPI_API_KEY") or "").strip()
    if serper_key:
        try:
            endpoint = (os.getenv("SERPER_BASE_URL") or "https://google.serper.dev/search").strip()
            payload = {"q": q, "num": max(1, min(n, 20))}
            headers = {
                "X-API-KEY": serper_key,
                "Content-Type": "application/json",
            }
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
            body_preview = str(resp.text or "")[:1000]
            if resp.status_code >= 400:
                raise RuntimeError(f"Serper error {resp.status_code}: {body_preview}")

            try:
                data = resp.json()
            except Exception as exc:
                raise RuntimeError(f"Serper returned non-JSON response: {body_preview}") from exc

            if not isinstance(data, dict):
                raise RuntimeError(f"Serper returned non-dict response: {type(data).__name__}")
            if data.get("error"):
                raise RuntimeError(f"Serper error: {data.get('error')}")
            organic = data.get("organic")
            if organic is None:
                return []
            if not isinstance(organic, list):
                raise RuntimeError(f"Serper organic field is not a list: {type(organic).__name__}")
            return _normalize_organic_results(organic, limit=n)
        except requests.RequestException as exc:
            if not serpapi_key:
                raise RuntimeError(f"Serper request failed: {type(exc).__name__}: {exc}") from exc
        except Exception as serper_exc:
            if not serpapi_key:
                raise RuntimeError(
                    "Serper search failed and no SerpAPI fallback key is set: "
                    f"{type(serper_exc).__name__}: {serper_exc}"
                ) from serper_exc

    if not serpapi_key:
        raise RuntimeError("Missing search API key: set SERPER_API_KEY (or fallback SERPAPI_API_KEY)")

    from serpapi import GoogleSearch

    params = {
        "engine": "google",
        "q": q,
        "api_key": serpapi_key,
        "num": n,
    }
    search = GoogleSearch(params)
    results = search.get_dict() or {}
    if not isinstance(results, dict):
        raise RuntimeError(f"SerpAPI returned non-dict response: {type(results).__name__}")
    if results.get("error"):
        raise RuntimeError(f"SerpAPI error: {results.get('error')}")
    organic = results.get("organic_results")
    if organic is None:
        meta = results.get("search_metadata") if isinstance(results.get("search_metadata"), dict) else {}
        status = meta.get("status") or meta.get("api_status") or "unknown"
        raise RuntimeError(f"SerpAPI returned no organic_results (status={status})")
    if not isinstance(organic, list):
        raise RuntimeError(f"SerpAPI organic_results is not a list: {type(organic).__name__}")
    return _normalize_organic_results(organic, limit=n)


async def fetch_async(url: str, max_length: int = 50000, raw: bool = False) -> str:
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        content = result.html if raw else result.markdown
        return str(content or "")[:max_length]


def web_fetch(url: str, max_length: int = 50000, raw: bool = False) -> str:
    try:
        try:
            asyncio.get_running_loop()
            has_loop = True
        except RuntimeError:
            has_loop = False

        if has_loop:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, fetch_async(url, max_length, raw))
                return future.result(timeout=60)
        return asyncio.run(fetch_async(url, max_length, raw))
    except Exception as exc:
        return f"Error fetching {url}: {exc}"


def execute_web_ops(plan: dict[str, Any]) -> str:
    ops = plan.get("ops", [])
    if not isinstance(ops, list) or not ops:
        return "ERR: no tool_calls provided. Expected 'query' for search or 'url' for fetch."
    results: list[str] = []

    for op in ops:
        if not isinstance(op, dict):
            results.append("SKIP: op is not a dict")
            continue
        op_type = canonicalize_op_type(op.get("type"))

        if op_type == "web_search":
            query = str(op.get("query", ""))
            num_results = parse_int(op.get("num_results"), 10, minimum=1)
            try:
                search_results = web_google_search(query, num_results=num_results)
                output_parts = []
                for r in search_results:
                    output_parts.append(f"Title: {r.get('title', 'N/A')}")
                    output_parts.append(f"Link: {r.get('link', 'N/A')}")
                    output_parts.append(f"Snippet: {r.get('snippet', 'N/A')}")
                    output_parts.append("---")
                text = "\n".join(output_parts)
                results.append(f"[web_search]\n{text or 'No results found'}")
            except Exception as exc:
                results.append(f"[web_search]\nERR: {exc}")
            continue

        if op_type == "fetch":
            url = str(op.get("url", ""))
            max_length = parse_int(op.get("max_length"), 50000, minimum=1)
            raw_flag = parse_bool(op.get("raw"), False)
            content = web_fetch(url, max_length=max_length, raw=raw_flag)
            results.append(f"[fetch]\n{_truncate_text(content, 50000) or 'No content fetched'}")
            continue

        results.append(f"unknown op_type: {op_type}")

    return "\n\n".join(results) if results else "OK"
