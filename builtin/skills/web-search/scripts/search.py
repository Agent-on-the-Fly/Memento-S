#!/usr/bin/env python3
"""Web search using Serper (with SerpAPI fallback)."""

from __future__ import annotations

import os
from typing import Any

import requests

_SERPER_SEARCH_URL = "https://google.serper.dev/search"
_SERPER_TIMEOUT_SECONDS = 20


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


def _search_with_serper(query: str, *, num_results: int) -> list[dict]:
    api_key = (os.getenv("SERPER_API_KEY") or os.getenv("SERPER_DEV_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not set")

    endpoint = (os.getenv("SERPER_BASE_URL") or _SERPER_SEARCH_URL).strip() or _SERPER_SEARCH_URL
    payload = {"q": query, "num": max(1, min(int(num_results), 20))}
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=_SERPER_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise RuntimeError(f"Serper request failed: {type(exc).__name__}: {exc}") from exc

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
    return _normalize_organic_results(organic, limit=num_results)


def _search_with_serpapi(query: str, *, num_results: int) -> list[dict]:
    from serpapi import GoogleSearch

    api_key = (os.getenv("SERPAPI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is not set")

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": num_results,
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
    return _normalize_organic_results(organic, limit=num_results)


def google_search(query: str, num_results: int = 10) -> list[dict]:
    """
    Run a Google search via Serper and return organic results.

    Falls back to SerpAPI when Serper is unavailable and SERPAPI_API_KEY is configured.
    """
    q = str(query or "").strip()
    if not q:
        return []
    try:
        n = max(1, int(num_results))
    except Exception:
        n = 10

    serper_key = (os.getenv("SERPER_API_KEY") or os.getenv("SERPER_DEV_API_KEY") or "").strip()
    serpapi_key = (os.getenv("SERPAPI_API_KEY") or "").strip()
    if serper_key:
        try:
            return _search_with_serper(q, num_results=n)
        except Exception as serper_exc:
            if serpapi_key:
                return _search_with_serpapi(q, num_results=n)
            raise RuntimeError(
                "Serper search failed and no SerpAPI fallback key is set: "
                f"{type(serper_exc).__name__}: {serper_exc}"
            ) from serper_exc
    if serpapi_key:
        return _search_with_serpapi(q, num_results=n)
    raise RuntimeError("Missing search API key: set SERPER_API_KEY (or fallback SERPAPI_API_KEY)")


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: search.py <query> [num_results]")
        sys.exit(1)

    query = sys.argv[1]
    num_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    results = google_search(query, num_results)
    print(json.dumps(results, indent=2))
