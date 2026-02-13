#!/usr/bin/env python3
"""Web search using SerpAPI - returns organic search results."""

import os
from serpapi import GoogleSearch


def google_search(query: str, num_results: int = 10) -> list[dict]:
    """
    Run a Google search via SerpAPI and return the organic results.

    Args:
        query: Search query string
        num_results: Number of results to return (default 10)

    Returns:
        List of organic search results, each containing:
        - title: Page title
        - link: URL
        - snippet: Brief description
        - position: Result ranking
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY", ""),
        "num": num_results,
    }
    if not params["api_key"]:
        raise RuntimeError("SERPAPI_API_KEY is not set")

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

    return organic


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
