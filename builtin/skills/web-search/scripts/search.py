#!/usr/bin/env python3
"""Web search using Tavily API - returns LLM-friendly markdown."""

from __future__ import annotations

import json


def tavily_search(
    query: str,
    *,
    search_depth: str = "basic",
    max_results: int = 3,
    include_raw_content: bool = False,
) -> str:
    """Search the web using Tavily API and return LLM-friendly markdown.

    Args:
        query: Search query (keep under 400 chars).
        search_depth: ultra-fast | fast | basic | advanced
        max_results: Maximum results (0-20).
        include_raw_content: Include full page content when available.
    """
    try:
        import httpx

        from middleware.config import g_config

        env = g_config.get_env()
        api_key = env.get("TAVILY_API_KEY")
        if not api_key:
            return (
                "ERR: Missing TAVILY_API_KEY in config env.\n"
                "To configure:\n"
                "  1. Get your API key at https://tavily.com\n"
                "  2. Add TAVILY_API_KEY to your config env settings"
            )

        payload = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
        }

        try:
            response = httpx.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=20.0,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            return "ERR: Tavily API request timed out. Please try again."
        except httpx.ConnectError as e:
            return f"ERR: Could not connect to Tavily API: {e}"
        except httpx.HTTPStatusError as e:
            return f"ERR: Tavily API returned HTTP {e.response.status_code}: {e}"

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            return f"ERR: Failed to parse Tavily API response as JSON: {e}"

        results = data.get("results", [])
        lines = [f"### Search Results for: {query}", ""]
        for idx, res in enumerate(results, start=1):
            title = res.get("title") or "(no title)"
            url = res.get("url") or ""
            content = res.get("raw_content") if include_raw_content else res.get("content")
            snippet = (content or "").strip()
            if len(snippet) > 3000:
                snippet = snippet[:3000] + "..."
            lines.append(f"**[{idx}] {title}**")
            if url:
                lines.append(f"URL: {url}")
            if snippet:
                lines.append("Content:" if include_raw_content else "Snippet:")
                lines.append(snippet)
            lines.append("")

        return "\n".join(lines).strip()
    except ImportError:
        return "ERR: Missing dependencies. Please run: pip install httpx"
    except Exception as e:
        return f"ERR: tavily_search failed: {e}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: search.py <query> [max_results]")
        sys.exit(1)

    query = sys.argv[1]
    try:
        max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    except ValueError:
        print(f"ERR: max_results must be an integer, got: {sys.argv[2]}", file=sys.stderr)
        sys.exit(1)

    output = tavily_search(query, max_results=max_results)
    print(output)
