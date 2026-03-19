"""
Web fetching and processing tool.
"""

from __future__ import annotations

import asyncio
import os
import re


async def fetch_webpage_tool(url: str) -> str:
    """
    Fetch a webpage and convert its main content to clean Markdown.
    Use this to read documentation, news, or API references.

    Args:
        url: The HTTP/HTTPS URL to fetch.
    """
    def _run() -> str:
        try:
            import httpx
            from bs4 import BeautifulSoup
            import markdownify

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            }

            response = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
                tag.decompose()

            main_content = soup.find("main") or soup.find("article") or soup.find("body") or soup
            md = markdownify.markdownify(str(main_content), heading_style="ATX")

            md = re.sub(r"\n\s*\n", "\n\n", md).strip()
            return f"--- Source: {url} ---\n{md[:15000]}"
        except ImportError:
            return "ERR: Missing dependencies. Please run: pip install httpx beautifulsoup4 markdownify"
        except Exception as e:
            return f"ERR: fetch_webpage failed: {e}"

    return await asyncio.to_thread(_run)


async def tavily_search_tool(
    query: str,
    *,
    search_depth: str = "basic",
    max_results: int = 3,
    include_raw_content: bool = False,
) -> str:
    """
    Search the web using Tavily API and return LLM-friendly markdown.

    Args:
        query: Search query (keep under 400 chars).
        search_depth: ultra-fast | fast | basic | advanced
        max_results: Maximum results (0-20).
        include_raw_content: Include full page content when available.
    """
    def _run() -> str:
        try:
            import httpx

            from middleware.config import g_config

            env = g_config.get_env()
            api_key = env.get("TAVILY_API_KEY")
            if not api_key:
                return "ERR: Missing TAVILY_API_KEY in config env."

            payload = {
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "include_raw_content": include_raw_content,
            }

            response = httpx.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=20.0,
            )
            response.raise_for_status()
            data = response.json()

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

    return await asyncio.to_thread(_run)
