"""Fetch URL content using crawl4ai."""

import asyncio
from crawl4ai import AsyncWebCrawler


async def _fetch_async(url, max_length=50000, raw=False):
    """Async fetch implementation using crawl4ai."""
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

        if raw:
            content = result.html or ""
        else:
            content = result.markdown or ""

        return content[:max_length]


def fetch(url, max_length=50000, raw=False):
    """Fetch page content using crawl4ai.

    Args:
        url: URL to fetch
        max_length: Maximum content length (default 50000)
        raw: If True, return HTML; if False, return markdown (default False)

    Returns:
        Markdown or HTML content string
    """
    try:
        # Handle running in existing event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in an async context, create new loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, _fetch_async(url, max_length, raw)
                )
                return future.result(timeout=60)
        else:
            return asyncio.run(_fetch_async(url, max_length, raw))

    except Exception as e:
        return f"Error fetching {url}: {e}"


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    content = fetch(url)
    print(content[:1000])
