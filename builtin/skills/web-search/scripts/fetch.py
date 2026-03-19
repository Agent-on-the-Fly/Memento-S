"""Fetch URL content using Tavily-compatible parser."""

from __future__ import annotations

import re


def fetch(url: str) -> str:
    """Fetch a webpage and convert its main content to clean Markdown.

    Args:
        url: The HTTP/HTTPS URL to fetch.
    """
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
    except httpx.TimeoutException:
        return f"ERR: Request to {url} timed out. Please try again."
    except Exception as e:
        return f"ERR: fetch_webpage failed: {e}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: fetch.py <url>", file=sys.stderr)
        sys.exit(1)
    url = sys.argv[1]
    content = fetch(url)
    print(content[:1000])
