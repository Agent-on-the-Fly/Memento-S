---
name: web-search
description: Web search and content fetching. Use when the user needs to search the web for information or fetch content from URLs.
metadata:
  dependencies:
    - httpx
    - beautifulsoup4
    - markdownify
---

# Web Search

Search the web and fetch content from URLs.

## Setup

Set `TAVILY_API_KEY` in config env to enable web search. Get your API key at https://tavily.com.

## Available Scripts

### `scripts/search.py` - Web Search

Search the web via Tavily API and return LLM-friendly markdown.

```bash
# Search from command line
python scripts/search.py "quantum computing" 5

# From Python
from scripts.search import tavily_search
output = tavily_search("quantum computing", max_results=5)
print(output)
```

**Parameters:**
- `query` (str): Search query (keep under 400 chars)
- `search_depth` (str): ultra-fast | fast | basic | advanced (default `basic`)
- `max_results` (int): Maximum results, default 3
- `include_raw_content` (bool): Include full page content when available

**Returns:** Markdown string with formatted results

### `scripts/fetch.py` - Fetch URL Content

Fetch and extract markdown content from a URL. No API key required.

```bash
# Fetch from command line
python scripts/fetch.py "https://example.com"

# From Python
from scripts.fetch import fetch
content = fetch("https://example.com")
print(content)
```

**Parameters:**
- `url` (str): URL to fetch

**Returns:** Markdown content string

## Workflow

1. **Search**: Use `tavily_search()` to find relevant pages
2. **Fetch**: Use `fetch()` to get full content from specific URLs
3. **Extract**: Parse the content to find the information you need

## Requirements

- `TAVILY_API_KEY` must be set in config env for search functionality
- `fetch` works without any API key
- Dependencies: `httpx`, `beautifulsoup4`, `markdownify` (auto-installed)
