---
name: web-search
description: Web search and content fetching using SerpAPI and crawl4ai. Use when the agent needs to search the web for information or fetch content from URLs.
---

# Web Search

Search the web and fetch content from URLs.

## Available Scripts

### `scripts/search.py` - Google Search

Search Google via SerpAPI and return organic results.

```python
from scripts.search import google_search

# Search for information
results = google_search("quantum computing", num_results=5)

# Each result contains: title, link, snippet, position
for r in results:
    print(r["title"], r["link"])
```

**Parameters:**
- `query` (str): Search query
- `num_results` (int): Number of results, default 10

**Returns:** List of dicts with `title`, `link`, `snippet`, `position`

### `scripts/fetch.py` - Fetch URL Content

Fetch and extract markdown content from a URL using crawl4ai.

```python
from scripts.fetch import fetch

# Fetch page content as markdown
content = fetch("https://example.com")

# Fetch with custom max length
content = fetch("https://example.com", max_length=100000)

# Fetch raw HTML
html = fetch("https://example.com", raw=True)
```

**Parameters:**
- `url` (str): URL to fetch
- `max_length` (int): Max content length, default 50000
- `raw` (bool): Return raw HTML instead of markdown, default False

**Returns:** Markdown (or HTML) content string

## Workflow

1. **Search**: Use `google_search()` to find relevant pages
2. **Fetch**: Use `fetch()` to get full content from specific URLs
3. **Extract**: Parse the content to find the information you need

## Example

```python
from scripts.search import google_search
from scripts.fetch import fetch

# Step 1: Search
results = google_search("Python asyncio tutorial")

# Step 2: Fetch top result
if results:
    url = results[0]["link"]
    content = fetch(url)
    print(content)
```

## Ops Format (Required)

Return JSON with `ops` array. **Do NOT use `call_skill_script` or `mcp_call` format.**

### web_search
Search Google for information:
```json
{"ops": [{"type": "web_search", "query": "search query", "num_results": 10}]}
```

### fetch
Fetch content from a URL:
```json
{"ops": [{"type": "fetch", "url": "https://example.com", "max_length": 50000, "raw": false}]}
```

### Combined workflow
Search then fetch top result:
```json
{
  "ops": [
    {"type": "web_search", "query": "Python asyncio tutorial", "num_results": 5}
  ]
}
```

After getting search results, fetch specific URLs:
```json
{
  "ops": [
    {"type": "fetch", "url": "https://docs.python.org/3/library/asyncio.html"}
  ]
}
```

### Final answer
When you have the answer, return:
```json
{"final": "Your answer here based on the search/fetch results"}
```

## Requirements

- `SERPAPI_API_KEY` environment variable must be set
- Dependencies: `serpapi`, `crawl4ai`
