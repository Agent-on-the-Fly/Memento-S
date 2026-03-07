---
name: web-search
description: Web search and content fetching using Serper and crawl4ai. Use when the agent needs to search the web for information or fetch content from URLs.
---

# Web Search

Search the web and fetch content from URLs.

## Available Scripts

### `scripts/search.py` - Google Search

Search Google via Serper and return organic results.

```bash
cd <skill_dir> && python3 scripts/search.py "quantum computing" 5
```

Output: JSON list of dicts with `title`, `link`, `snippet`, `position`.

**Arguments:**
- arg1 `query` (required): Search query
- arg2 `num_results` (optional): Number of results, default 10

### `scripts/fetch.py` - Fetch URL Content

Fetch and extract markdown content from a URL using crawl4ai.

```bash
cd <skill_dir> && python3 scripts/fetch.py "https://example.com"
```

Output: Markdown (or HTML) content string (truncated to first 1000 chars by default).

**Arguments:**
- arg1 `url` (required): URL to fetch

## Guardrails For `fetch`

- Use `fetch` for HTML/article pages.
- Do NOT use `fetch` for download-like URLs (for example: `arxiv.org/src/...`, `.pdf`, `.zip`, `.xlsx`, `.csv`, explicit `download=` query).
- If the URL is download-like, call skill `download` first, then continue analysis using local file tools/skills.
- If `fetch` returns `duplicate_fetch_blocked`, stop repeating the same URL and switch strategy.

## Workflow

1. **Search**: `python3 scripts/search.py "<query>" [num]` to find relevant pages
2. **Fetch**: `python3 scripts/fetch.py "<url>"` to get content from specific URLs
3. **Extract**: Parse the output to find the information you need

## Example

```bash
# Step 1: Search
cd <skill_dir> && python3 scripts/search.py "Python asyncio tutorial" 5

# Step 2: Fetch top result (use a URL from the search output)
cd <skill_dir> && python3 scripts/fetch.py "https://docs.python.org/3/library/asyncio.html"
```

## Requirements

- `SERPER_API_KEY` environment variable must be set
- Optional fallback: `SERPAPI_API_KEY`
- Optional override endpoint: `SERPER_BASE_URL` (default: `https://google.serper.dev/search`)
- Dependencies: `requests`, `crawl4ai` (optional fallback dependency: `serpapi`)
