# SPDX-License-Identifier: Apache-2.0
"""Prompts for memory consolidation — quick and deep modes."""

from __future__ import annotations

MAX_ENTRYPOINT_LINES = 200


QUICK_CONSOLIDATION_PROMPT = """\
You are a memory consolidation assistant. Extract and organize new knowledge.

## New Session Knowledge (_staging.md):
```
{staging_content}
```

## Memory Index:
```
{index_content}
```

## Task:
Review the new session knowledge and:
1. Update existing topic files if the knowledge extends or contradicts them
2. Create new topic files for distinct new subjects
3. Each topic: slug (kebab-case), title, content (markdown), hook (one-line description for index)

Return ONLY valid JSON:
{{
  "updated_topics": [{{"slug": str, "title": str, "content": str, "hook": str}}],
  "new_topics": [{{"slug": str, "title": str, "content": str, "hook": str}}],
  "deleted_topics": [],
  "index_content": str
}}

Keep output concise. Use max {max_lines} lines per topic content.\
"""


DEEP_CONSOLIDATION_PROMPT = """\
# Memory Consolidation

You are performing deep memory consolidation — synthesize recent session knowledge
into durable, well-organized memories.

## Current Memory Index:
```
{index}
```

## Existing Topics:
{topics}

## Staging Sessions (new knowledge):
```
{staging}
```

## Instructions:

### Phase 1 — Orient
- Understand the current knowledge structure from the index and existing topics

### Phase 2 — Gather
- Review pending session summaries for new knowledge
- Identify facts that contradict or update existing memories

### Phase 3 — Consolidate
- Update existing topic files with new information
- Create new topic files for distinct new subjects
- Merge related entries, remove contradictions
- Convert relative dates to absolute

### Phase 4 — Prune and Index
- Update MEMORY.md index (keep under {max_lines} lines / ~25KB)
- Each entry: `- [Title](file.md) — one-line hook`
- Remove stale/superseded entries

Return ONLY valid JSON:
{{
  "updated_topics": [{{"slug": str, "title": str, "content": str, "hook": str}}],
  "new_topics": [{{"slug": str, "title": str, "content": str, "hook": str}}],
  "deleted_topics": [slug: str],
  "index_content": str
}}"""
