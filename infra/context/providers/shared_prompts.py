# SPDX-License-Identifier: Apache-2.0
"""Shared prompt strings — 无 core/ 依赖的纯数据常量。

迁移自 core/context/prompts.py。
"""

from __future__ import annotations

import re

from utils.token_utils import estimate_tokens_fast

# ---------------------------------------------------------------------------
# 工具调用防护
# ---------------------------------------------------------------------------

NO_TOOLS_PREAMBLE: str = """\
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use any tool calls. You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

"""

NO_TOOLS_TRAILER: str = (
    "\n\nREMINDER: Do NOT call any tools. Respond with plain text only — "
    "an <analysis> block followed by a <summary> block. "
    "Tool calls will be rejected and you will fail the task."
)

# ---------------------------------------------------------------------------
# 分析指令
# ---------------------------------------------------------------------------

ANALYSIS_INSTRUCTION_FULL: str = """\
Before providing your final summary, wrap your analysis in <analysis> tags to organize \
your thoughts and ensure you've covered all necessary points. In your analysis:

1. Chronologically analyze each message and section of the conversation. For each section identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details: file names, full code snippets, function signatures, file edits
   - Errors encountered and how they were fixed
   - Specific user feedback, especially corrections or requests to do something differently
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly."""

ANALYSIS_INSTRUCTION_PARTIAL: str = """\
Before providing your final summary, wrap your analysis in <analysis> tags to organize \
your thoughts and ensure you've covered all necessary points. In your analysis:

1. Analyze the RECENT messages chronologically. For each section identify:
   - The user's explicit requests and intents
   - Your approach to addressing requests
   - Key decisions, technical concepts and code patterns
   - Specific details: file names, code snippets, function signatures
   - Errors and how they were fixed
   - User feedback, especially corrections
2. Double-check for technical accuracy and completeness."""

# ---------------------------------------------------------------------------
# 通用 system prompt
# ---------------------------------------------------------------------------

COMPACT_SYSTEM_PROMPT: str = (
    "You are a helpful AI assistant tasked with summarizing conversations."
)

COMPRESS_TOOL_RESULT_SYSTEM: str = (
    "You are a precise summarizer. Compress the following tool result "
    "while preserving all key facts, data paths, error messages, and actionable output. "
    "Return ONLY the compressed text."
)

# ---------------------------------------------------------------------------
# COMPRESS_EMERGENCY_PROMPT — 全量 9 段摘要
# ---------------------------------------------------------------------------

COMPRESS_EMERGENCY_PROMPT: str = f"""{NO_TOOLS_PREAMBLE}\
Your task is to create a detailed summary of the conversation so far, \
paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, \
and architectural decisions that would be essential for continuing development work \
without losing context.

{ANALYSIS_INSTRUCTION_FULL}

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. \
Pay special attention to the most recent messages and include full code snippets where applicable \
and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. \
Pay special attention to specific user feedback that you received, especially if the user told you \
to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. \
These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this \
summary request, paying special attention to the most recent messages from both user and assistant. \
Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent \
work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most \
recent explicit requests. If your last task was concluded, only list next steps if they are \
explicitly in line with the users request. \
Include direct quotes from the most recent conversation showing exactly what task you were \
working on and where you left off.

Please provide your summary based on the conversation so far, following this structure \
and ensuring precision and thoroughness in your response.{NO_TOOLS_TRAILER}"""

# ---------------------------------------------------------------------------
# COMPRESS_SM_COMPACT_PROMPT — SM compact 用
# ---------------------------------------------------------------------------

COMPRESS_SM_COMPACT_PROMPT: str = f"""{NO_TOOLS_PREAMBLE}\
Your task is to create a detailed summary of this conversation. \
This summary will be placed at the start of a continuing session; newer messages \
that build on this context will follow after your summary. \
Summarize thoroughly so that someone reading only your summary and then the newer \
messages can fully understand what happened and continue the work.

{ANALYSIS_INSTRUCTION_FULL}

Your summary should include the following sections:

1. Primary Request and Intent: Capture the user's explicit requests and intents in detail
2. Key Technical Concepts: List important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. \
Include full code snippets where applicable.
4. Errors and fixes: List errors encountered and how they were fixed.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results.
7. Pending Tasks: Outline any pending tasks.
8. Work Completed: Describe what was accomplished by the end of this portion.
9. Context for Continuing Work: Summarize any context, decisions, or state that would be needed \
to understand and continue the work in subsequent messages.

Please provide your summary following this structure.{NO_TOOLS_TRAILER}"""

# ---------------------------------------------------------------------------
# L1 Session Memory Template
# ---------------------------------------------------------------------------

DEFAULT_SESSION_MEMORY_TEMPLATE: str = """\
# Session Title
_A short 5-10 word descriptive title_

# Current State
_What is actively being worked on? Pending tasks? Immediate next steps?_

# Task Specification
_What did the user ask to build? Design decisions and context_

# Files and Functions
_Important files, what they contain, why they matter_

# Workflow
_Approaches tried, decisions made, commands run_

# Errors & Corrections
_Errors encountered, fixes applied, user corrections, approaches to avoid_

# Key Results
_Important outputs, findings, answers_

# Worklog
_Chronological log: timestamp, what was attempted/done_"""


SESSION_MEMORY_UPDATE_PROMPT: str = """\
Based on the recent conversation messages below, update the session notes file.

## Current Notes Content:
```
{current_notes}
```

## Recent Messages:
{messages_text}

## RULES:
- Output the COMPLETE updated notes file (all sections)
- Preserve ALL section headers (lines starting with "# ")
- Preserve italic descriptions under each header
- Only update content below italic descriptions
- Keep each section under ~2000 tokens
- ALWAYS update "# Current State" to reflect the most recent work
- Write DETAILED, INFO-DENSE content — avoid vague summaries
- For "# Files and Functions": include actual file paths and function names
- For "# Errors & Corrections": include exact error messages
- For "# Worklog": append new entries chronologically, do NOT remove old ones
- Convert relative time references to absolute dates where possible
{section_warnings}

Output ONLY the updated notes file content. No explanations or markdown fences."""

# ---------------------------------------------------------------------------
# Dream Consolidation Prompt
# ---------------------------------------------------------------------------

DREAM_CONSOLIDATION_PROMPT: str = """\
# Dream: Memory Consolidation

You are performing a dream — synthesize recent session knowledge into
durable, well-organized memories.

## Current Memory Index:
```
{index}
```

## Existing Topics:
{topics}

## Pending Sessions (new knowledge):
```
{pending}
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

Return a JSON object with:
- "updated_topics": [{{slug: str, title: str, content: str, hook: str}}]
- "new_topics": [{{slug: str, title: str, content: str, hook: str}}]
- "deleted_topics": [slug]
- "index_content": "updated MEMORY.md content"

Output ONLY valid JSON."""

# ---------------------------------------------------------------------------
# 格式化工具函数
# ---------------------------------------------------------------------------

_ANALYSIS_RE = re.compile(r"<analysis>[\s\S]*?</analysis>", re.DOTALL)
_SUMMARY_RE = re.compile(r"<summary>([\s\S]*?)</summary>", re.DOTALL)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def format_compact_summary(raw_summary: str) -> str:
    """剥离 <analysis> 草稿区，解包 <summary> 标签，规范化空行。"""
    result = _ANALYSIS_RE.sub("", raw_summary)

    match = _SUMMARY_RE.search(result)
    if match:
        content = match.group(1).strip()
        result = _SUMMARY_RE.sub(f"Summary:\n{content}", result)

    result = _MULTI_NEWLINE_RE.sub("\n\n", result)
    return result.strip()


def get_compact_user_summary_message(
    summary: str,
    *,
    suppress_follow_up: bool = True,
    transcript_path: str | None = None,
    recent_preserved: bool = False,
) -> str:
    """生成注入上下文的摘要消息。"""
    formatted = format_compact_summary(summary)

    parts: list[str] = [
        "This session is being continued from a previous conversation that ran out of context. "
        "The summary below covers the earlier portion of the conversation.",
        "",
        formatted,
    ]

    if transcript_path:
        parts.append(
            f"\nIf you need specific details from before compaction "
            f"(like exact code snippets, error messages, or content you generated), "
            f"read the full transcript at: {transcript_path}"
        )

    if recent_preserved:
        parts.append("\nRecent messages are preserved verbatim.")

    if suppress_follow_up:
        parts.append(
            "\nContinue the conversation from where it left off without asking "
            "the user any further questions. Resume directly — do not acknowledge "
            "the summary, do not recap what was happening, do not preface with "
            '"I\'ll continue" or similar. Pick up the last task as if the break '
            "never happened."
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# L1 helpers
# ---------------------------------------------------------------------------

def analyze_section_sizes(content: str) -> dict[str, int]:
    """按段统计 tokens。"""
    sections: dict[str, int] = {}
    current_header = ""
    current_text: list[str] = []

    for line in content.split("\n"):
        if line.startswith("# "):
            if current_header:
                sections[current_header] = estimate_tokens_fast("\n".join(current_text))
            current_header = line.strip()
            current_text = []
        else:
            current_text.append(line)

    if current_header:
        sections[current_header] = estimate_tokens_fast("\n".join(current_text))

    return sections


def generate_section_reminders(
    sizes: dict[str, int], max_per_section: int = 2000
) -> str:
    """生成超限 section 的警告文本。"""
    warnings: list[str] = []
    for header, tokens in sizes.items():
        if tokens > max_per_section:
            warnings.append(
                f"- WARNING: Section '{header}' has {tokens} tokens "
                f"(limit {max_per_section}). Condense it."
            )
    return "\n".join(warnings) if warnings else ""


def build_session_memory_update_prompt(
    current_notes: str, messages_text: str
) -> str:
    """构建 SM 更新 prompt，动态附加超限警告。"""
    sizes = analyze_section_sizes(current_notes)
    warnings = generate_section_reminders(sizes)
    warning_section = f"\n{warnings}" if warnings else ""

    return SESSION_MEMORY_UPDATE_PROMPT.format(
        current_notes=current_notes,
        messages_text=messages_text,
        section_warnings=warning_section,
    )
