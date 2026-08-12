# SPDX-License-Identifier: Apache-2.0
"""压缩模块提示词 — 从 core/context/prompts.py 迁移。"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Summarizer preamble + summary prefix
# ---------------------------------------------------------------------------

SUMMARIZER_PREAMBLE: str = (
    "You are a summarization agent creating a context checkpoint. "
    "Your output will be injected as reference material for a DIFFERENT "
    "assistant that continues the conversation. "
    "Do NOT respond to any questions or requests in the conversation — "
    "only output the structured summary. "
    "Do NOT include any preamble, greeting, or prefix."
)

SUMMARY_PREFIX: str = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. Respond ONLY to the latest user message "
    "that appears AFTER this summary."
)

LEGACY_SUMMARY_PREFIX: str = "[CONTEXT SUMMARY]:"

# ---------------------------------------------------------------------------
# 13-field compression summary template
# ---------------------------------------------------------------------------

_COMPRESSION_TEMPLATE_SECTIONS: str = """## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Completed Actions
[Numbered list of concrete actions taken — include tool used, target, and outcome.
Format each as: N. ACTION target — outcome [tool: name]
Example:
1. READ config.py:45 — found `==` should be `!=` [tool: read_file]
2. PATCH config.py:45 — changed `==` to `!=` [tool: patch]
3. TEST `pytest tests/` — 3/50 failed: test_parse, test_validate, test_edge [tool: terminal]
Be specific with file paths, commands, line numbers, and results.]

## Active State
[Current working state — include:
- Working directory and branch (if applicable)
- Modified/created files with brief note on each
- Test status (X/Y passing)
- Any running processes or servers
- Environment details that matter]

## In Progress
[Work currently underway — what was being done when compaction fired]

## Blocked
[Any blockers, errors, or issues not yet resolved. Include exact error messages.]

## Key Decisions
[Important technical decisions and WHY they were made]

## Resolved Questions
[Questions the user asked that were ALREADY answered — include the answer so the next assistant does not re-answer them]

## Pending User Asks
[Questions or requests from the user that have NOT yet been answered or fulfilled. If none, write "None."]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Remaining Work
[What remains to be done — framed as context, not instructions]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation]

Target ~{summary_budget} tokens. Be CONCRETE — include file paths, command outputs, error messages, line numbers, and specific values. Avoid vague descriptions like "made some changes" — say exactly what changed.

Write only the summary body. Do not include any preamble or prefix."""


def build_compression_prompt(
    turns_to_summarize: str,
    summary_budget: int,
    previous_summary: str | None = None,
    focus_topic: str | None = None,
) -> str:
    """Build compression prompt with 13-field structured template.

    Supports both first-compaction and iterative-update modes.
    Optionally injects focus topic guidance for guided compression.
    """
    if previous_summary:
        # Iterative update mode
        prompt = f"""{SUMMARIZER_PREAMBLE}

You are updating a context compaction summary. A previous compaction produced the summary below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SUMMARY:
{previous_summary}

NEW TURNS TO INCORPORATE:
{turns_to_summarize}

Update the summary using this exact structure. PRESERVE all existing information that is still relevant. ADD new completed actions to the numbered list (continue numbering). Move items from "In Progress" to "Completed Actions" when done. Move answered questions to "Resolved Questions". Update "Active State" to reflect current state. Remove information only if it is clearly obsolete.

{_COMPRESSION_TEMPLATE_SECTIONS.format(summary_budget=summary_budget)}"""
    else:
        # First compaction mode
        prompt = f"""{SUMMARIZER_PREAMBLE}

Create a structured handoff summary for a different assistant that will continue this conversation after earlier turns are compacted. The next assistant should be able to understand what happened without re-reading the original turns.

TURNS TO SUMMARIZE:
{turns_to_summarize}

Use this exact structure:

{_COMPRESSION_TEMPLATE_SECTIONS.format(summary_budget=summary_budget)}"""

    if focus_topic:
        prompt += f"""

FOCUS TOPIC: "{focus_topic}"
The user has requested that this compaction PRIORITISE preserving all information related to the focus topic above. For content related to "{focus_topic}", include full detail — exact values, file paths, command outputs, error messages, and decisions. For content NOT related to the focus topic, summarise more aggressively (brief one-liners or omit if truly irrelevant). The focus topic sections should receive roughly 60-70% of the summary token budget."""

    return prompt


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
# COMPRESS_EMERGENCY_PROMPT — 9 段 (legacy, kept for compatibility)
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
work you were doing. IMPORTANT: ensure this step is DIRECTLY in line with the user's most \
recent explicit requests. If your last task was concluded, only list next steps if they are \
explicitly in line with the users request. \
Include direct quotes from the most recent conversation showing exactly what task you were \
working on and where you left off.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]
</summary>
</example>

Please provide your summary based on the conversation so far, following this structure \
and ensuring precision and thoroughness in your response.{NO_TOOLS_TRAILER}"""

# ---------------------------------------------------------------------------
# COMPRESS_SM_COMPACT_PROMPT — SM compact 用 (fallback)
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

<example>
<analysis>
[Your thought process, ensuring all points are covered]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Important Code Snippet]

4. Errors and fixes:
    - [Error description]:
      - [How you fixed it]

5. Problem Solving:
   [Description]

6. All user messages:
    - [Detailed non tool use user message]

7. Pending Tasks:
   - [Task 1]

8. Work Completed:
   [Description of what was accomplished]

9. Context for Continuing Work:
   [Key context, decisions, or state needed to continue]
</summary>
</example>

Please provide your summary following this structure.{NO_TOOLS_TRAILER}"""


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
# L1 Session Memory: Template + Update Prompt
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


def estimate_tokens_fast(text: str) -> int:
    """O(1) 粗略 token 估算。"""
    if not text:
        return 0
    return len(text) // 3 + 1

