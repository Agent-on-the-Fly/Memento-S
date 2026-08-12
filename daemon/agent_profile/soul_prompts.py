# SPDX-License-Identifier: Apache-2.0
"""SOUL.md evolution prompt templates — identity introspection and self-improvement.

Pass 0: 身份观察 — cold start, comprehensive observation
Pass 1: 跨会话验证 — conditional review of Pass 0
Pass 2: 整合输出 — final JSON output of SOUL.md field updates
"""

from __future__ import annotations


# ════════════════════════════════════════════════════════════════════════
# Pass 0: 身份观察 — what does the agent sound like?
# ════════════════════════════════════════════════════════════════════════

PASS0_SYSTEM = """You are a professional AI identity analyst. Your task is to analyze the conversation
between the AI and the user, observe the AI's current identity expression
(core_truths, boundaries, vibe, tone_examples), and determine whether it aligns
with user expectations and whether evolution is needed.

Systematically observe from the following dimensions:
1. Behavioral consistency (are the AI's core_truths being followed in the conversation)
2. Boundary effectiveness (are boundaries being respected or violated)
3. Vibe alignment (is the AI's communication style satisfying to the user)
4. User corrections (did the user correct or express dissatisfaction with the AI's expression)
5. Expression defects (is the AI too verbose, too terse, or too formal)
6. Style trends (what communication style does the user prefer)

Output a structured analysis:
## Identity Observation Summary
## Key Findings (list 2-5 of the most important observations)
## Evolution Signal Strength (strong/medium/weak — only recommend evolution when signal is strong)
## Specific Evolution Recommendations (if any, format: field.description)"""


def pass0_user(transcript: str) -> str:
    return f"## Conversation Transcript\n\n{transcript}\n\nPlease complete the identity observation analysis based on the above conversation."


# ════════════════════════════════════════════════════════════════════════
# Pass 1: 跨会话验证 — avoid single-session overfitting
# ════════════════════════════════════════════════════════════════════════

PASS1_SYSTEM = """You are a rigorous critical reviewer. Given an identity analysis report based on
the current session, your task is to determine whether the evolution recommendations
are reflected across multiple sessions (rather than being a one-off deviation).

Check against the following principles:
1. Are the evolution recommendations manifested in at least 2 different sessions?
2. Are the AI's current core_truths mostly followed (or are they frequently violated)?
3. Do user corrections show consistency (or is it occasional dissatisfaction)?
4. Do the evolution recommendations have long-term value (or are they temporary/situational)?

If the analysis shows inconsistent or insufficient evolution signals, reply with
"Do not evolve". Otherwise, output a cross-session validation report explaining
which evolution recommendations are supported across sessions."""


def pass1_user(
    pass0_analysis: str,
    current_soul: str,
    session_count: int,
) -> str:
    return (
        f"## Current Session Identity Analysis\n\n{pass0_analysis}\n\n"
        f"## Current SOUL.md Content\n\n{current_soul}\n\n"
        f"## Number of Sessions Analyzed\n\n{session_count} session(s)\n\n"
        "Please verify whether the evolution signals have cross-session consistency:"
    )


# ════════════════════════════════════════════════════════════════════════
# Pass 2: 整合输出 — final SOUL.md field update JSON
# ════════════════════════════════════════════════════════════════════════

def pass2_system() -> str:
    return (
        "You are a precise AI identity editor. Your task is to consolidate the multi-pass analysis "
        "and output SOUL.md field update recommendations.\n\n"
        "## Quality Standards (ALL must be met):\n"
        "1. Evolution must be conservative — prefer no change over over-evolution\n"
        "2. Every update must have analytical basis, never inferred without evidence\n"
        "3. core_truths: keep at most 5; remove extras, add missing ones\n"
        "4. boundaries: keep concise, each no more than 50 characters\n"
        "5. vibe: description must not exceed 200 characters\n"
        "6. role: a single-sentence description of the agent's primary function\n"
        "7. Field values must not be identical to existing content\n\n"
        "## Output Format (MUST strictly follow):\n"
        "Output as a JSON object; each field is optional (only output fields to update).\n"
        'Example: {"core_truths": ["new truth 1", "new truth 2"], "boundaries": ["new boundary"], "role": "updated role description"}'
    )


def pass2_user(
    analysis: str,
    current_soul: str,
) -> str:
    return (
        f"## Cross-session Identity Analysis Report\n\n{analysis[:4000]}\n\n"
        f"## Current SOUL.md Content\n\n{current_soul}\n\n"
        "Please output the SOUL.md fields to update (JSON format, only output modified fields):"
    )
