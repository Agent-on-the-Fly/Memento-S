# SPDX-License-Identifier: Apache-2.0
"""USER.md evolution prompt templates — dialectic multi-pass reasoning.

Pass 0: Observation — cold/warm observation across multiple dimensions
Pass 1: Blind-spot critique — conditional review of Pass 0
Pass 2: Synthesis — output final atomic user facts as JSON

(从 prompts.py 重命名而来，prompts.py 保留向后兼容导出)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from daemon.agent_profile.constants import FACT_MAX_CHARS

if TYPE_CHECKING:
    pass


# ════════════════════════════════════════════════════════════════════════
# Pass 0: Observation
# ════════════════════════════════════════════════════════════════════════

PASS0_SYSTEM = """You are a professional user insight analyst. Your task is to deeply analyze the user from the conversation between the AI and the user, helping the AI better understand and serve this user.

Systematically observe from the following dimensions:
1. Preferences and habits (what the user likes/dislikes; any fixed patterns)
2. Work and communication style (professional vs casual; prefers long or short responses)
3. Expertise and background (what domains is the user proficient in; which topics do they understand deeply)
4. Current goals and needs (what goals or needs did the user express in the conversation)
5. Special agreements or corrections (did the user explicitly correct or make agreements with the AI)
6. Emotions and attitudes (does the user show specific emotions toward certain topics)
7. Implicit needs (what the user didn't say directly but can be inferred from behavior)

## IMPORTANT — What to EXCLUDE:
- **File paths, directory paths, or workspace locations**: NEVER extract paths like
  `/Users/.../workspace/output/`, `/path/to/file.pdf`, or similar as user facts.
  These are system-generated paths, not user preferences.
- **Tool-generated absolute paths in AI responses**: When the AI mentions file paths
  in its responses, these reflect system behavior, not user preferences. Focus on what
  the USER explicitly requested or corrected.
- **System configuration values**: Do not extract infrastructure or system-level details
  as user preferences.

Output a structured analysis report:
## Observation Summary
## Key Findings (list 3-6 of the most important facts)
## Confidence Assessment (high/medium/low for each finding)
## Facts vs Inferences (distinguish what was said directly vs what you inferred)"""


def pass0_user(transcript: str) -> str:
    return f"## Conversation Transcript\n\n{transcript}\n\nPlease complete the analysis based on the above conversation."


# ════════════════════════════════════════════════════════════════════════
# Pass 1: Blind-spot Critique
# ════════════════════════════════════════════════════════════════════════

PASS1_SYSTEM = """You are a rigorous critical analyst. Given a preliminary user analysis report and the original conversation, your task is to identify omissions and blind spots, and provide targeted supplementary insights.

Strictly check the following:
1. Are there any implicit user needs that were not discovered?
2. Are there any preferences/habits shown in the conversation that the analysis missed?
3. Are there any special aspects of the user's communication style (language preference, directness, formatting preferences)?
4. Are there any unstated but inferable expectations the user has of the AI?
5. Are there any contradictions or inconsistencies that need special annotation?
6. Does Pass 0 have any findings marked as low confidence — is there supplementary evidence?

If Pass 0 is already comprehensive with no obvious omissions, reply with "No additions needed".
Otherwise, output a supplementary analysis focusing on the gaps."""


def pass1_user(pass0_result: str, transcript: str) -> str:
    return (
        f"## Pass 0 Analysis Result\n\n{pass0_result}\n\n"
        f"## Original Conversation\n\n{transcript}\n\n"
        "Please supplement the blind spots in the above analysis "
        '(reply "No additions needed" if none):'
    )


# ════════════════════════════════════════════════════════════════════════
# Pass 2: Synthesis
# ════════════════════════════════════════════════════════════════════════

def pass2_system(max_facts: int) -> str:
    return (
        "You are a precise fact-synthesis assistant. Your task is to consolidate the user analysis "
        "report and distill the most important, valuable facts — up to "
        + str(max_facts)
        + " items.\n\n"
        "## Quality Standards (ALL must be met):\n"
        "1. Each fact must be atomic — single topic, not compound\n"
        "2. Facts must come directly from the conversation content, not from irrelevant inferences\n"
        "3. Avoid synonymous duplicates — if two facts say the same thing, keep the more precise one\n"
        "4. Prioritize long-term information that helps the AI reduce repeated explanations and improve service quality\n"
        "5. Each fact must not exceed " + str(FACT_MAX_CHARS) + " characters\n"
        "6. Assign each fact to exactly one section below\n\n"
        "## Section Definitions:\n"
        "- Identity & Preferences: who the user is, what they like/dislike, habits\n"
        "- Communication Style: formal/casual, verbosity preference, language preference\n"
        "- Expertise & Background: domains of knowledge, professional background, skills\n"
        "- Current Goals & Context: what the user is trying to achieve right now\n"
        "- Agreements & Corrections: explicit rules the user set, corrections made\n\n"
        "## CRITICAL — What NOT to Extract:\n"
        "- File paths, directory paths, or workspace locations (e.g. `/Users/.../workspace/output/`)\n"
        "- Absolute paths mentioned in AI responses (these are system behavior, not user preferences)\n"
        "- System configuration or infrastructure details\n\n"
        "## Output Format (MUST strictly follow):\n"
        "Output as a JSON object with section titles as keys and arrays of fact strings as values. "
        "Only include sections that have at least one fact. No other content.\n"
        'Example: {"Communication Style": ["Prefers concise replies"], "Identity & Preferences": ["Uses dark mode"]}'
    )


def pass2_user(analysis: str, existing_context: str, max_facts: int) -> str:
    existing_block = ""
    if existing_context:
        existing_block = (
            "\n\n## Existing User Profile\n\n"
            + existing_context
            + "\n\nIf a new fact has the same meaning as an existing fact above, skip it. "
            "Place each new fact under the most appropriate section."
        )
    return (
        f"## User Analysis Report\n\n{analysis[:4000]}\n\n"
        f"{existing_block}\n\n"
        f"Please output up to {max_facts} most important user facts (JSON object by section):"
    )
