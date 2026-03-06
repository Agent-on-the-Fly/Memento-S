
SKILL_TYPE_PROMPT = """\
You are a skill architect. Determine if this task requires executable Python code or \
a knowledge-based instruction guide.

Task: {request}

Rules:
- "code": The task involves computation, API calls, data processing, file I/O, \
or any concrete programmatic action that produces output.
- "knowledge": The task is about workflow guidance, best practices, configuration, \
documentation, integration instructions, or reference material that guides human/AI \
behavior rather than executing logic.

Respond with EXACTLY one word: code or knowledge"""

GENERATE_PROMPT = """\
You are a Python expert. Create a production-quality, self-contained Python script.

Function name: {name}
Task description: {request}

Requirements:
1. Define a function named EXACTLY: {name}
2. Provide type hints for ALL parameters and return value.
3. Write a comprehensive docstring:
   - First line: A clear, specific one-sentence summary (describe its actual capability, \
NOT just the task name).
   - Then "Args:" section with each parameter described in detail (type, purpose, \
default value if any).
   - Then "Returns:" section.
4. Handle errors gracefully with try/except where appropriate.
5. Import all necessary modules at the TOP of the file (outside the function).
6. **CRITICAL — CLI entry point**: After the function, include a `if __name__ == "__main__":` \
block that:
   - Uses `argparse` to accept ALL function parameters as CLI arguments.
   - Calls the function with parsed arguments.
   - Prints the result to stdout (use `print()` or `json.dumps()` for structured data).
   This ensures the script is executable both as an importable module AND as a standalone CLI tool.
7. Return the code ONLY — no markdown fences, no explanation, no commentary."""

KNOWLEDGE_SKILL_PROMPT = """\
You are an expert at creating Agent Skills following the Anthropic skill format.

Create a comprehensive SKILL.md for this task:
Skill name: {name}
Task description: {request}

The output MUST follow this exact structure:

---
name: {kebab_name}
description: >-
  [Write a detailed 1-2 sentence description. WHAT this skill does + WHEN to use it. \
Third person. Include specific trigger scenarios. \
Example: "Downloads videos from YouTube and other platforms using yt-dlp. \
Use when the user wants to download, save, or convert online videos."]
---

# [Title]

## Overview
[1-2 sentences explaining what this skill enables]

## Usage
[Step-by-step instructions, task-based operations, or reference material. \
Choose the structure that fits best.]

## Examples
[2-3 concrete usage examples with realistic user requests and expected outputs]

Rules:
1. Keep SKILL.md under 500 lines.
2. Be concise — only include information the AI doesn't already know.
3. Frontmatter `name` must be kebab-case (lowercase + hyphens).
4. Frontmatter `description` MUST be a real sentence, NOT a placeholder or TODO.
5. Frontmatter `description` must not contain angle brackets (< or >).

Output the complete SKILL.md content ONLY — no explanation, no wrapping."""

REFLECTION_PROMPT = """\
The python function you wrote failed.
Function Name: {name}
Error Traceback: {error}
Original Code:
{code}

Please fix the code based on the error. Return the FULL fixed code only.
Keep the `if __name__ == "__main__":` CLI entry point intact."""

EVOLVE_PROMPT = """\
You are a Python expert. You need to ENHANCE an existing function to support a new capability.

Function name: {name}
Current description: {current_description}
Current code:
{current_code}

{new_requirement}

1. Keep the SAME function name: {name}
2. Preserve ALL existing functionality — do NOT break any current behavior.
3. ADD the new capability by extending parameters, logic branches, or both.
4. Update the docstring to reflect the expanded capabilities.
5. Keep backward compatibility — existing callers should still work without changes.
6. Import all necessary modules at the top of the code (outside the function).
7. Keep or add the `if __name__ == "__main__":` CLI entry point with argparse.
8. Return the FULL updated code ONLY — no markdown fences, no explanation.
"""
