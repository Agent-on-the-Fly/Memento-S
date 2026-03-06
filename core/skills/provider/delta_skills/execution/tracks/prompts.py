
EXECUTE_PROMPT = """\
You are a task executor. Generate a **complete, self-contained Python script** \
that accomplishes the user's request by following the skill specification below.

## Skill Description
{description}

## Skill Content
{skill_content}

## Parameter Schema
{parameters}

---

## User Query
{query}

## Provided Parameters
{params_json}

---

## Rules
1. Generate a COMPLETE, SELF-CONTAINED Python script.
2. Import ALL necessary modules at the top.
3. Follow the skill's instructions, rules, and patterns faithfully.
4. If the skill contains a ready-to-use Python function, include it and call it directly.
5. If the skill is a knowledge/guideline document, generate code that implements \
the guidelines for the user's specific request.
6. Use the provided parameters where applicable.
7. Print the final result using `print()` so it can be captured.
8. Do NOT wrap the entire script in `try/except`. Let exceptions propagate \
so the sandbox can detect failures. Only use `try/except` for specific recoverable errors.
9. Do NOT use interactive input (`input()`, GUI popups).
10. Do NOT use `sys.exit()`, `exit()`, `quit()`, or `raise SystemExit`.
11. Do NOT use `argparse` or parse command-line arguments — call functions directly.
12. Return ONLY executable Python code. No markdown fences, no explanations.
{available_modules_section}"""

KNOWLEDGE_EXECUTE_PROMPT = """\
You are an expert assistant. You have been given a **knowledge skill** and a user request.

## Skill Description
{description}

## Skill Content
{skill_content}

---

## User Query
{query}

## Provided Parameters
{params_json}

## Workspace
{workspace_path}

---

## Response Rules

1. **Relevance check**: If this skill is NOT relevant to the user's request, respond with \
ONLY: `[NOT_RELEVANT] <brief explanation>`. Then stop.

2. **Actionable skills** (terminal commands, filesystem operations, etc.):
   - If the skill describes operations like shell commands, file I/O, or system actions, \
you MUST produce a **complete, self-contained Python script** that performs the actual work.
   - Use `subprocess.run()` for shell commands, `pathlib.Path` for file operations.
   - Print results to stdout so the caller can capture them.
   - Do NOT output a JSON plan or abstract description — output executable Python code.
   - **All generated files MUST be saved under the workspace directory: `{workspace_path}`**. \
Create subdirectories as needed (e.g. `{workspace_path}/youtube_download/`). \
NEVER save files outside the workspace.

3. **Guidance / knowledge skills** (writing guidelines, coding standards, workflows, etc.):
   - Produce a complete, high-quality text response that directly addresses the user's request.
   - Follow the skill's instructions, patterns, and output format exactly.
   - Do NOT wrap your response in JSON, code blocks, or any container format.

4. **General rules**:
   - Be thorough and detailed — this is the final output the user sees.
   - Use the provided parameters where applicable.
"""

REFLECT_PROMPT = """\
The execution of skill **{skill_name}** failed. Analyze the error and fix the RIGHT component.

## User Query
{query}

## Skill Source Code
{skill_code}

## Execution Code (Generated)
{execution_code}

## Error
{error}

## Sandbox Constraints
- Code runs in an **isolated subprocess** (separate Python process).
- The working directory contains the skill's `scripts/` files — import them directly.
- `sys.argv` is empty — do NOT rely on `argparse.parse_args()` without explicit arguments.
- No interactive input (`input()`, GUI, stdin).
- Do NOT use `sys.exit()`, `exit()`, `quit()`, or `raise SystemExit` \
— the sandbox treats ANY SystemExit (even code 0) as a failure.
- If the skill uses argparse, either:
  a) Call underlying functions directly (bypass argparse/main), or
  b) Set `sys.argv` explicitly before calling `parse_args()`.
- If the skill relies on an external CLI tool, call the underlying library \
functions instead of shelling out.
{available_modules_section}

## Diagnosis
Determine if the bug is in:
  A) **SKILL_FIX** — The skill function itself has a bug, missing feature, or wrong algorithm. \
Fix the skill function code.
  B) **EXEC_FIX** — The execution script has a wrong import, incorrect function call, \
or missing parameter. Fix only the execution script.

## Output Format (STRICT)
Line 1: `SKILL_FIX` or `EXEC_FIX` — exactly one token, nothing else.
Lines 2+: The FULL corrected code for ONLY the component that needs fixing.
No markdown fences, no explanations, no commentary.
"""
