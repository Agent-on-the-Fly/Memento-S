# SPDX-License-Identifier: Apache-2.0
"""Prompts for attribution, targeted rewriting, and synthetic test judging."""

ATTRIBUTION_PROMPT = """\
You are the failure-attribution selector in a Read-Write reflective learning loop.
Assign credit for this failed trajectory to exactly one candidate skill.

Task:
{task}

Supervisor/Judge rationale:
{rationale}

Candidate skills:
{candidates}

Execution trace:
{trace}

Return ONLY JSON:
{{
  "skill_name": "one exact candidate name",
  "confidence": 0.0,
  "failure_mode": "specific, reusable failure mode",
  "rationale": "why this skill, citing trace evidence",
  "evidence": ["short evidence item"]
}}
Do not select a skill merely because it ran last. Prefer the skill whose instructions or code
most directly caused the failed behaviour. Never invent a skill name.
"""


REWRITE_PROMPT = """\
You are the skill rewriter in a Read-Write reflective learning loop.
Make a targeted, general fix to one skill folder. Preserve unrelated behaviour and never
hard-code the task's expected answer. You may update SKILL.md, scripts, references, or tests.

Task that exposed the failure:
{task}

Failure attribution:
{attribution}

Execution trace:
{trace}

Current skill folder (relative path -> text):
{files}

Return ONLY JSON:
{{
  "summary": "short description of the general fix",
  "files": {{
    "relative/path.ext": "complete replacement contents"
  }},
  "synthetic_test": {{
    "request": "a new test request that exercises the failure mode without copying the answer",
    "pass_criteria": "observable criteria for success"
  }}
}}

Rules:
- Every path must already exist or be a safe relative path inside the skill folder.
- Return complete contents for each changed file, not patches.
- Include at least one changed file and keep SKILL.md valid Agent Skills Markdown.
- The synthetic test must test the general failure mode, not the original exact answer.
- Do not add task IDs, exact question phrases, expected values, or answer-matching branches.
- Improve the reusable methodology, guardrails, tools, or scripts for this class of tasks.
"""


DISCOVERY_REWRITE_PROMPT = """\
You are the skill discovery component in a Read-Write reflective learning loop.
The existing skill has repeatedly failed and its measured utility is below the discovery
threshold. Rebuild the skill with a fundamentally different, more reliable methodology while
preserving its exact skill name. Work only inside the isolated candidate folder: the runtime
will test it and atomically deploy it only if every gate passes.

Task that exposed the latest failure:
{task}

Failure attribution:
{attribution}

Measured utility:
{utility}

Execution trace:
{trace}

Current skill folder (relative path -> text):
{files}

Return ONLY JSON:
{{
  "summary": "short description of the newly discovered approach",
  "files": {{
    "relative/path.ext": "complete replacement contents"
  }},
  "synthetic_test": {{
    "request": "a new test request that exercises the failure class",
    "pass_criteria": "observable criteria for success"
  }}
}}

Rules:
- Keep the existing frontmatter name exactly unchanged.
- Replace weak methodology rather than merely rephrasing the old instructions.
- You may add scripts, references, and tests when they make execution deterministic.
- Every path must be safe and relative to the skill folder; return complete file contents.
- Do not hard-code task IDs, exact question phrases, expected values, or answers.
- The synthetic test must be a different example of the general failure mode.
"""


SYNTHETIC_JUDGE_PROMPT = """\
You are the unit-test judge for an evolved agent skill.

Synthetic request:
{request}

Pass criteria:
{pass_criteria}

Candidate execution result:
{result}

Return ONLY JSON:
{{"passed": true, "score": 0.0, "rationale": "concise evidence-based judgment"}}
Use passed=true only when the result concretely satisfies the criteria and contains no
unresolved execution error.
"""
