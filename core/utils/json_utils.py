"""JSON utility functions for parsing, extracting, and repairing JSON from LLM output.

These are standalone functions with no dependencies on other agent modules.
"""

import json
import re
from typing import Any


def parse_json_output(text: str) -> Any:
    """Parse JSON from LLM output text, handling markdown fences and malformed JSON.

    Strips markdown code fences, extracts JSON candidates (objects/arrays),
    attempts repair of common issues (unescaped newlines, trailing commas),
    and returns the first successfully parsed result.

    Returns an empty dict if no valid JSON is found.
    """
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\r?\n", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
    candidates = extract_json_candidates(raw)
    for cand in candidates:
        cand = repair_json_string(cand)
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\\s*(?=[}\\]])", "", cand)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    return {}


def extract_json_candidates(text: str) -> list[str]:
    """Extract top-level JSON object/array substrings from *text*.

    Uses a simple brace/bracket depth counter that respects JSON string
    literals (including backslash escapes) so that braces inside strings
    are not miscounted.

    Returns a list of candidate substrings, each starting with ``{`` or
    ``[`` and ending with the matching ``}`` or ``]``.
    """
    candidates: list[str] = []
    start = None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : i + 1])
                start = None
    return candidates


def repair_json_string(raw: str) -> str:
    """Repair common JSON issues: unescaped newlines/tabs inside string literals.

    Walks the string character-by-character, tracking whether we are inside
    a JSON string literal.  Raw newline (``\\n``, ``\\r``) and tab
    (``\\t``) characters found inside strings are replaced with their
    escaped counterparts so that ``json.loads`` can succeed.
    """
    out: list[str] = []
    in_str = False
    escape = False
    for ch in raw:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            out.append(ch)
            in_str = not in_str
            continue
        if in_str and ch in ("\n", "\r"):
            out.append("\\n")
            continue
        if in_str and ch == "\t":
            out.append("\\t")
            continue
        out.append(ch)
    return "".join(out)
