"""Pure text helpers — zero dependencies on Memento-S internals."""

from __future__ import annotations

import re


def normalize_space(text: str) -> str:
    """Collapse all whitespace sequences to a single space."""
    return re.sub(r"\s+", " ", str(text or "").strip())


def extract_answer_text(model_text: str) -> str:
    """Extract the ``Answer: <value>`` line from model output."""
    text = str(model_text or "")
    m = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def extract_confidence_text(model_text: str) -> str:
    text = str(model_text or "")
    m = re.search(r"(?im)^\s*confidence\s*:\s*(.+?)\s*$", text)
    if not m:
        return ""
    return m.group(1).strip()


def normalize_confidence_text(raw_confidence: str) -> str:
    text = str(raw_confidence or "").strip()
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not m:
        return "70%"
    try:
        value = float(m.group(1))
    except Exception:
        return "70%"
    value = max(0.0, min(100.0, value))
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}%"
    return f"{value:.1f}%"


def looks_like_final_answer_format(text: str) -> bool:
    s = str(text or "")
    if not s.strip():
        return False
    has_explanation = re.search(r"(?im)^\s*explanation\s*:\s*.+$", s) is not None
    has_answer = re.search(r"(?im)^\s*answer\s*:\s*.+$", s) is not None
    has_confidence = re.search(r"(?im)^\s*confidence\s*:\s*.+$", s) is not None
    return has_explanation and has_answer and has_confidence


def brief_explanation_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "Used intermediate results to derive the final answer."
    for line in raw.splitlines():
        cur = str(line or "").strip()
        if not cur:
            continue
        lower = cur.lower()
        if lower.startswith("explanation:"):
            exp = cur.split(":", 1)[1].strip()
            if exp:
                return normalize_space(exp)
            continue
        if lower.startswith("answer:") or lower.startswith("confidence:"):
            continue
        return normalize_space(cur)
    return "Used intermediate results to derive the final answer."


def ensure_final_answer_format(text: str) -> str:
    """Enforce the canonical Explanation/Answer/Confidence format."""
    raw = str(text or "").strip()
    if looks_like_final_answer_format(raw):
        return raw
    answer = normalize_space(extract_answer_text(raw))
    if not answer:
        answer = normalize_space(raw)
    explanation = brief_explanation_text(raw)
    confidence = normalize_confidence_text(extract_confidence_text(raw))
    return f"Explanation: {explanation}\nAnswer: {answer}\nConfidence: {confidence}"


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def strip_markdown_fence(text: str) -> str:
    raw = str(text or "").strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if lines:
        lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def json_fragment(text: str) -> dict | None:
    """Try to extract a JSON dict from *text*."""
    import json
    s = str(text or "").strip()
    if not s:
        return None
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    return item
        return None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def json_list_fragment(text: str) -> list[dict] | None:
    """Try to extract a JSON list of dicts from text."""
    import json
    s = str(text or "").strip()
    if not s:
        return None
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            dicts = [item for item in parsed if isinstance(item, dict)]
            return dicts if dicts else None
        return None
    except Exception:
        pass
    m = re.search(r"\[[\s\S]*\]", s)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        if isinstance(parsed, list):
            dicts = [item for item in parsed if isinstance(item, dict)]
            return dicts if dicts else None
        return None
    except Exception:
        return None


def normalize_skill_name(name: str) -> str:
    """Normalize a skill name for comparison."""
    return str(name or "").strip().lower().replace("_", "-")
