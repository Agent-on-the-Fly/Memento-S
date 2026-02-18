"""Shared JSONL catalog parsing helpers."""

from __future__ import annotations

import json
from typing import Any


def _parse_int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def choose_catalog_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    return max(
        entries,
        key=lambda e: (
            _parse_int_or_zero(e.get("stars")),
            _parse_int_or_zero(e.get("updatedAt")),
            len(str(e.get("description") or "")),
            -_parse_int_or_zero(e.get("_line")),
        ),
    )


def parse_catalog_jsonl_text(
    text: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    name_order: list[str] = []
    by_name: dict[str, list[dict[str, Any]]] = {}
    for line_no, raw_line in enumerate(str(text or "").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        name = str(obj.get("name") or "").strip()
        if not name:
            continue
        if name not in by_name:
            by_name[name] = []
            name_order.append(name)

        by_name[name].append(
            {
                "name": name,
                "description": str(obj.get("description") or "").strip(),
                "githubUrl": str(obj.get("githubUrl") or obj.get("github_url") or "").strip(),
                "skillUrl": str(obj.get("skillUrl") or "").strip(),
                "id": str(obj.get("id") or "").strip(),
                "author": str(obj.get("author") or "").strip(),
                "stars": _parse_int_or_zero(obj.get("stars")),
                "updatedAt": _parse_int_or_zero(obj.get("updatedAt")),
                "_line": line_no,
            }
        )

    skills: list[dict[str, Any]] = []
    for name in name_order:
        preferred = choose_catalog_entry(by_name.get(name) or [])
        if preferred is None:
            continue
        skill: dict[str, Any] = {
            "name": name,
            "description": str(preferred.get("description") or "").strip(),
        }
        github_url = str(preferred.get("githubUrl") or "").strip()
        if github_url:
            skill["githubUrl"] = github_url
        skills.append(skill)
    return skills, by_name
