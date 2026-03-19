"""Template-driven config merge without versioning.

This merge strategy only adds missing keys from the template. It never
overwrites existing user values. For map-like sections, it also preserves
user-only keys and merges nested template defaults.
"""

from __future__ import annotations

from typing import Any


def merge_template_defaults(template: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    """Merge template defaults into user config without overwriting.

    Rules:
    - If a key is missing in user, copy it from template.
    - If both are dicts, recursively merge.
    - If user provides a value (including None), keep it as-is.
    - Preserve user-only keys that don't exist in template.
    """
    if not isinstance(template, dict) or not isinstance(user, dict):
        return user if user is not None else template

    merged: dict[str, Any] = {}

    for key, template_value in template.items():
        if key in user:
            user_value = user[key]
            if isinstance(template_value, dict) and isinstance(user_value, dict):
                merged[key] = merge_template_defaults(template_value, user_value)
            else:
                merged[key] = user_value
        else:
            merged[key] = template_value

    for key, user_value in user.items():
        if key not in merged:
            merged[key] = user_value

    return merged
