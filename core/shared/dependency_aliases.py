"""Centralized dependency alias normalization.

Single source of truth for mapping import-style names to canonical install specs.
"""

from __future__ import annotations

import re

# Strip extras/version markers to derive dependency base name.
_VERSION_EXTRAS_RE = re.compile(r"[\[=<>!~].*$")

# Canonical alias table (lower-cased keys).
_DEPENDENCY_ALIASES: dict[str, str] = {
    # Office
    "pptx": "python-pptx",
    "docx": "python-docx",
    "xlsx": "openpyxl",
    # Data/ML
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    # CV / imaging
    "cv2": "opencv-python",
    "pil": "Pillow",
    # Web / utils
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    # Common security-sensitive mismatches
    "crypto": "pycryptodome",
    "jwt": "PyJWT",
    # Standard library (do not install)
    "sqlite3": "",
}


def strip_version_extras(spec: str) -> str:
    """Strip version specifiers and extras from a dependency spec."""
    return _VERSION_EXTRAS_RE.sub("", (spec or "")).strip()


def normalize_dependency_spec(spec: str) -> str:
    """Normalize a dependency spec using centralized aliases.

    Preserves version/extras suffix where possible.
    Returns empty string when dependency should be skipped (e.g., stdlib module).
    """
    raw = (spec or "").strip()
    if not raw:
        return ""

    base = strip_version_extras(raw)
    if not base:
        return ""

    mapped = _DEPENDENCY_ALIASES.get(base.lower(), base)
    if mapped == "":
        return ""

    if mapped == base:
        return raw

    suffix = raw[len(base) :] if raw.startswith(base) else ""
    return f"{mapped}{suffix}"


def normalize_dependency_name(name: str) -> str:
    """Normalize a bare module/package name to installable canonical name."""
    return normalize_dependency_spec(name)


def get_dependency_aliases() -> dict[str, str]:
    """Expose a copy of alias table for diagnostics/logging."""
    return dict(_DEPENDENCY_ALIASES)
