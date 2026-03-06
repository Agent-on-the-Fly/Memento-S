
from __future__ import annotations

import ast
import re

import yaml


def parse_code(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def validate_syntax(code: str) -> bool:
    return parse_code(code) is not None


def validate_skill_md(content: str) -> bool:
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return False
    try:
        frontmatter = yaml.safe_load(match.group(1))
        if not isinstance(frontmatter, dict):
            return False
        return "name" in frontmatter and "description" in frontmatter
    except yaml.YAMLError:
        return False
