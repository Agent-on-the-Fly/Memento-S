"""AST 解析与校验"""

from __future__ import annotations

import ast
import re

import yaml


def parse_code(code: str) -> ast.Module | None:
    """安全解析 Python 代码，失败返回 None"""
    try:
        return ast.parse(code)
    except SyntaxError:
        return None



def validate_skill_md(content: str) -> bool:
    """校验 SKILL.md 格式（含 YAML frontmatter + name/description）"""
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
