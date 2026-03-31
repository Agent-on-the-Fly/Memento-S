"""通用代码解析工具"""

from __future__ import annotations

import ast


def parse_code(code: str) -> ast.Module | None:
    """安全解析 Python 代码，失败返回 None"""
    try:
        return ast.parse(code)
    except SyntaxError:
        return None
