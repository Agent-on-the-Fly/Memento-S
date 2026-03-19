"""Persistence utilities - 工具类.

包含命名转换和代码检测等通用工具函数。
"""

from __future__ import annotations

import ast
import re


def is_python_code(code: str) -> bool:
    """判断内容是否为有效的 Python 代码"""
    if not code or not code.strip():
        return False
    if code.lstrip().startswith("---"):
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def to_kebab_case(name: str) -> str:
    """snake_case / camelCase → kebab-case（agentskills.io 规范要求）

    Examples:
        get_weather_mock -> get-weather-mock
        getWeatherMock   -> get-weather-mock
    """
    # 将大写字母前插入下划线，然后转小写
    s = re.sub(r"([A-Z])", r"_\1", name).lower()
    # 将下划线和空格替换为连字符
    s = re.sub(r"[_\s]+", "-", s)
    # 去除首尾的连字符
    return s.strip("-")


def to_title(kebab_name: str) -> str:
    """kebab-case → Title Case，用于 SKILL.md 标题"""
    return " ".join(word.capitalize() for word in kebab_name.split("-"))
