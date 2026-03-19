"""Skill 持久化子模块入口.

按功能划分为四个模块：
- writer: 落盘（下载后保存）
- generator: 生成（创建新 skill）
- reader: 解析（读取 skill）
- utils: 工具类

遵循 agentskills.io 规范: https://agentskills.io/specification
"""

# 落盘相关
from .writer import save_skill_to_disk

# 生成相关
from .generator import (
    generate_skill_md,
    validate_name,
    validate_description,
    validate_compatibility,
)

# 解析相关
from .reader import (
    extract_frontmatter,
    load_skill_from_dir,
    load_all_skills,
    parse_skill_md,
    parse_frontmatter_value,
)

# 工具类
from .utils import (
    is_python_code,
    to_kebab_case,
    to_title,
)

__all__ = [
    # 落盘
    "save_skill_to_disk",
    # 生成
    "generate_skill_md",
    "validate_name",
    "validate_description",
    "validate_compatibility",
    # 解析
    "extract_frontmatter",
    "load_skill_from_dir",
    "load_all_skills",
    "parse_skill_md",
    "parse_frontmatter_value",
    # 工具
    "is_python_code",
    "to_kebab_case",
    "to_title",
]
