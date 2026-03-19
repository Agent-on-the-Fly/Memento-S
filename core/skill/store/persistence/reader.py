"""Skill reading and parsing - 解析（读取 skill）.

从磁盘读取 skill 并解析为 Skill 对象，遵循 agentskills.io 规范。
https://agentskills.io/specification
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from utils.logger import get_logger
from core.skill.schema import Skill

logger = get_logger(__name__)

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def extract_frontmatter(content: str) -> re.Match | None:
    """从 SKILL.md 内容中提取 frontmatter match 对象。

    Returns:
        match 对象（group(1) 为 YAML 文本），无匹配则返回 None。
    """
    return FRONTMATTER_RE.match(content)


def parse_skill_md(skill_md_path: Path) -> dict:
    """解析 SKILL.md 的 YAML frontmatter.

    Args:
        skill_md_path: SKILL.md 文件路径

    Returns:
        frontmatter 字典

    Raises:
        ValueError: 如果缺少 frontmatter 或解析失败
    """
    content = skill_md_path.read_text(encoding="utf-8")
    match = extract_frontmatter(content)
    if not match:
        raise ValueError(
            f"Invalid SKILL.md: missing YAML frontmatter in {skill_md_path}"
        )
    frontmatter = yaml.safe_load(match.group(1))
    return frontmatter if isinstance(frontmatter, dict) else {}


def parse_frontmatter_value(raw_value: Any) -> str:
    """解析 frontmatter 字段值，处理列表等复杂类型.

    Args:
        raw_value: frontmatter 中的原始值

    Returns:
        解析后的字符串
    """
    if isinstance(raw_value, list):
        parts = []
        for item in raw_value:
            if isinstance(item, dict):
                parts.extend(str(v) for v in item.values())
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(raw_value) if raw_value else ""


def load_skill_from_dir(skill_dir: Path) -> Skill:
    """从单个 skill 目录加载 Skill 对象.

    遵循 agentskills.io 规范，支持以下结构：
    - skill-name/
      ├── SKILL.md          # 必需
      ├── scripts/          # 可选
      ├── references/       # 可选（单独存储，不拼接到 content）
      └── assets/           # 可选（当前不处理）

    Args:
        skill_dir: skill 目录路径

    Returns:
        Skill 对象

    Raises:
        FileNotFoundError: 如果缺少 SKILL.md
        ValueError: 如果解析失败
    """
    skill_md_path = skill_dir / "SKILL.md"
    scripts_dir = skill_dir / "scripts"
    refs_dir = skill_dir / "references"

    if not skill_md_path.exists():
        raise FileNotFoundError(f"Missing SKILL.md in {skill_dir}")

    # 解析 frontmatter
    meta = parse_skill_md(skill_md_path)

    # 读取 SKILL.md 内容（不包括 frontmatter）
    skill_md_text = skill_md_path.read_text(encoding="utf-8")
    content = skill_md_text

    # 加载 scripts 目录下的 Python 文件
    files: dict[str, str] = {}
    if scripts_dir.exists():
        for py_file in sorted(scripts_dir.glob("*.py")):
            files[py_file.name] = py_file.read_text(encoding="utf-8")
        if files and "__init__.py" not in files:
            files["__init__.py"] = ""

    # 加载 references 目录下的文件（单独存储，不拼接到 content）
    # 遵循渐进式披露原则：按需加载
    references: dict[str, str] = {}
    if refs_dir.exists():
        for ref_file in sorted(refs_dir.iterdir()):
            if ref_file.is_file() and ref_file.suffix in (".md", ".txt", ".rst"):
                try:
                    ref_content = ref_file.read_text(encoding="utf-8")
                    if ref_content.strip():
                        references[ref_file.name] = ref_content
                except Exception as e:
                    logger.debug(
                        "Failed to read reference file '{}': {}", ref_file.name, e
                    )

    # 从 frontmatter 提取基本字段
    skill_name = (
        meta.get("metadata", {}).get("function_name")
        or meta.get("name", "").replace("-", "_")
        or skill_dir.name.replace("-", "_")
    )

    description = parse_frontmatter_value(meta.get("description", ""))

    # 提取依赖（从 metadata）
    declared_deps = meta.get("metadata", {}).get("dependencies", [])
    if not isinstance(declared_deps, list):
        declared_deps = []
    all_deps = sorted(set(declared_deps))

    # 其他元数据
    execution_mode = meta.get("metadata", {}).get("execution_mode")
    entry_script = meta.get("metadata", {}).get("entry_script")
    required_keys = meta.get("metadata", {}).get("required_keys") or []

    # 兼容旧字段（如果 metadata 中没有，尝试顶层）
    if not execution_mode:
        execution_mode = meta.get("execution_mode")
    if not entry_script:
        entry_script = meta.get("entry_script")
    if not required_keys:
        required_keys = meta.get("required_keys") or []

    # 解析 parameters（OpenAI Function Schema 格式）
    parameters = meta.get("metadata", {}).get("parameters")

    # 解析 allowed-tools（agentskills.io 规范，从 metadata 或顶层读取）
    allowed_tools_raw = meta.get("metadata", {}).get("allowed-tools") or meta.get(
        "allowed-tools"
    )
    allowed_tools = []
    if allowed_tools_raw:
        if isinstance(allowed_tools_raw, str):
            allowed_tools = [t.strip() for t in allowed_tools_raw.split() if t.strip()]
        elif isinstance(allowed_tools_raw, list):
            allowed_tools = [
                str(t).strip() for t in allowed_tools_raw if str(t).strip()
            ]

    return Skill(
        name=skill_name,
        description=description,
        content=content,
        dependencies=all_deps,
        files=files,
        references=references,  # 新增：references 单独存储
        source_dir=str(skill_dir),
        execution_mode=execution_mode,
        entry_script=entry_script,
        required_keys=required_keys,
        parameters=parameters,
        allowed_tools=allowed_tools,
    )


def load_all_skills(
    skills_directory: Path,
) -> dict[str, Skill]:
    """从 skills/ 目录加载所有技能，附带安全审查.

    Args:
        skills_directory: skills 目录路径

    Returns:
        技能名称到 Skill 对象的映射
    """
    cache: dict[str, Skill] = {}
    if not skills_directory.exists():
        return cache

    loaded_count = 0
    dir_to_name: dict[str, str] = {}
    for skill_dir in sorted(skills_directory.iterdir()):
        if not skill_dir.is_dir():
            continue
        if not (skill_dir / "SKILL.md").exists():
            continue
        try:
            skill = load_skill_from_dir(skill_dir)

            if skill.name in cache and skill.name in dir_to_name:
                logger.warning(
                    "Name collision: '{}' from '{}' overwrites '{}'",
                    skill.name,
                    skill_dir.name,
                    dir_to_name[skill.name],
                )

            cache[skill.name] = skill
            dir_to_name[skill.name] = skill_dir.name
            loaded_count += 1

        except Exception as e:
            logger.warning("Failed to load skill '{}': {}", skill_dir.name, e)

    if loaded_count > 0:
        unique_count = len(cache)
        if unique_count < loaded_count:
            logger.warning(
                "Loaded {} skill dir(s) but only {} unique name(s) — {} collision(s)",
                loaded_count,
                unique_count,
                loaded_count - unique_count,
            )
        logger.info("Loaded {} skill(s) from {}", unique_count, skills_directory.name)

    return cache
