# SPDX-License-Identifier: Apache-2.0
"""Skill Loader - 从磁盘加载技能

负责从磁盘加载 skill 文件，解析 SKILL.md 和关联资源。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import frontmatter

from core.skill.schema import Skill
from utils.strings import to_kebab_case, to_snake_case
from utils.logger import get_logger

logger = get_logger(__name__)


def _parse_allowed_tools(allowed_tools_raw: Any) -> list[str]:
    """解析 allowed-tools 配置。

    Args:
        allowed_tools_raw: 原始配置（字符串或列表）

    Returns:
        工具名称列表
    """
    allowed_tools = []
    if allowed_tools_raw:
        if isinstance(allowed_tools_raw, str):
            allowed_tools = [
                t.strip() for t in allowed_tools_raw.split() if t.strip()
            ]
        elif isinstance(allowed_tools_raw, list):
            allowed_tools = [
                str(t).strip() for t in allowed_tools_raw if str(t).strip()
            ]
    return allowed_tools


def _load_scripts(scripts_dir: Path) -> dict[str, str]:
    """加载 scripts 目录下的 Python 文件。

    Args:
        scripts_dir: scripts 目录路径

    Returns:
        文件名到内容的字典
    """
    files: dict[str, str] = {}
    if scripts_dir.exists():
        for py_file in sorted(scripts_dir.glob("*.py")):
            files[py_file.name] = py_file.read_text(encoding="utf-8")
        if files and "__init__.py" not in files:
            files["__init__.py"] = ""
    return files


def _load_references(refs_dir: Path) -> dict[str, str]:
    """加载 references 目录下的参考文件。

    Args:
        refs_dir: references 目录路径

    Returns:
        文件名到内容的字典
    """
    references: dict[str, str] = {}
    if refs_dir.exists():
        for ref_file in sorted(refs_dir.iterdir()):
            if ref_file.is_file() and ref_file.suffix in (".md", ".txt", ".rst"):
                try:
                    ref_content = ref_file.read_text(encoding="utf-8")
                    if ref_content.strip():
                        references[ref_file.name] = ref_content
                except Exception as e:
                    logger.warning(
                        "Failed to read reference file '{}': {}", ref_file.name, e
                    )
    return references


def _parse_skill_md(skill_md_path: Path) -> dict[str, Any]:
    """解析 SKILL.md 文件并返回 frontmatter 字典。

    Args:
        skill_md_path: SKILL.md 文件路径

    Returns:
        frontmatter 字典

    Raises:
        ValueError: 如果文件缺少 frontmatter 或解析失败
        FileNotFoundError: 如果文件不存在
    """
    try:
        post = frontmatter.load(str(skill_md_path))
        if not post.metadata:
            raise ValueError(
                f"Invalid SKILL.md: missing frontmatter in {skill_md_path}"
            )
        return post.metadata
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Invalid SKILL.md: failed to parse frontmatter in {skill_md_path}: {e}"
        )


def load_from_dir(skill_dir: Path, full: bool = True) -> Skill:
    """从目录加载 skill。

    解析 SKILL.md frontmatter，提取 name、description、license、compatibility、
    metadata 下的所有自定义字段，以及 scripts/references 目录内容。

    Args:
        skill_dir: skill 目录路径
        full: True=完整加载(含scripts/references), False=仅元数据(用于扫描)

    Returns:
        Skill 对象

    Raises:
        FileNotFoundError: 如果 SKILL.md 不存在
        ValueError: 如果 frontmatter 解析失败
    """
    skill_md_path = skill_dir / "SKILL.md"

    if not skill_md_path.exists():
        raise FileNotFoundError(f"Missing SKILL.md in {skill_dir}")

    meta = _parse_skill_md(skill_md_path)
    content = skill_md_path.read_text(encoding="utf-8")

    raw_name = str(meta.get("name") or skill_dir.name)
    skill_name = to_snake_case(raw_name)

    description = str(meta.get("description", ""))

    declared_deps = meta.get("metadata", {}).get("dependencies", [])
    if not isinstance(declared_deps, list):
        declared_deps = []
    all_deps = sorted(set(declared_deps))

    metadata = meta.get("metadata", {}) or {}
    # Allow execution_mode at top-level frontmatter or under metadata:
    if "execution_mode" not in metadata and meta.get("execution_mode") is not None:
        metadata = {**metadata, "execution_mode": meta["execution_mode"]}

    if full:
        files = _load_scripts(skill_dir / "scripts")
        references = _load_references(skill_dir / "references")
    else:
        files = {}
        references = {}

    return Skill(
        name=skill_name,
        description=description,
        content=content,
        dependencies=all_deps,
        version=0,
        files=files,
        references=references,
        source_dir=str(skill_dir),
        execution_mode=metadata.get("execution_mode"),
        entry_script=metadata.get("entry_script"),
        required_keys=metadata.get("required_keys") or [],
        parameters=metadata.get("parameters"),
        allowed_tools=_parse_allowed_tools(
            metadata.get("allowed-tools") or meta.get("allowed-tools")
        ),
        license=meta.get("license"),
        compatibility=meta.get("compatibility"),
    )


class SkillLoader:
    """Skill 加载器 - 从磁盘加载技能文件。

    职责：
    1. 从指定目录加载 skill
    2. 解析 SKILL.md frontmatter
    3. 加载 scripts/ 和 references/ 目录下的文件
    4. 构建 Skill 对象

    使用：
        loader = SkillLoader(Path("/path/to/skills"))
        skill = loader.load("my-skill")          # 按名称（自动 kebab-case）
        skill = loader.load_from_dir(Path("/path/to/skills/my-skill"))  # 直接路径
    """

    def __init__(self, skills_dir: Path | str) -> None:
        self._skills_dir = Path(skills_dir)

    def load(self, name: str) -> Skill | None:
        """按名称加载 skill（含 scripts/references）。

        将名称转换为 kebab-case 后在 skills_dir 下查找。
        如果 kebab-case 目录不存在，额外尝试 normalized 名称重试。

        Args:
            name: skill 名称（支持 snake_case, camelCase, PascalCase）

        Returns:
            Skill 对象，未找到返回 None
        """
        kebab_name = to_kebab_case(name)
        skill_dir = self._skills_dir / kebab_name

        if skill_dir.exists():
            try:
                return load_from_dir(skill_dir, full=True)
            except FileNotFoundError:
                pass

        normalized = kebab_name.replace("-", "_")
        if normalized != kebab_name:
            normalized_dir = self._skills_dir / normalized
            if normalized_dir.exists():
                try:
                    return load_from_dir(normalized_dir, full=True)
                except FileNotFoundError:
                    pass

        return None

    def load_from_dir(self, skill_dir: Path, full: bool = True) -> Skill:
        """从目录加载 skill。

        Args:
            skill_dir: skill 目录路径
            full: True=完整加载(含scripts/references), False=仅元数据(用于扫描)

        Returns:
            Skill 对象

        Raises:
            FileNotFoundError: 如果 SKILL.md 不存在
            ValueError: 如果 frontmatter 解析失败
        """
        return load_from_dir(skill_dir, full=full)
