"""Skill writing and persistence - 落盘（下载后保存）.

将 Skill 对象保存到磁盘，生成符合 agentskills.io 规范的目录结构。
https://agentskills.io/specification
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from utils.logger import get_logger
from core.skill.schema import Skill, ExecutionMode
from core.skill.store.persistence.generator import generate_skill_md
from core.skill.store.persistence.reader import extract_frontmatter
from core.skill.store.persistence.utils import is_python_code, to_kebab_case

logger = get_logger(__name__)

_PLACEHOLDER_FILES: dict[str, list[str]] = {
    "scripts": ["example.py"],
    "references": ["api_reference.md"],
    "assets": ["example_asset.txt"],
}


def _cleanup_placeholder_files(skill_dir: Path) -> None:
    """清理 init_skill 模板生成的占位文件"""
    for subdir, filenames in _PLACEHOLDER_FILES.items():
        for fname in filenames:
            fp = skill_dir / subdir / fname
            if fp.exists():
                try:
                    fp.unlink()
                    logger.debug("Removed placeholder: {}", fp)
                except OSError as e:
                    logger.warning("Failed to remove placeholder {}: {}", fp, e)


def _inject_execution_meta(
    content: str,
    execution_mode: str | None,
    entry_script: str | None,
) -> str:
    """向已有 frontmatter 的 SKILL.md 注入 execution_mode / entry_script。"""
    if not execution_mode and not entry_script:
        return content

    fm_match = extract_frontmatter(content)
    if not fm_match:
        return content

    try:
        frontmatter = yaml.safe_load(fm_match.group(1))
        if not isinstance(frontmatter, dict):
            return content
    except yaml.YAMLError:
        return content

    if execution_mode:
        frontmatter["execution_mode"] = execution_mode
    if entry_script:
        frontmatter["entry_script"] = entry_script

    new_fm_str = yaml.dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).rstrip("\n")

    rest = content[fm_match.end() :]
    return f"---\n{new_fm_str}\n---{rest}"


def save_skill_to_disk(skill: Skill, skills_directory: Path) -> None:
    """将技能保存为符合 Anthropic Skill 规范的目录结构"""
    kebab_name = to_kebab_case(skill.name)
    skill_dir = skills_directory / kebab_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    _is_python = is_python_code(skill.content)

    if _is_python and not skill.execution_mode:
        skill.execution_mode = ExecutionMode.PLAYBOOK
    elif not _is_python and not skill.execution_mode:
        skill.execution_mode = ExecutionMode.KNOWLEDGE

    if _is_python:
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        script_filename = f"{skill.name}.py"
        (scripts_dir / script_filename).write_text(skill.content, encoding="utf-8")

        _cleanup_placeholder_files(skill_dir)

        # 构建 metadata（包含向后兼容的字段）
        metadata: dict[str, Any] = {
            "function_name": skill.name,
        }
        if skill.dependencies:
            metadata["dependencies"] = skill.dependencies
        if skill.execution_mode:
            metadata["execution_mode"] = (
                skill.execution_mode.value
                if hasattr(skill.execution_mode, "value")
                else str(skill.execution_mode)
            )
        if skill.entry_script:
            metadata["entry_script"] = skill.entry_script
        if skill.required_keys:
            metadata["required_keys"] = skill.required_keys

        # 生成 instructions
        instructions = f"""Run the script to execute this skill:

```bash
python scripts/{script_filename}
```"""

        skill_md_content = generate_skill_md(
            name=kebab_name,
            description=skill.description,
            metadata=metadata,
            instructions=instructions,
        )
        (skill_dir / "SKILL.md").write_text(skill_md_content, encoding="utf-8")
    else:
        skill_md_path = skill_dir / "SKILL.md"
        if skill.content.lstrip().startswith("---"):
            content = _inject_execution_meta(
                skill.content, skill.execution_mode, skill.entry_script
            )
            skill_md_path.write_text(content, encoding="utf-8")
        else:
            # 构建 metadata（包含向后兼容的字段）
            metadata: dict[str, Any] = {
                "function_name": skill.name,
            }
            if skill.dependencies:
                metadata["dependencies"] = skill.dependencies
            if skill.execution_mode:
                metadata["execution_mode"] = (
                    skill.execution_mode.value
                    if hasattr(skill.execution_mode, "value")
                    else str(skill.execution_mode)
                )
            if skill.entry_script:
                metadata["entry_script"] = skill.entry_script

            skill_md_content = generate_skill_md(
                name=kebab_name,
                description=skill.description,
                metadata=metadata,
                instructions=skill.content
                if not skill.content.lstrip().startswith("---")
                else "",
            )
            skill_md_path.write_text(skill_md_content, encoding="utf-8")

    skill.source_dir = str(skill_dir)

    logger.debug(
        "Skill saved to disk: {} ({})",
        skill_dir,
        "python" if _is_python else "knowledge",
    )
