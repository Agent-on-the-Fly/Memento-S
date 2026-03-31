"""Skill Builder - 从原始数据构建规范的 skill 目录结构

负责：
1. 推断 execution_mode
2. 分离 Python 代码到 scripts/
3. 生成规范的 SKILL.md（含 frontmatter）
4. 组织 references/
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml

from core.skill.schema import Skill, ExecutionMode
from core.utils.text import to_kebab_case, to_title
from utils.logger import get_logger

logger = get_logger(__name__)


# ========== 验证函数 ==========


def validate_name(name: str) -> tuple[bool, str | None]:
    """验证 skill name 是否符合 agentskills.io 规范.

    规范要求：
    - 1-64字符
    - 只能包含小写字母、数字和连字符
    - 不能以连字符开头或结尾
    - 不能有连续连字符

    Args:
        name: 要验证的 skill 名称

    Returns:
        (是否有效, 错误信息或 None)
    """
    if not name:
        return False, "name cannot be empty"

    if len(name) > 64:
        return False, f"name must be 1-64 characters, got {len(name)}"

    if name.startswith("-") or name.endswith("-"):
        return False, "name cannot start or end with hyphen"

    if "--" in name:
        return False, "name cannot contain consecutive hyphens"

    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-")
    invalid_chars = set(name) - allowed
    if invalid_chars:
        return False, f"name contains invalid characters: {invalid_chars}"

    return True, None


def validate_description(description: str) -> tuple[bool, str | None]:
    """验证 description 是否符合 agentskills.io 规范.

    规范要求：
    - 1-1024字符
    - 必须非空

    Args:
        description: 要验证的描述

    Returns:
        (是否有效, 错误信息或 None)
    """
    if not description:
        return False, "description cannot be empty"

    if len(description) > 1024:
        return False, f"description must be 1-1024 characters, got {len(description)}"

    return True, None


# ========== 辅助函数 ==========


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


@dataclass
class BuiltSkill:
    """构建好的 skill 结构"""

    skill_dir: Path
    skill: Skill
    files_created: Dict[str, Path]


class SkillBuilder:
    """Skill 构建器 - 从原始数据构建规范的 skill 目录结构

    职责：
    1. 推断 execution_mode
    2. 分离 Python 代码到 scripts/
    3. 生成规范的 SKILL.md（含 frontmatter）
    4. 组织 references/

    使用：
        builder = SkillBuilder()
        built = builder.build(skill, target_dir)
        # built.skill_dir 是构建好的目录
        # built.files_created 是创建的文件映射
    """

    def build(self, skill: Skill, target_dir: Path) -> BuiltSkill:
        """构建规范的 skill 目录结构

        Args:
            skill: 要构建的 Skill 对象
            target_dir: 目标目录（如 /path/to/skills/my-skill）

        Returns:
            BuiltSkill: 包含构建好的目录结构和元数据
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        files_created: Dict[str, Path] = {}

        # 推断 execution_mode（如果未设置）
        _is_python = is_python_code(skill.content)
        if not skill.execution_mode:
            if _is_python:
                skill.execution_mode = ExecutionMode.PLAYBOOK
            else:
                skill.execution_mode = ExecutionMode.KNOWLEDGE

        # 构建目录名（kebab-case）
        kebab_name = to_kebab_case(skill.name)

        # 如果是 Python 代码，分离到 scripts/
        if _is_python and skill.execution_mode == ExecutionMode.PLAYBOOK:
            self._build_playbook(skill, target_dir, kebab_name, files_created)
        else:
            self._build_knowledge(skill, target_dir, kebab_name, files_created)

        # 保存 references
        if skill.references:
            refs_dir = target_dir / "references"
            refs_dir.mkdir(exist_ok=True)
            for filename, content in skill.references.items():
                ref_path = refs_dir / filename
                ref_path.write_text(content, encoding="utf-8")
                files_created[f"references/{filename}"] = ref_path

        logger.info("Built skill at: {}", target_dir)
        return BuiltSkill(
            skill_dir=target_dir,
            skill=skill,
            files_created=files_created,
        )

    def _build_playbook(
        self,
        skill: Skill,
        skill_dir: Path,
        kebab_name: str,
        files_created: Dict[str, Path],
    ) -> None:
        """构建 PLAYBOOK 类型的 skill"""
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # 分离代码到 scripts/
        script_filename = f"{skill.name}.py"
        script_path = scripts_dir / script_filename
        script_path.write_text(skill.content, encoding="utf-8")
        files_created[f"scripts/{script_filename}"] = script_path

        # 构建元数据
        metadata = {
            "function_name": skill.name,
        }
        if skill.dependencies:
            metadata["dependencies"] = skill.dependencies
        if skill.execution_mode:
            metadata["execution_mode"] = skill.execution_mode.value
        if skill.entry_script:
            metadata["entry_script"] = skill.entry_script
        if skill.required_keys:
            metadata["required_keys"] = skill.required_keys
        if skill.allowed_tools:
            metadata["allowed-tools"] = skill.allowed_tools

        # 生成 SKILL.md 内容
        skill_md_content = self._generate_skill_md(
            kebab_name=kebab_name,
            description=skill.description,
            metadata=metadata,
            script_filename=script_filename,
        )

        skill_md_path = skill_dir / "SKILL.md"
        skill_md_path.write_text(skill_md_content, encoding="utf-8")
        files_created["SKILL.md"] = skill_md_path

    def _build_knowledge(
        self,
        skill: Skill,
        skill_dir: Path,
        kebab_name: str,
        files_created: Dict[str, Path],
    ) -> None:
        """构建 KNOWLEDGE 类型的 skill"""
        skill_md_path = skill_dir / "SKILL.md"

        # 如果 content 已有 frontmatter，直接写入
        if skill.content.lstrip().startswith("---"):
            skill_md_path.write_text(skill.content, encoding="utf-8")
        else:
            # 需要添加 frontmatter
            metadata = {
                "name": kebab_name,
                "description": skill.description,
            }
            if skill.dependencies:
                metadata["metadata"] = {"dependencies": skill.dependencies}

            fm_str = yaml.dump(
                metadata,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            ).rstrip("\n")

            content = f"""---
{fm_str}
---

{skill.content}
"""
            skill_md_path.write_text(content, encoding="utf-8")

        files_created["SKILL.md"] = skill_md_path

    def _generate_skill_md(
        self,
        kebab_name: str,
        description: str,
        metadata: dict,
        script_filename: str,
    ) -> str:
        """生成 PLAYBOOK 类型的 SKILL.md 内容

        包含验证，确保符合 agentskills.io 规范。
        """
        # 验证 name 和 description
        valid, error = validate_name(kebab_name)
        if not valid:
            raise ValueError(f"Invalid skill name: {error}")

        valid, error = validate_description(description)
        if not valid:
            raise ValueError(f"Invalid skill description: {error}")

        fm_str = yaml.dump(
            {
                "name": kebab_name,
                "description": description,
                "metadata": metadata,
            },
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ).rstrip("\n")

        instructions = f"""Run the script to execute this skill:

```bash
python scripts/{script_filename}
```"""

        return f"""---
{fm_str}
---

# {to_title(kebab_name)}

{description}

## Instructions

{instructions}

## Examples

### Example 1: Basic usage

```
# Add example here
```

## Notes

Add any important notes, edge cases, or limitations here.
"""
