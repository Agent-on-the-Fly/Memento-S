"""Skill content generation - 生成（创建新 skill）.

根据 agentskills.io 规范生成符合标准的 SKILL.md 文件。
https://agentskills.io/specification
"""

from __future__ import annotations

from typing import Any

import yaml

from core.skill.store.persistence.utils import to_title


def validate_name(name: str) -> tuple[bool, str | None]:
    """验证 skill name 是否符合 agentskills.io 规范.

    规范要求：
    - 1-64字符
    - 只能包含小写字母、数字和连字符
    - 不能以连字符开头或结尾
    - 不能有连续连字符
    - 必须匹配父目录名

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
    - 应描述 skill 做什么以及何时使用

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


def validate_compatibility(compatibility: str) -> tuple[bool, str | None]:
    """验证 compatibility 字段是否符合规范.

    规范要求：
    - 最多500字符（如果提供）
    - 用于说明环境要求、依赖工具等

    Args:
        compatibility: 要验证的兼容性说明

    Returns:
        (是否有效, 错误信息或 None)
    """
    if len(compatibility) > 500:
        return (
            False,
            f"compatibility must be <= 500 characters, got {len(compatibility)}",
        )

    return True, None


def generate_skill_md(
    name: str,
    description: str,
    *,
    # 可选字段
    license: str | None = None,
    compatibility: str | None = None,
    metadata: dict[str, Any] | None = None,
    allowed_tools: list[str] | None = None,
    # 内容
    instructions: str = "",
    examples: list[dict[str, str]] | None = None,
) -> str:
    """生成符合 agentskills.io 规范的 SKILL.md 内容.

    Args:
        name: skill 名称（必须符合命名规范）
        description: skill 描述（1-1024字符），应描述做什么和何时使用
        license: 许可证名称或文件引用
        compatibility: 兼容性说明，环境要求、系统包、网络访问等
        metadata: 额外元数据（建议包含 author, version, function_name 等）
        allowed_tools: 预批准的工具列表（实验性）
        instructions: 详细使用说明
        examples: 使用示例列表，每项包含 title 和 code

    Returns:
        生成的 SKILL.md 内容

    Raises:
        ValueError: 如果 name 或 description 不符合规范

    Example:
        >>> content = generate_skill_md(
        ...     name="pdf-processing",
        ...     description="Extract PDF text, fill forms, merge files.",
        ...     license="Apache-2.0",
        ...     metadata={"author": "example-org", "version": "1.0"},
        ...     instructions="Step-by-step instructions...",
        ... )
    """
    # 验证必填字段
    valid, error = validate_name(name)
    if not valid:
        raise ValueError(f"Invalid name: {error}")

    valid, error = validate_description(description)
    if not valid:
        raise ValueError(f"Invalid description: {error}")

    # 验证可选字段
    if compatibility:
        valid, error = validate_compatibility(compatibility)
        if not valid:
            raise ValueError(f"Invalid compatibility: {error}")

    # 构建 frontmatter（严格按规范顺序）
    frontmatter: dict = {
        "name": name,
        "description": description,
    }

    # 可选字段（按规范推荐的顺序）
    if license:
        frontmatter["license"] = license

    if compatibility:
        frontmatter["compatibility"] = compatibility

    if metadata:
        frontmatter["metadata"] = metadata

    if allowed_tools:
        frontmatter["allowed-tools"] = " ".join(allowed_tools)

    # 生成 YAML（保留字段顺序）
    fm_str = yaml.dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).rstrip("\n")

    # 生成 Markdown body（遵循渐进式披露原则）
    body_parts: list[str] = []

    # 标题和描述
    body_parts.extend(
        [
            f"# {to_title(name)}",
            "",
            description,
            "",
        ]
    )

    # 使用说明
    if instructions:
        body_parts.extend(
            [
                "## Instructions",
                "",
                instructions,
                "",
            ]
        )
    else:
        body_parts.extend(
            [
                "## Instructions",
                "",
                "Add step-by-step instructions here.",
                "",
            ]
        )

    # 示例
    if examples:
        body_parts.extend(
            [
                "## Examples",
                "",
            ]
        )
        for i, example in enumerate(examples, 1):
            title = example.get("title", f"Example {i}")
            code = example.get("code", "")
            body_parts.extend(
                [
                    f"### {title}",
                    "",
                    f"```\n{code}\n```" if code else "```\n# Add example here\n```",
                    "",
                ]
            )
    else:
        body_parts.extend(
            [
                "## Examples",
                "",
                "### Example 1: Basic usage",
                "",
                "```\n# Add example here\n```",
                "",
            ]
        )

    # 注意事项
    body_parts.extend(
        [
            "## Notes",
            "",
            "Add any important notes, edge cases, or limitations here.",
        ]
    )

    body = "\n".join(body_parts)

    return f"""---
{fm_str}
---

{body}
"""
