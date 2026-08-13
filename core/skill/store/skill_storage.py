# SPDX-License-Identifier: Apache-2.0
"""Skill 存储实现：本地文件存储 + 注册表管理。

组合 FileStorage 和 SkillRegistry 的功能，提供完整的 skill 持久化能力：
- 磁盘文件读写（SKILL.md + scripts/ + references/）
- 注册表管理（skill.json 的读写）
- 增量/全量同步

不再依赖 DB 或 Vector 存储。
"""

from __future__ import annotations

import ast
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Set

import yaml

from shared.schema import SkillConfig, ExecutionMode
from core.skill.loader import load_from_dir
from core.skill.registry import SkillRegistry
from core.skill.schema import Skill
from utils.strings import to_kebab_case, to_snake_case, to_title
from utils.logger import get_logger

logger = get_logger(__name__)


# ========== 验证函数 ==========


def validate_name(name: str) -> tuple[bool, str | None]:
    """验证 skill name 是否符合 agentskills.io 规范。

    规范要求：
    - 1-64字符
    - 只能包含小写字母、数字和连字符
    - 不能以连字符开头或结尾
    - 不能有连续连字符
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
    """验证 description 是否符合 agentskills.io 规范。

    规范要求：1-1024字符，必须非空。
    """
    if not description:
        return False, "description cannot be empty"
    if len(description) > 1024:
        return False, f"description must be 1-1024 characters, got {len(description)}"
    return True, None


def _is_python_code(code: str) -> bool:
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


# ========== SkillStorage 实现 ==========


class SkillStorage:
    """Skill 存储实现：本地文件存储 + 注册表管理。

    组合了原有 FileStorage 的磁盘读写能力和 SkillRegistry 的注册表管理能力，
    提供完整的 skill 持久化接口。

    不再依赖 DB 或 Vector 存储。
    """

    def __init__(self, skills_dir: Path, registry: SkillRegistry) -> None:
        self._skills_dir = Path(skills_dir)
        self._registry = registry
        self._initialized = False

    async def init(self) -> None:
        """创建 skills 目录"""
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        logger.info("SkillStorage initialized: {}", self._skills_dir)

    async def close(self) -> None:
        """文件存储无需关闭"""
        pass

    @property
    def is_ready(self) -> bool:
        return self._initialized

    # ── 磁盘读写 ──────────────────────────────────────────────────────────

    async def save(self, name: str, skill: Skill) -> None:
        """保存 skill 到磁盘。

        推断 execution_mode，生成规范的目录结构：
        - PLAYBOOK：创建 scripts/，写入 {name}.py + 规范 SKILL.md
        - KNOWLEDGE：直接写入 SKILL.md（含 frontmatter）
        - 处理 references/ 目录
        """
        skill_dir = self._resolve_skill_dir(name, skill)
        skill_dir.mkdir(parents=True, exist_ok=True)

        is_python = _is_python_code(skill.content)
        if not skill.execution_mode:
            skill.execution_mode = (
                ExecutionMode.PLAYBOOK if is_python else ExecutionMode.KNOWLEDGE
            )

        kebab_name = to_kebab_case(name)

        if is_python and skill.execution_mode == ExecutionMode.PLAYBOOK:
            self._write_playbook(skill, skill_dir, kebab_name)
        else:
            self._write_knowledge(skill, skill_dir, kebab_name)

        if skill.references:
            refs_dir = skill_dir / "references"
            refs_dir.mkdir(exist_ok=True)
            for filename, content in skill.references.items():
                (refs_dir / filename).write_text(content, encoding="utf-8")

        logger.debug("Saved skill to disk: {}", skill_dir)

    def _resolve_skill_dir(self, name: str, skill: Skill) -> Path:
        """解析 skill 目录路径。优先使用已有的 source_dir。"""
        if skill.source_dir and Path(skill.source_dir).exists():
            return Path(skill.source_dir)
        return self._skills_dir / to_kebab_case(name)

    def _write_playbook(
        self, skill: Skill, skill_dir: Path, kebab_name: str
    ) -> None:
        """写入 PLAYBOOK 类型 skill：scripts/ + SKILL.md"""
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        script_filename = f"{kebab_name}.py"
        (scripts_dir / script_filename).write_text(skill.content, encoding="utf-8")

        metadata: dict = {"function_name": kebab_name}
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

        valid, error = validate_name(kebab_name)
        if not valid:
            raise ValueError(f"Invalid skill name: {error}")
        valid, error = validate_description(skill.description or "")
        if not valid:
            raise ValueError(f"Invalid skill description: {error}")

        fm_dict: dict[str, Any] = {
            "name": kebab_name,
            "description": skill.description or "",
            "metadata": metadata,
        }
        if skill.license:
            fm_dict["license"] = skill.license
        if skill.compatibility:
            fm_dict["compatibility"] = skill.compatibility
        fm_str = yaml.dump(
            fm_dict,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ).rstrip("\n")

        instructions = f"""Run the script to execute this skill:

```bash
python scripts/{script_filename}
```"""

        content = f"""---
{fm_str}
---

# {to_title(kebab_name)}

{skill.description or ""}

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
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    def _write_knowledge(
        self, skill: Skill, skill_dir: Path, kebab_name: str
    ) -> None:
        """写入 KNOWLEDGE 类型 skill：SKILL.md（含 frontmatter）"""
        if skill.content.lstrip().startswith("---"):
            (skill_dir / "SKILL.md").write_text(skill.content, encoding="utf-8")
        else:
            fm_dict: dict[str, Any] = {
                "name": kebab_name,
                "description": skill.description or "",
            }
            if skill.dependencies:
                fm_dict["metadata"] = {"dependencies": skill.dependencies}
            if skill.license:
                fm_dict["license"] = skill.license
            if skill.compatibility:
                fm_dict["compatibility"] = skill.compatibility

            fm_str = yaml.dump(
                fm_dict,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            ).rstrip("\n")

            content = f"""---
{fm_str}
---

{skill.content}
"""
            (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    async def load(self, name: str) -> Skill | None:
        """从磁盘加载单个 skill"""
        try:
            return load_from_dir(self._skills_dir / to_kebab_case(name))
        except FileNotFoundError:
            return None

    async def delete(self, name: str) -> bool:
        """删除 skill 目录"""
        kebab_name = to_kebab_case(name)
        skill_dir = self._skills_dir / kebab_name

        if skill_dir.exists():
            shutil.rmtree(skill_dir)
            logger.info("Deleted skill directory: {}", skill_dir)
            return True
        return False

    async def list_names(self) -> Set[str]:
        """扫描目录返回所有 skill 名称"""
        names: Set[str] = set()
        if not self._skills_dir.exists():
            return names

        for skill_dir in self._skills_dir.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                try:
                    skill = load_from_dir(skill_dir)
                    names.add(skill.name)
                except Exception as e:
                    logger.warning(
                        "Failed to load skill from '{}': {}", skill_dir.name, e
                    )
        return names

    # ── 注册表操作 ────────────────────────────────────────────────────────

    async def add_skill(self, skill: Skill) -> None:
        """添加 skill 到文件存储并更新注册表"""
        storage_name = to_kebab_case(to_snake_case(skill.name))

        await self.save(storage_name, skill)

        self._registry.register(storage_name, {
            "location": str((self._skills_dir / storage_name).resolve()),
            "source": "local",
            "version": skill.version or 1,
            "installed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "active",
            "tags": [],
        })

        logger.info("SkillStorage.add_skill: {}", skill.name)

    async def remove_skill(self, name: str) -> bool:
        """从文件存储和注册表删除 skill"""
        storage_name = to_kebab_case(to_snake_case(name))

        deleted = await self.delete(storage_name)
        self._registry.unregister(storage_name)

        if deleted:
            logger.info("SkillStorage.remove_skill: {}", storage_name)
        return deleted

    async def get_skill(self, name: str) -> Skill | None:
        """获取 skill（从文件加载完整数据）"""
        storage_name = to_kebab_case(to_snake_case(name))
        return await self.load(storage_name)

    async def list_all_skills(self) -> Dict[str, Skill]:
        """列出所有 skill（从注册表获取名称，从文件加载内容）"""
        skills: Dict[str, Skill] = {}
        all_names = list(self._registry.list_all().keys())
        for name in all_names:
            try:
                skill = await self.load(name)
                if skill:
                    skills[skill.name] = skill
            except Exception as e:
                logger.warning("Failed to load skill '{}': {}", name, e)
        return skills

    # ── 同步 ───────────────────────────────────────────────────────────────

    async def refresh_from_disk(self) -> int:
        """增量同步：从磁盘扫描新增 skill，添加到注册表"""
        added = 0
        if not self._skills_dir.exists():
            return added

        registry_skills = self._registry.list_all()
        for skill_dir in sorted(self._skills_dir.iterdir()):
            if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
                continue
            name = to_kebab_case(skill_dir.name)
            if name in registry_skills:
                continue
            try:
                skill = await self.load(name)
                if skill:
                    self._registry.register(name, {
                        "location": str(skill_dir.resolve()),
                        "source": "local",
                        "version": skill.version or 1,
                        "installed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "status": "active",
                        "tags": [],
                    })
                    added += 1
                    logger.info("refresh_from_disk: added {}", name)
            except Exception as e:
                logger.warning("refresh_from_disk: skip '{}': {}", skill_dir.name, e)

        return added

    async def sync_from_disk(
        self,
        skills_dir: Path | None = None,
        builtin_dir: Path | None = None,
    ) -> int:
        """全量同步：将磁盘 skill 同步到注册表"""
        if skills_dir is None:
            skills_dir = self._skills_dir

        result = self._registry.sync_from_disk(skills_dir, builtin_dir)
        count = len(result.get("added", [])) + len(result.get("unchanged", []))
        logger.info("sync_from_disk: {} skills synced", count)
        return count

    async def cleanup_orphans(self) -> list[str]:
        """清理孤儿：从注册表移除磁盘已删除的 skill"""
        cleaned: list[str] = []
        disk_names = await self.list_names()
        registry_names = set(self._registry.list_all().keys())
        disk_names_snake = {to_snake_case(n) for n in disk_names}
        for name in registry_names:
            if to_snake_case(name) not in disk_names_snake:
                self._registry.unregister(name)
                cleaned.append(name)
        if cleaned:
            logger.info("cleanup_orphans: removed {}", cleaned)
        return cleaned

    # ── 工厂方法 ──────────────────────────────────────────────────────────

    @classmethod
    async def from_config(cls, config: SkillConfig) -> "SkillStorage":
        """从配置异步构建 SkillStorage"""
        registry = SkillRegistry()
        store = cls(config.skills_dir, registry)
        await store.init()
        await store.sync_from_disk(config.skills_dir, config.builtin_skills_dir)
        return store
