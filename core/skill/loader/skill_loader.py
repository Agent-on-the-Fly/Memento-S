"""Skill Loader - 从磁盘加载技能

负责从磁盘加载 skill 文件，解析 SKILL.md 和关联资源。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import frontmatter

from core.skill.schema import Skill
from core.utils.text import to_kebab_case, to_snake_case
from utils.logger import get_logger

logger = get_logger(__name__)


class SkillLoader:
    """Skill 加载器 - 从磁盘加载技能文件

    职责：
    1. 从指定目录加载 skill
    2. 解析 SKILL.md frontmatter
    3. 加载 scripts/ 和 references/ 目录下的文件
    4. 构建 Skill 对象

    使用：
        loader = SkillLoader(Path("/path/to/skills"))
        skill = loader.load_from_dir(Path("/path/to/skills/my-skill"))
    """

    def __init__(self, skills_dir: Path | str) -> None:
        """初始化 SkillLoader

        Args:
            skills_dir: skill 根目录（用于相对路径计算）
        """
        self._skills_dir = Path(skills_dir)

    def load(self, name: str) -> Skill | None:
        """按名称加载 skill

        将名称转换为 kebab-case 后在 skills_dir 下查找。

        Args:
            name: skill 名称（支持 snake_case, camelCase, PascalCase）

        Returns:
            Skill 对象，未找到返回 None
        """
        kebab_name = to_kebab_case(name)
        skill_dir = self._skills_dir / kebab_name

        if not skill_dir.exists():
            return None

        return self.load_from_dir(skill_dir, full=True)

    def load_from_dir(self, skill_dir: Path, full: bool = True) -> Skill:
        """从目录加载 skill

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

        # 解析 frontmatter
        meta = self._parse_skill_md(skill_md_path)

        # 读取 SKILL.md 完整内容
        content = skill_md_path.read_text(encoding="utf-8")

        # 提取 skill 名称：仅使用 SKILL.md 顶层 name（不读 metadata），缺失时回退目录名
        raw_name = str(meta.get("name") or skill_dir.name)
        skill_name = to_snake_case(raw_name)

        # 提取描述
        description = str(meta.get("description", ""))

        # 提取依赖
        declared_deps = meta.get("metadata", {}).get("dependencies", [])
        if not isinstance(declared_deps, list):
            declared_deps = []
        all_deps = sorted(set(declared_deps))

        # 提取其他元数据
        execution_mode = meta.get("metadata", {}).get("execution_mode")
        entry_script = meta.get("metadata", {}).get("entry_script")
        required_keys = meta.get("metadata", {}).get("required_keys") or []
        parameters = meta.get("metadata", {}).get("parameters")

        # 解析 allowed-tools
        allowed_tools = self._parse_allowed_tools(
            meta.get("metadata", {}).get("allowed-tools") or meta.get("allowed-tools")
        )

        # 根据 full 参数决定是否加载 scripts 和 references
        if full:
            scripts_dir = skill_dir / "scripts"
            refs_dir = skill_dir / "references"
            files = self._load_scripts(scripts_dir)
            references = self._load_references(refs_dir)
        else:
            # 轻量级模式：不加载 scripts 和 references
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
            execution_mode=execution_mode,
            entry_script=entry_script,
            required_keys=required_keys,
            parameters=parameters,
            allowed_tools=allowed_tools,
        )

    def _parse_skill_md(self, skill_md_path: Path) -> Dict[str, Any]:
        """解析 SKILL.md 文件并返回 frontmatter 字典

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

    def _load_scripts(self, scripts_dir: Path) -> Dict[str, str]:
        """加载 scripts 目录下的 Python 文件

        Args:
            scripts_dir: scripts 目录路径

        Returns:
            文件名到内容的字典
        """
        files: Dict[str, str] = {}
        if scripts_dir.exists():
            for py_file in sorted(scripts_dir.glob("*.py")):
                files[py_file.name] = py_file.read_text(encoding="utf-8")
            if files and "__init__.py" not in files:
                files["__init__.py"] = ""
        return files

    def _load_references(self, refs_dir: Path) -> Dict[str, str]:
        """加载 references 目录下的参考文件

        Args:
            refs_dir: references 目录路径

        Returns:
            文件名到内容的字典
        """
        references: Dict[str, str] = {}
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

    def _parse_allowed_tools(self, allowed_tools_raw: Any) -> list[str]:
        """解析 allowed-tools 配置

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
