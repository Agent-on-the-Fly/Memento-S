
import re
from pathlib import Path

import yaml

from core.config.logging import get_logger
from ..execution.analyzer import extract_dependencies
from ..schema import Skill, SkillImportError, get_delta_meta

logger = get_logger(__name__)


class SkillAdapter:

    @staticmethod
    def from_directory(skill_dir: Path) -> Skill:
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            raise SkillImportError(skill_dir.name, "Missing SKILL.md")

        meta = SkillAdapter._parse_frontmatter(skill_md_path)
        skill_name = meta.get("name", skill_dir.name)
        description = meta.get("description", "")

        code = ""
        files = {}
        scripts_dir = skill_dir / "scripts"

        if scripts_dir.exists():
            py_files = sorted(scripts_dir.glob("*.py"))

            for py_file in py_files:
                files[py_file.name] = py_file.read_text(encoding="utf-8")

            candidates = [f for f in py_files if f.name != "__init__.py"]
            if candidates:
                target_file = candidates[0]
                logger.info("Loading code from %s for skill %s", target_file, skill_name)
                code = files[target_file.name]
            elif py_files:
                code = files[py_files[0].name]

            if files and "__init__.py" not in files:
                files["__init__.py"] = ""

        if not code:
            code = skill_md_path.read_text(encoding="utf-8")

        dmeta = get_delta_meta(meta)

        function_name = (
            dmeta.get("function_name")
            or meta.get("function_name", "")
            or skill_name.replace("-", "_")
        )

        parameters = dmeta.get("parameters", meta.get("parameters", {}))
        declared_deps = dmeta.get("dependencies", meta.get("dependencies", []))
        if not isinstance(declared_deps, list):
            declared_deps = []

        analyzed_deps = extract_dependencies(code) if code else []
        all_deps = sorted(set(declared_deps) | set(analyzed_deps))

        execution_mode = dmeta.get("execution_mode")
        entry_script = dmeta.get("entry_script")

        logger.info(
            "Adapted skill '%s' from %s (code=%d bytes, params=%d, deps=%s, mode=%s)",
            function_name, skill_dir, len(code), len(parameters), all_deps, execution_mode,
        )

        return Skill(
            name=function_name,
            description=description,
            code=code,
            parameters=parameters if isinstance(parameters, dict) else {},
            dependencies=all_deps,
            files=files,
            source_dir=str(skill_dir),
            execution_mode=execution_mode,
            entry_script=entry_script,
        )

    @staticmethod
    def from_skill_md_text(text: str) -> Skill:
        match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
        if not match:
            raise SkillImportError("unknown", "Invalid SKILL.md: missing YAML frontmatter")
        try:
            frontmatter = yaml.safe_load(match.group(1))
        except yaml.YAMLError as e:
            raise SkillImportError("unknown", f"YAML parse error: {e}")
        if not isinstance(frontmatter, dict):
            raise SkillImportError("unknown", "YAML frontmatter is not a dict")
        dmeta = get_delta_meta(frontmatter)
        name = frontmatter.get("name", "unknown")
        function_name = (
            dmeta.get("function_name")
            or frontmatter.get("function_name", "")
            or name.replace("-", "_")
        )
        description = frontmatter.get("description", "")
        parameters = dmeta.get("parameters", frontmatter.get("parameters", {}))
        dependencies = dmeta.get("dependencies", frontmatter.get("dependencies", []))

        execution_mode = dmeta.get("execution_mode")
        entry_script = dmeta.get("entry_script")

        code = SkillAdapter._extract_code_from_markdown(text)
        return Skill(
            name=function_name,
            description=description,
            code=code,
            parameters=parameters if isinstance(parameters, dict) else {},
            dependencies=dependencies if isinstance(dependencies, list) else [],
            execution_mode=execution_mode,
            entry_script=entry_script,
        )

    @staticmethod
    def _parse_frontmatter(skill_md_path: Path) -> dict:
        content = skill_md_path.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not match:
            raise SkillImportError(
                skill_md_path.parent.name,
                "Invalid SKILL.md: missing YAML frontmatter",
            )

        try:
            frontmatter = yaml.safe_load(match.group(1))
            return frontmatter if isinstance(frontmatter, dict) else {}
        except yaml.YAMLError as e:
            raise SkillImportError(skill_md_path.parent.name, f"YAML error: {e}")

    @staticmethod
    def _extract_code_from_markdown(text: str) -> str:
        pattern = r"```python\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "\n\n".join(m.strip() for m in matches)
        return ""
