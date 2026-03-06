
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import yaml

from core.config.logging import get_logger
from ...execution.analyzer import extract_dependencies
from ...retrieval import NAME_STOPWORDS
from ...schema import Skill, get_delta_meta
from ..auditor import SecurityAuditor, Severity

logger = get_logger(__name__)

_SEMANTIC_TAGS_PATH = Path(__file__).parent.parent / "semantic_tags.yaml"


def _load_semantic_tags() -> dict:
    try:
        if _SEMANTIC_TAGS_PATH.exists():
            with open(_SEMANTIC_TAGS_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
            logger.warning("semantic_tags.yaml  dict %s", type(data).__name__)
    except Exception as e:
        logger.warning(" semantic_tags.yaml : %s", e)
    return {}


_SEMANTIC_TAGS: dict = _load_semantic_tags()




def to_kebab_case(name: str) -> str:
    s = re.sub(r"([A-Z])", r"_\1", name).lower()
    s = re.sub(r"[_\s]+", "-", s)
    return s.strip("-")


def to_title(kebab_name: str) -> str:
    return " ".join(word.capitalize() for word in kebab_name.split("-"))




def is_python_code(code: str) -> bool:
    if not code or not code.strip():
        return False
    if code.lstrip().startswith("---"):
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False




def parse_skill_md(skill_md_path: Path) -> dict:
    content = skill_md_path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        raise ValueError(f"Invalid SKILL.md: missing YAML frontmatter in {skill_md_path}")
    frontmatter = yaml.safe_load(match.group(1))
    return frontmatter if isinstance(frontmatter, dict) else {}


  




def generate_skill_md(
    name: str,
    title: str,
    description: str,
    function_name: str,
    script_filename: str,
    parameters: dict,
    dependencies: list,
    extra_content: str = "",
    *,
    execution_mode: str | None = None,
    entry_script: str | None = None,
) -> str:
    frontmatter: dict = {
        "name": name,
        "description": description,
    }

    delta_meta: dict = {}
    if function_name:
        delta_meta["function_name"] = function_name
    if parameters:
        delta_meta["parameters"] = _sanitize_for_yaml(parameters)
    if dependencies:
        delta_meta["dependencies"] = dependencies
    if execution_mode:
        delta_meta["execution_mode"] = execution_mode
    if entry_script:
        delta_meta["entry_script"] = entry_script
    if delta_meta:
        frontmatter["metadata"] = {"delta-skills": delta_meta}

    fm_str = yaml.dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).rstrip("\n")

    if script_filename:
        usage_md = f"""\

Run the script to execute this skill:

```bash
python scripts/{script_filename}
```"""
    else:
        usage_md = """\

This is a knowledge/guideline skill. Follow the instructions below \
to accomplish the task."""

    params_section = ""
    if parameters:
        display_params = parameters.get("properties", parameters) if isinstance(parameters, dict) and "properties" in parameters else parameters
        params_lines = []
        for param_name, param_info in display_params.items():
            if isinstance(param_info, dict):
                param_type = param_info.get("type", "Any")
                param_desc = param_info.get("description", "")
                default = param_info.get("default")
                line = f"- `{param_name}` ({param_type})"
                if param_desc:
                    line += f": {param_desc}"
                if default is not None:
                    line += f" (default: `{default}`)"
                params_lines.append(line)
            else:
                params_lines.append(f"- `{param_name}`: {param_info}")
        params_section = f"""


{chr(10).join(params_lines)}"""

    deps_section = ""
    if dependencies:
        deps_md = "\n".join(f"- `{dep}`" for dep in dependencies)
        deps_section = f"""


{deps_md}"""

    extra_section = ""
    if extra_content:
        extra_section = f"""


{extra_content}"""

    return f"""\
---
{fm_str}
---


{description}

{usage_md}
{params_section}
{deps_section}
{extra_section}
"""




_PLACEHOLDER_FILES: dict[str, list[str]] = {
    "scripts": ["example.py"],
    "references": ["api_reference.md"],
    "assets": ["example_asset.txt"],
}


def _cleanup_placeholder_files(skill_dir: Path) -> None:
    for subdir, filenames in _PLACEHOLDER_FILES.items():
        for fname in filenames:
            fp = skill_dir / subdir / fname
            if fp.exists():
                try:
                    fp.unlink()
                    logger.debug("Removed placeholder: %s", fp)
                except OSError as e:
                    logger.warning("Failed to remove placeholder %s: %s", fp, e)




def _inject_execution_meta(
    content: str,
    execution_mode: str | None,
    entry_script: str | None,
) -> str:
    if not execution_mode and not entry_script:
        return content

    fm_match = re.match(r"^(---\s*\n)(.*?)(\n---)", content, re.DOTALL)
    if not fm_match:
        return content  # 

    try:
        frontmatter = yaml.safe_load(fm_match.group(2))
        if not isinstance(frontmatter, dict):
            return content
    except yaml.YAMLError:
        return content

    metadata = frontmatter.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return content
    delta = metadata.setdefault("delta-skills", {})
    if not isinstance(delta, dict):
        return content

    if execution_mode:
        delta["execution_mode"] = execution_mode
    if entry_script:
        delta["entry_script"] = entry_script

    new_fm_str = yaml.dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).rstrip("\n")

    rest = content[fm_match.end():]
    return f"---\n{new_fm_str}\n---{rest}"




def save_skill_to_disk(skill: Skill, skills_directory: Path) -> None:
    kebab_name = to_kebab_case(skill.name)
    skill_dir = skills_directory / kebab_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    _is_python = is_python_code(skill.code)

    if _is_python and not skill.execution_mode:
        skill.execution_mode = "function"
    elif not _is_python and not skill.execution_mode:
        skill.execution_mode = "knowledge"

    if _is_python:
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        script_filename = f"{skill.name}.py"
        (scripts_dir / script_filename).write_text(skill.code, encoding="utf-8")

        _cleanup_placeholder_files(skill_dir)

        skill_md_content = generate_skill_md(
            name=kebab_name,
            title=to_title(kebab_name),
            description=skill.description,
            function_name=skill.name,
            script_filename=script_filename,
            parameters=skill.parameters,
            dependencies=skill.dependencies,
            execution_mode=skill.execution_mode,
            entry_script=skill.entry_script,
        )
        (skill_dir / "SKILL.md").write_text(skill_md_content, encoding="utf-8")
    else:
        skill_md_path = skill_dir / "SKILL.md"
        if skill.code.lstrip().startswith("---"):
            content = _inject_execution_meta(skill.code, skill.execution_mode, skill.entry_script)
            skill_md_path.write_text(content, encoding="utf-8")
        else:
            skill_md_content = generate_skill_md(
                name=kebab_name,
                title=to_title(kebab_name),
                description=skill.description,
                function_name=skill.name,
                script_filename="",
                parameters=skill.parameters,
                dependencies=skill.dependencies,
                extra_content=skill.code,
                execution_mode=skill.execution_mode,
                entry_script=skill.entry_script,
            )
            skill_md_path.write_text(skill_md_content, encoding="utf-8")

    skill.source_dir = str(skill_dir)
    skill.invalidate_cache()

    logger.debug("Skill saved to disk: %s (%s)", skill_dir, "python" if _is_python else "knowledge")


def load_skill_from_dir(skill_dir: Path) -> Skill:
    skill_md_path = skill_dir / "SKILL.md"
    scripts_dir = skill_dir / "scripts"

    if not skill_md_path.exists():
        raise FileNotFoundError(f"Missing SKILL.md in {skill_dir}")

    meta = parse_skill_md(skill_md_path)
    delta_meta = get_delta_meta(meta)

    code = ""
    is_python_skill = False
    skill_md_text = skill_md_path.read_text(encoding="utf-8")

    py_files = []
    if scripts_dir.exists():
        py_files = sorted(
            p for p in scripts_dir.glob("*.py")
            if p.name != "__init__.py"
        )
    has_scripts = bool(py_files)

    if has_scripts:
        skill_dir_name = skill_dir.name.replace("-", "_")
        main_script = next(
            (p for p in py_files if p.stem == skill_dir_name),
            py_files[0],
        )
        code = main_script.read_text(encoding="utf-8")
        is_python_skill = True
    else:
        code = skill_md_text
        refs_dir = skill_dir / "references"
        if refs_dir.exists():
            ref_parts: list[str] = []
            for ref_file in sorted(refs_dir.iterdir()):
                if ref_file.is_file() and ref_file.suffix in (".md", ".txt", ".rst"):
                    try:
                        ref_content = ref_file.read_text(encoding="utf-8").strip()
                        if ref_content:
                            ref_parts.append(
                                f"\n\n---\n## [Reference] {ref_file.name}\n\n{ref_content}"
                            )
                    except Exception:
                        pass
            if ref_parts:
                code += "".join(ref_parts)

    skill_name = (
        delta_meta.get("function_name")
        or meta.get("name", "").replace("-", "_")
        or skill_dir.name.replace("-", "_")
    )

    all_deps = extract_all_deps(meta, code, is_python_skill)
    delta_deps = delta_meta.get("dependencies", [])
    if isinstance(delta_deps, list):
        existing = set(all_deps)
        for d in delta_deps:
            if d not in existing:
                all_deps.append(d)

    raw_desc = meta.get("description", "")
    if isinstance(raw_desc, list):
        parts = []
        for item in raw_desc:
            if isinstance(item, dict):
                parts.extend(str(v) for v in item.values())
            else:
                parts.append(str(item))
        raw_desc = " ".join(parts)
    description = str(raw_desc) if raw_desc else ""
    tags = auto_generate_tags(skill_name, description, meta, code, skill_dir)
    parameters = delta_meta.get("parameters", meta.get("parameters", {}))

    execution_mode = delta_meta.get("execution_mode")
    entry_script = delta_meta.get("entry_script")

    return Skill(
        name=skill_name,
        description=description,
        code=code,
        parameters=parameters,
        dependencies=all_deps,
        tags=tags,
        source_dir=str(skill_dir),
        execution_mode=execution_mode,
        entry_script=entry_script,
    )


def load_all_skills(
    skills_directory: Path,
    version_manager: Any = None,
) -> dict[str, Skill]:
    cache: dict[str, Skill] = {}
    if not skills_directory.exists():
        return cache

    loaded_count = 0
    dir_to_name: dict[str, str] = {}
    for skill_dir in sorted(skills_directory.iterdir()):
        if not skill_dir.is_dir():
            continue
        try:
            skill = load_skill_from_dir(skill_dir)

            if skill.name in cache and skill.name in dir_to_name:
                logger.warning(
                    "Name collision: '%s' from '%s' overwrites '%s'",
                    skill.name, skill_dir.name, dir_to_name[skill.name],
                )

            cache[skill.name] = skill
            dir_to_name[skill.name] = skill_dir.name
            loaded_count += 1

            if version_manager:
                try:
                    latest = version_manager.get_latest_version(skill.name)
                    if not latest:
                        logger.info("Syncing skill '%s' from disk to version DB", skill.name)
                        version_manager.save_version(
                            skill,
                            change_type="discovered",
                            change_note="Auto-discovered from disk during initialization",
                        )
                        skill.version = 1
                    else:
                        skill.version = latest.version
                except Exception as e:
                    logger.warning("Failed to sync skill '%s' to version DB: %s", skill.name, e)

        except Exception as e:
            logger.warning("Failed to load skill '%s': %s", skill_dir, e)

    if loaded_count > 0:
        unique_count = len(cache)
        if unique_count < loaded_count:
            logger.warning(
                "Loaded %d skill dir(s) but only %d unique name(s) — %d collision(s)",
                loaded_count, unique_count, loaded_count - unique_count,
            )
        logger.info("Loaded %d skill(s) from %s", unique_count, skills_directory)

        try:
            security_errors = 0
            for skill_dir in skills_directory.iterdir():
                if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
                    continue
                issues = SecurityAuditor.audit_directory(skill_dir)
                errors = [i for i in issues if i.severity == Severity.ERROR]
                if errors:
                    security_errors += len(errors)
                    for err in errors:
                        logger.warning("Security: [%s] %s", skill_dir.name, err.message)
            if security_errors:
                logger.warning("Security audit: %d issue(s) found across skills", security_errors)
        except Exception as e:
            logger.debug("Security audit skipped: %s", e)

    return cache




def extract_all_deps(meta: dict, code: str, is_python: bool) -> list:
    deps: set = set()

    declared = meta.get("dependencies", [])
    if isinstance(declared, list):
        deps.update(declared)

    runtime_meta = _extract_runtime_meta(meta)
    if isinstance(runtime_meta, dict):
        requires = runtime_meta.get("requires", {})
        if isinstance(requires, dict):
            bins = requires.get("bins", [])
            if isinstance(bins, list):
                deps.update(bins)

        install_list = runtime_meta.get("install", [])
        if isinstance(install_list, list):
            for item in install_list:
                if isinstance(item, dict):
                    for key in ("formula", "package"):
                        val = item.get(key)
                        if val and isinstance(val, str):
                            deps.add(val.split("/")[-1])

    if is_python and code:
        deps.update(extract_dependencies(code))

    if not is_python and code:
        deps.update(_scan_md_for_deps(code))

    return sorted(deps)


def _scan_md_for_deps(md_content: str) -> set:
    deps: set = set()
    for m in re.finditer(
        r"(?:pip|pip3|brew|apt|apt-get|npm)\s+install\s+(?:--?\S+\s+)*(\S+)",
        md_content,
    ):
        pkg = m.group(1).strip("`'\"")
        if pkg and not pkg.startswith("-"):
            deps.add(pkg.split("/")[-1])
    return deps




def auto_generate_tags(
    name: str, description: str, meta: dict, code: str, skill_dir: Path,
) -> list:
    tags: set = set()

    declared_tags = meta.get("tags", [])
    if isinstance(declared_tags, list):
        tags.update(t for t in declared_tags if isinstance(t, str))

    scripts_dir = skill_dir / "scripts"
    has_py = scripts_dir.exists() and any(scripts_dir.glob("*.py"))
    has_sh = scripts_dir.exists() and any(scripts_dir.glob("*.sh"))

    if has_py:
        tags.add("python")
    if has_sh:
        tags.add("shell")
    if not has_py:
        tags.add("knowledge")

    runtime_meta = _extract_runtime_meta(meta)
    if isinstance(runtime_meta, dict) and runtime_meta.get("requires", {}).get("bins"):
        tags.add("cli-tool")

    text = f"{name} {description}".lower()

    for tag, keywords in _SEMANTIC_TAGS.items():
        if keywords and any(kw in text for kw in keywords if kw):
            tags.add(tag)

    name_parts = name.lower().replace("-", "_").replace(" ", "_").split("_")
    for part in name_parts:
        if len(part) >= 3 and part not in NAME_STOPWORDS:
            tags.add(part)

    refs_dir = skill_dir / "references"
    assets_dir = skill_dir / "assets"
    if refs_dir.exists() and any(refs_dir.iterdir()):
        tags.add("has-references")
    if assets_dir.exists() and any(assets_dir.iterdir()):
        tags.add("has-assets")
    if has_py or has_sh:
        tags.add("has-scripts")

    return sorted(tags)




def get_skill_mtime(skill_dir: Path) -> float:
    if not skill_dir.exists():
        return 0
    mtime = skill_dir.stat().st_mtime
    for p in skill_dir.rglob("*"):
        if p.is_file():
            mtime = max(mtime, p.stat().st_mtime)
    return mtime


def _extract_runtime_meta(meta: dict) -> dict:
    raw_metadata = meta.get("metadata")
    if isinstance(raw_metadata, dict):
        return raw_metadata.get("runtime", {})
    if isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                return parsed.get("runtime", {})
        except (ValueError, TypeError):
            pass
    return {}


def _sanitize_for_yaml(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(item) for item in obj]
    if isinstance(obj, set):
        return sorted(_sanitize_for_yaml(item) for item in obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)
