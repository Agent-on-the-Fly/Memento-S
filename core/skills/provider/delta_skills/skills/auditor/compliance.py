
import re
from pathlib import Path
from typing import List

import yaml

from .models import AuditIssue, Severity
from ...schema import EXECUTION_MODES

_ALLOWED_FM_KEYS = {"name", "description", "license", "allowed-tools", "metadata", "compatibility"}
_KEBAB_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def _try_fix_yaml(raw_yaml: str) -> dict | None:
    fixed_lines = []
    for line in raw_yaml.split("\n"):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            fixed_lines.append(line)
            continue

        m = re.match(r"^(\s*)([\w-]+):\s+(.+)$", line)
        if m:
            indent, key, value = m.group(1), m.group(2), m.group(3)
            if not (value.startswith('"') or value.startswith("'")):
                if ":" in value or value.startswith("{") or value.startswith("["):
                    escaped = value.replace('"', '\\"')
                    fixed_lines.append(f'{indent}{key}: "{escaped}"')
                    continue
        fixed_lines.append(line)

    try:
        result = yaml.safe_load("\n".join(fixed_lines))
        if isinstance(result, dict):
            return result
    except yaml.YAMLError:
        pass
    return None


class ComplianceAuditor:

    @staticmethod
    def audit(skill_dir: Path) -> List[AuditIssue]:
        issues: List[AuditIssue] = []

        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            issues.append(AuditIssue(
                Severity.ERROR, "compliance",
                "Missing SKILL.md",
                fix_hint="Create a SKILL.md with YAML frontmatter",
            ))
            return issues

        content = skill_md.read_text(encoding="utf-8")

        if not content.lstrip().startswith("---"):
            issues.append(AuditIssue(
                Severity.ERROR, "compliance",
                "SKILL.md missing YAML frontmatter (must start with ---)",
                fix_hint="Add YAML frontmatter block at the top",
            ))
            return issues

        fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not fm_match:
            issues.append(AuditIssue(
                Severity.ERROR, "compliance",
                "Invalid frontmatter format (missing closing ---)",
            ))
            return issues

        try:
            frontmatter = yaml.safe_load(fm_match.group(1))
        except yaml.YAMLError:
            frontmatter = _try_fix_yaml(fm_match.group(1))
            if frontmatter is None:
                issues.append(AuditIssue(
                    Severity.WARNING, "compliance",
                    "YAML frontmatter parse error (auto-fix failed). "
                    "Likely cause: unquoted value contains ':' or other YAML special chars.",
                    fix_hint="Wrap description/values containing ':' in quotes",
                    auto_fixable=True,
                ))
                return issues
            else:
                issues.append(AuditIssue(
                    Severity.WARNING, "compliance",
                    "YAML frontmatter required auto-fix to parse "
                    "(unquoted values with special chars like ':')",
                    fix_hint="Wrap description/values containing ':' in quotes",
                    auto_fixable=True,
                ))

        if not isinstance(frontmatter, dict):
            issues.append(AuditIssue(
                Severity.WARNING, "compliance",
                "Frontmatter is not a YAML dictionary",
            ))
            return issues

        non_standard = set(frontmatter.keys()) - _ALLOWED_FM_KEYS
        if non_standard:
            issues.append(AuditIssue(
                Severity.WARNING, "compliance",
                f"Non-standard frontmatter keys: {', '.join(sorted(non_standard))}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_FM_KEYS))}",
                fix_hint="Move custom fields to metadata.delta-skills namespace",
                auto_fixable=True,
            ))

        _check_name(frontmatter, skill_dir, issues)
        _check_description(frontmatter, issues)
        _check_execution_mode(frontmatter, issues)
        _check_directories(skill_dir, issues)

        return issues


def _check_name(frontmatter: dict, skill_dir: Path, issues: List[AuditIssue]) -> None:
    name = frontmatter.get("name")
    if not name:
        issues.append(AuditIssue(
            Severity.ERROR, "compliance",
            "Missing 'name' in frontmatter",
        ))
        return

    if not isinstance(name, str):
        return

    name = name.strip()
    if not _KEBAB_PATTERN.match(name):
        issues.append(AuditIssue(
            Severity.WARNING, "compliance",
            f"Name '{name}' is not kebab-case (lowercase + hyphens only)",
            fix_hint=f"Rename to: {name.lower().replace('_', '-')}",
            auto_fixable=True,
        ))
    if len(name) > 64:
        issues.append(AuditIssue(
            Severity.ERROR, "compliance",
            f"Name too long ({len(name)} chars, max 64)",
        ))
    dir_name = skill_dir.name
    if name != dir_name and name.replace("_", "-") != dir_name:
        issues.append(AuditIssue(
            Severity.WARNING, "compliance",
            f"Name '{name}' doesn't match directory name '{dir_name}'",
            fix_hint=f"Rename directory to '{name}' or update frontmatter name",
        ))


def _check_description(frontmatter: dict, issues: List[AuditIssue]) -> None:
    desc = frontmatter.get("description")
    if not desc:
        issues.append(AuditIssue(
            Severity.ERROR, "compliance",
            "Missing 'description' in frontmatter",
        ))
        return

    if not isinstance(desc, str):
        return

    desc = desc.strip()
    if len(desc) < 20:
        issues.append(AuditIssue(
            Severity.INFO, "compliance",
            f"Description is very short ({len(desc)} chars). "
            "A good description includes WHEN to use this skill.",
        ))
    if len(desc) > 1024:
        issues.append(AuditIssue(
            Severity.ERROR, "compliance",
            f"Description too long ({len(desc)} chars, max 1024)",
        ))
    if "<" in desc or ">" in desc:
        issues.append(AuditIssue(
            Severity.WARNING, "compliance",
            "Description contains angle brackets (< or >)",
            auto_fixable=True,
        ))


def _check_execution_mode(frontmatter: dict, issues: List[AuditIssue]) -> None:
    raw_metadata = frontmatter.get("metadata")
    if not isinstance(raw_metadata, dict):
        return
    delta = raw_metadata.get("delta-skills")
    if not isinstance(delta, dict):
        return
    mode = delta.get("execution_mode")
    if mode is None:
        return  #  — 
    if mode not in EXECUTION_MODES:
        issues.append(AuditIssue(
            Severity.ERROR, "compliance",
            f"Invalid execution_mode '{mode}'. "
            f"Must be one of: {', '.join(EXECUTION_MODES)}",
            fix_hint=f"Set execution_mode to one of: {', '.join(EXECUTION_MODES)}",
        ))


def _check_directories(skill_dir: Path, issues: List[AuditIssue]) -> None:
    unexpected_dirs = []
    for item in skill_dir.iterdir():
        if item.is_dir() and item.name not in (
            "scripts", "references", "assets", "__pycache__", ".git",
            "node_modules", ".venv", "venv",
        ):
            unexpected_dirs.append(item.name)
    if unexpected_dirs:
        issues.append(AuditIssue(
            Severity.INFO, "compliance",
            f"Non-standard directories: {', '.join(unexpected_dirs)}. "
            "Standard: scripts/, references/, assets/",
        ))
