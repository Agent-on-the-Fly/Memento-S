
import re
from pathlib import Path
from typing import Optional

import yaml

from core.config.logging import get_logger

logger = get_logger(__name__)

_MIGRATE_KEYS = {"function_name", "parameters", "dependencies"}


class SkillMigrator:

    @staticmethod
    def needs_migration(skill_dir: Path) -> bool:
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return False

        content = skill_md.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not fm_match:
            return False

        try:
            frontmatter = yaml.safe_load(fm_match.group(1))
        except yaml.YAMLError:
            return False

        if not isinstance(frontmatter, dict):
            return False

        return bool(set(frontmatter.keys()) & _MIGRATE_KEYS)

    @staticmethod
    def migrate(skill_dir: Path, dry_run: bool = False) -> Optional[str]:
        skill_md = skill_dir / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8")

        fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not fm_match:
            return None

        try:
            frontmatter = yaml.safe_load(fm_match.group(1))
        except yaml.YAMLError:
            return None

        if not isinstance(frontmatter, dict):
            return None

        delta_meta = {}
        new_fm = {}
        for key, value in frontmatter.items():
            if key in _MIGRATE_KEYS:
                delta_meta[key] = value
            else:
                new_fm[key] = value

        if not delta_meta:
            return None

        if "metadata" not in new_fm:
            new_fm["metadata"] = {}
        elif not isinstance(new_fm["metadata"], dict):
            new_fm["metadata"] = {}

        existing_delta = new_fm["metadata"].get("delta-skills", {})
        if isinstance(existing_delta, dict):
            existing_delta.update(delta_meta)
        else:
            existing_delta = delta_meta
        new_fm["metadata"]["delta-skills"] = existing_delta

        if "name" in new_fm:
            name = str(new_fm["name"])
            if "_" in name:
                new_fm["name"] = name.replace("_", "-")

        new_fm_str = yaml.dump(
            new_fm,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ).rstrip("\n")

        body = content[fm_match.end():]
        new_content = f"---\n{new_fm_str}\n---{body}"

        if dry_run:
            return new_content

        backup_path = skill_md.with_suffix(".md.bak")
        if not backup_path.exists():
            backup_path.write_text(content, encoding="utf-8")

        skill_md.write_text(new_content, encoding="utf-8")
        logger.info("Migrated SKILL.md in '%s' (backup: %s)", skill_dir.name, backup_path.name)
        return None
