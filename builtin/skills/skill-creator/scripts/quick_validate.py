#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Modified by Memento-Skills contributors, 2026; see THIRD_PARTY_NOTICES.md.
"""
Quick validation script for skills - validates one or all skills
"""

import sys
import re
import yaml
from pathlib import Path


def validate_skill(skill_path):
    """Basic validation of a skill. Returns (is_valid, message, errors)."""
    skill_path = Path(skill_path)

    errors = []

    # Check SKILL.md exists
    skill_md = skill_path / 'SKILL.md'
    if not skill_md.exists():
        errors.append("SKILL.md not found")
        return False, "SKILL.md not found", errors

    # Read and validate frontmatter
    content = skill_md.read_text()
    if not content.startswith('---'):
        errors.append("No YAML frontmatter found")
        return False, "No YAML frontmatter found", errors

    # Extract frontmatter
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        errors.append("Invalid frontmatter format")
        return False, "Invalid frontmatter format", errors

    frontmatter_text = match.group(1)

    # Parse YAML frontmatter
    try:
        frontmatter = yaml.safe_load(frontmatter_text)
        if not isinstance(frontmatter, dict):
            errors.append("Frontmatter must be a YAML dictionary")
            return False, "Frontmatter must be a YAML dictionary", errors
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML in frontmatter: {e}")
        return False, f"Invalid YAML in frontmatter: {e}", errors

    # Define allowed properties
    ALLOWED_PROPERTIES = {'name', 'description', 'license', 'allowed-tools', 'metadata', 'compatibility'}

    # Check for unexpected properties (excluding nested keys under metadata)
    unexpected_keys = set(frontmatter.keys()) - ALLOWED_PROPERTIES
    if unexpected_keys:
        err = (
            f"Unexpected key(s) in SKILL.md frontmatter: {', '.join(sorted(unexpected_keys))}. "
            f"Allowed properties are: {', '.join(sorted(ALLOWED_PROPERTIES))}"
        )
        errors.append(err)

    # Check required fields
    if 'name' not in frontmatter:
        errors.append("Missing 'name' in frontmatter")
    if 'description' not in frontmatter:
        errors.append("Missing 'description' in frontmatter")

    # Validate folder name matches kebab-case convention
    folder_name = skill_path.name
    if not re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$', folder_name):
        errors.append(
            f"Folder name '{folder_name}' must be kebab-case: lowercase letters, digits, and single hyphens only. "
            f"No underscores, uppercase, or spaces."
        )
    if '--' in folder_name:
        errors.append(f"Folder name '{folder_name}' cannot contain consecutive hyphens (--)")

    # Extract name for validation
    name = frontmatter.get('name', '')
    if isinstance(name, str):
        name = name.strip()
        if name:
            # Check naming convention: kebab-case only (lowercase letters, digits, single hyphens)
            # Reject: underscores, uppercase, spaces, leading/trailing/consecutive hyphens
            if not re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$', name):
                errors.append(
                    f"Name '{name}' must be kebab-case: lowercase letters, digits, and single hyphens only. "
                    f"Examples: qrcode-decoder, web-search, plan-mode. "
                    f"Invalid: qrcode_decoder, WebSearch, plan mode."
                )
            if '--' in name:
                errors.append(f"Name '{name}' cannot contain consecutive hyphens (--)")
            # Check name length (max 64 characters per spec)
            if len(name) > 64:
                errors.append(f"Name is too long ({len(name)} characters). Maximum is 64 characters.")
            # Check that frontmatter name matches folder name
            if name != folder_name:
                errors.append(
                    f"Name mismatch: frontmatter name is '{name}' but folder name is '{folder_name}'. "
                    f"These must match exactly. Rename the folder or update the frontmatter."
                )
        else:
            errors.append("Name is empty")

    # Extract and validate description
    description = frontmatter.get('description', '')
    if isinstance(description, str):
        description = description.strip()
        if description:
            # Check for angle brackets
            if '<' in description or '>' in description:
                errors.append("Description cannot contain angle brackets (< or >)")
            # Check description length (max 1024 characters per spec)
            if len(description) > 1024:
                errors.append(f"Description is too long ({len(description)} characters). Maximum is 1024 characters.")
        else:
            errors.append("Description is empty")

    # Validate compatibility field if present (optional)
    compatibility = frontmatter.get('compatibility', '')
    if compatibility and isinstance(compatibility, str) and len(compatibility) > 500:
        errors.append(f"Compatibility is too long ({len(compatibility)} characters). Maximum is 500 characters.")

    if errors:
        return False, errors[0], errors
    return True, "Skill is valid!", errors


def validate_all_skills(skills_root):
    """Validate all skills under a root directory. Returns (valid_count, invalid_skills)."""
    skills_root = Path(skills_root)
    valid_count = 0
    invalid_skills = []

    # List all subdirectories (each is a skill)
    for skill_dir in sorted(skills_root.iterdir()):
        if not skill_dir.is_dir():
            continue
        # Skip hidden dirs and known non-skill dirs
        if skill_dir.name.startswith('.') or skill_dir.name == '__pycache__':
            continue

        rel_path = skill_dir.relative_to(skills_root)
        valid, _, errors = validate_skill(skill_dir)
        if valid:
            valid_count += 1
        else:
            invalid_skills.append((rel_path, errors))

    return valid_count, invalid_skills


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate skill(s). Use --all to check all skills in builtin/skills/."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to a single skill directory. If omitted along with --all, defaults to builtin/skills/."
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Validate all skills under builtin/skills/"
    )
    parser.add_argument(
        "--root",
        default="builtin/skills",
        help="Root directory for --all mode (default: builtin/skills)"
    )
    args = parser.parse_args()

    if args.all or args.path is None:
        # Validate all skills
        root = Path(args.root)
        if not root.exists():
            print(f"Error: root directory not found: {root}")
            sys.exit(1)

        valid_count, invalid_skills = validate_all_skills(root)
        total = valid_count + len(invalid_skills)

        print(f"\n=== Validation Results ({total} skills) ===\n")
        print(f"  Valid:   {valid_count}")
        print(f"  Invalid: {len(invalid_skills)}")

        if invalid_skills:
            print(f"\n{'='*60}")
            for skill_path, errors in invalid_skills:
                print(f"\n[INVALID] {skill_path}")
                for err in errors:
                    print(f"  - {err}")
            print(f"\n{'='*60}")
            sys.exit(1)
        else:
            print(f"\nAll {total} skills are valid!")
            sys.exit(0)
    else:
        # Validate single skill
        skill_path = Path(args.path)
        valid, message, errors = validate_skill(skill_path)
        if not valid:
            print(f"Invalid skill: {skill_path}")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
        else:
            print(message)
            sys.exit(0)
