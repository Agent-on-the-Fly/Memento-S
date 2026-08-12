# SPDX-License-Identifier: Apache-2.0
"""Check source and package artifacts for MLOSS release-readiness invariants."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path

REQUIRED_ROOT_FILES = {
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
}
EXPECTED_VERSION = "0.4.0"
SPDX_MARKER = "SPDX-License-Identifier: Apache-2.0"
PROJECT_PYTHON_ROOTS = (
    "build_scripts",
    "builtin",
    "cli",
    "core",
    "daemon",
    "gui",
    "im",
    "infra",
    "middleware",
    "scripts",
    "server",
    "shared",
    "tests",
    "tools",
    "utils",
)
SPDX_TEXT_FILES = (
    ".github/workflows/ci.yml",
    "builtin/skills/filesystem/SKILL.md",
    "builtin/skills/uv-pip-install/SKILL.md",
    "builtin/skills/web-search/SKILL.md",
    "middleware/storage/migrations/script.py.mako",
    "pyproject.toml",
    "requirements-dev.txt",
    "requirements-prod.txt",
)
REQUIRED_BUILTINS = {
    "filesystem",
    "skill-creator",
    "uv-pip-install",
    "web-search",
}
RETIRED_PATHS = {
    "3rd/weixin_sdk",
    "builtin/skills/docx",
    "builtin/skills/pdf",
    "builtin/skills/pptx",
    "builtin/skills/xlsx",
}


def _fail(message: str) -> None:
    raise SystemExit(f"release-readiness check failed: {message}")


def check_source_tree(root: Path) -> None:
    missing = sorted(
        name for name in REQUIRED_ROOT_FILES if not (root / name).is_file()
    )
    if missing:
        _fail(f"missing required root files: {', '.join(missing)}")

    present_retired = sorted(path for path in RETIRED_PATHS if (root / path).exists())
    if present_retired:
        _fail(
            f"retired, non-distributable paths are present: {', '.join(present_retired)}"
        )

    missing_builtins = sorted(
        name
        for name in REQUIRED_BUILTINS
        if not (root / "builtin" / "skills" / name / "SKILL.md").is_file()
    )
    if missing_builtins:
        _fail(f"missing expected built-in skills: {', '.join(missing_builtins)}")

    with (root / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    project = metadata["project"]
    if project.get("license") != "Apache-2.0":
        _fail("pyproject.toml must declare Apache-2.0")

    version_text = (root / "version.py").read_text(encoding="utf-8")
    version_match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)', version_text, re.MULTILINE
    )
    if not version_match:
        _fail("version.py does not define __version__")
    with (root / "middleware/config/system_config.json").open(encoding="utf-8") as handle:
        system_version = json.load(handle).get("version")
    versions = {
        "project": project.get("version"),
        "Flet": metadata["tool"]["flet"]["app"].get("build_version"),
        "Briefcase": metadata["tool"]["briefcase"].get("version"),
        "version.py": version_match.group(1),
        "system config": system_version,
    }
    inconsistent = {
        name: value for name, value in versions.items() if value != EXPECTED_VERSION
    }
    if inconsistent:
        details = ", ".join(f"{name}={value!r}" for name, value in inconsistent.items())
        _fail(f"version fields must be {EXPECTED_VERSION}: {details}")

    project_python = [root / "bootstrap.py", root / "version.py"]
    for directory in PROJECT_PYTHON_ROOTS:
        project_python.extend((root / directory).rglob("*.py"))
    spdx_missing = []
    for path in (*project_python, *(root / name for name in SPDX_TEXT_FILES)):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("builtin/skills/skill-creator/"):
            continue
        if SPDX_MARKER not in path.read_text(encoding="utf-8"):
            spdx_missing.append(relative)
    if spdx_missing:
        _fail(f"project files missing Apache-2.0 SPDX markers: {', '.join(sorted(spdx_missing))}")


def _archive_members(path: Path) -> set[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return set(archive.namelist())
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return set(archive.getnames())
    _fail(f"unsupported distribution artifact: {path.name}")
    return set()


def check_distributions(dist_dir: Path) -> None:
    artifacts = sorted((*dist_dir.glob("*.whl"), *dist_dir.glob("*.tar.gz")))
    if not artifacts:
        _fail(f"no wheel or source archive found in {dist_dir}")

    for artifact in artifacts:
        normalized = {
            member.replace("\\", "/") for member in _archive_members(artifact)
        }
        for retired in RETIRED_PATHS:
            if any(f"/{retired}/" in f"/{member}/" for member in normalized):
                _fail(f"{artifact.name} contains retired path {retired}")
        for required in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
            if not any(Path(member).name == required for member in normalized):
                _fail(f"{artifact.name} does not contain {required}")
        if artifact.suffix == ".whl":
            if "version.py" not in normalized:
                _fail(f"{artifact.name} does not contain version.py")
        else:
            for required in ("CITATION.cff", "SECURITY.md", "version.py"):
                if not any(Path(member).name == required for member in normalized):
                    _fail(f"{artifact.name} does not contain {required}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist",
        type=Path,
        help="also inspect wheel and source archives in this directory",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    check_source_tree(root)
    if args.dist:
        check_distributions(args.dist.resolve())
    print("release-readiness checks passed")


if __name__ == "__main__":
    main()
