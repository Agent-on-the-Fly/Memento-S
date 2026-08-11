"""Check source and package artifacts for MLOSS release-readiness invariants."""

from __future__ import annotations

import argparse
import tarfile
import tomllib
import zipfile
from pathlib import Path

REQUIRED_ROOT_FILES = {
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
}
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
        project = tomllib.load(handle)["project"]
    if project.get("license") != "Apache-2.0":
        _fail("pyproject.toml must declare Apache-2.0")


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
