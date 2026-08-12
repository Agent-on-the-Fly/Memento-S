# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for sandbox environments."""

from __future__ import annotations

import os
from pathlib import Path


def build_node_path_from_dir(search_root: Path) -> dict[str, str] | None:
    """Search upward for node_modules/ directories and build NODE_PATH.

    Node.js standard module resolution already walks upward to find node_modules/,
    so this is only needed when running from a working directory that is not
    under the skill root.  It ensures the skill's node_modules/ are reachable
    regardless of where the sandbox's work_dir happens to be.

    Returns None if no node_modules/ directories are found.
    """
    node_path_entries: list[str] = []

    current = search_root.resolve()
    for _ in range(10):  # max 10 levels up
        if (current / "node_modules").is_dir():
            node_path_entries.append(str(current / "node_modules"))

        parent = current.parent
        if parent == current:
            break
        current = parent

    if not node_path_entries:
        return None

    existing = os.environ.get("NODE_PATH", "")
    all_entries = node_path_entries + ([existing] if existing else [])
    return {"NODE_PATH": os.pathsep.join(e for e in all_entries if e)}


def auto_install_deps(
    target_dir: Path,
    package_managers: list[str] | None = None,
) -> str:
    """Detect lockfile and return the appropriate install prefix command.

    Returns an empty string if no lockfile is found.
    """
    if package_managers is None:
        package_managers = ["bun", "pnpm", "yarn", "npm"]

    target_dir = Path(target_dir)
    if (target_dir / "bun.lockb").exists() or (target_dir / "bun.lock").exists():
        return "bun install && "
    if (target_dir / "pnpm-lock.yaml").exists():
        return "pnpm install && "
    if (target_dir / "yarn.lock").exists():
        return "yarn install && "
    if (target_dir / "package-lock.json").exists() or (target_dir / "package.json").exists():
        if "npm" in package_managers:
            return "npm install && "

    return ""
