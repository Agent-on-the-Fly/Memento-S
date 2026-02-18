"""Skill resolution and dynamic fetch functions.

Extracted from agent.py to support modular architecture.
"""

import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

from core.config import (
    PROJECT_ROOT,
    SKILLS_DIR,
    SKILLS_EXTRA_DIRS,
    SKILL_DYNAMIC_FETCH_ENABLED,
    SKILL_DYNAMIC_FETCH_CATALOG_JSONL,
    SKILL_DYNAMIC_FETCH_ROOT,
    SKILL_DYNAMIC_FETCH_TIMEOUT_SEC,
    SEMANTIC_ROUTER_DEBUG,
)
from core.utils.logging_utils import log_event
from core.utils.path_utils import _run_command_capture, _NO_GIT_PROMPT_ENV
from core.skill_engine.skill_catalog import _load_router_catalog_from_jsonl, _choose_catalog_entry

_SKILL_FETCH_LOCK = threading.Lock()


def _iter_skill_roots() -> list[Path]:
    """Iterate over all configured skill directory roots.

    Returns resolved paths for SKILLS_DIR, each SKILLS_EXTRA_DIRS entry,
    and SKILL_DYNAMIC_FETCH_ROOT, de-duplicated and in order.
    """
    roots: list[Path] = []
    for raw_root in [SKILLS_DIR, *SKILLS_EXTRA_DIRS, SKILL_DYNAMIC_FETCH_ROOT]:
        p = raw_root
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        else:
            p = p.resolve()
        if p not in roots:
            roots.append(p)
    return roots


def _resolve_skill_dir(skill_name: str | None) -> Path | None:
    """Resolve a skill name to its directory path.

    Searches all skill roots (SKILLS_DIR, SKILLS_EXTRA_DIRS,
    SKILL_DYNAMIC_FETCH_ROOT) for a directory matching the skill name.

    Returns:
        Path to the skill directory if found, None otherwise.
    """
    if not isinstance(skill_name, str) or not skill_name.strip():
        return None
    for root in _iter_skill_roots():
        candidate = (root / skill_name.strip()).resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def has_local_skill_dir(skill_name: str) -> bool:
    """Check if a skill directory exists locally with a SKILL.md file.

    Args:
        skill_name: The name of the skill to check.

    Returns:
        True if a local directory with SKILL.md exists for the skill.
    """
    skill_dir = _resolve_skill_dir(skill_name)
    if skill_dir is None:
        return False
    return (skill_dir / "SKILL.md").exists()


def _parse_github_tree_url(github_url: str) -> tuple[str, str, str] | None:
    """Parse a GitHub tree/blob URL into (repo_url, ref, subpath).

    Handles URLs like:
        https://github.com/owner/repo/tree/main/path/to/skill
        https://github.com/owner/repo/blob/main/path/to/SKILL.md
        https://github.com/owner/repo

    Args:
        github_url: A GitHub URL string.

    Returns:
        Tuple of (clone_url, git_ref, subpath) or None if not a valid GitHub URL.
    """
    raw = str(github_url or "").strip()
    if not raw:
        return None
    try:
        parsed = urllib.parse.urlsplit(raw)
    except Exception:
        return None

    host = (parsed.netloc or "").lower()
    if host not in {"github.com", "www.github.com"}:
        return None

    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        return None
    owner = parts[0]
    repo = parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    ref = "HEAD"
    subpath = ""
    if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
        ref = parts[3]
        if len(parts) >= 5:
            subpath = "/".join(parts[4:])
        if parts[2] == "blob" and subpath:
            parent = str(Path(subpath).parent).replace("\\", "/")
            subpath = "" if parent == "." else parent

    repo_url = f"https://github.com/{owner}/{repo}.git"
    return repo_url, ref, subpath


def _pick_skill_dir_from_checkout(repo_root: Path, subpath: str, skill_name: str) -> Path | None:
    """Find the directory containing SKILL.md in a git checkout.

    Searches the checkout starting from the subpath (if given), then
    falls back to the repo root. Prefers directories whose name matches
    the skill_name.

    Args:
        repo_root: Root of the cloned repository.
        subpath: Subpath within the repo to search first.
        skill_name: Expected skill name (used for tie-breaking).

    Returns:
        Path to the directory containing SKILL.md, or None if not found.
    """
    search_root = repo_root
    if subpath:
        candidate = (repo_root / subpath).resolve()
        try:
            candidate.relative_to(repo_root.resolve())
        except Exception:
            candidate = repo_root
        if candidate.is_file():
            if candidate.name == "SKILL.md":
                return candidate.parent
            candidate = candidate.parent
        if candidate.is_dir() and (candidate / "SKILL.md").exists():
            return candidate
        if candidate.is_dir():
            search_root = candidate

    if (repo_root / "SKILL.md").exists():
        return repo_root

    candidates: list[Path] = []
    for md in search_root.rglob("SKILL.md"):
        candidates.append(md.parent)
    if not candidates and search_root != repo_root:
        for md in repo_root.rglob("SKILL.md"):
            candidates.append(md.parent)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (0 if p.name == skill_name else 1, len(p.parts)))
    return candidates[0]


def _parse_skill_repo_allowlist() -> set[str]:
    raw = str(os.getenv("SKILL_DYNAMIC_FETCH_ALLOWED_REPOS") or "").strip()
    if not raw:
        return set()
    out: set[str] = set()
    for token in raw.replace(";", ",").split(","):
        value = str(token or "").strip().lower().rstrip("/")
        if value:
            out.add(value)
    return out


def _is_allowed_repo(repo_url: str) -> tuple[bool, str]:
    allow = _parse_skill_repo_allowlist()
    if not allow:
        return True, ""

    raw = str(repo_url or "").strip().lower().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    host = str(parsed.netloc or "").strip()
    path = str(parsed.path or "").strip().strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    owner_repo = "/".join(path.split("/")[:2]) if path else ""

    candidates = {
        raw,
        f"{host}/{owner_repo}" if host and owner_repo else "",
        owner_repo,
    }
    candidates = {c for c in candidates if c}
    return any(c in allow for c in candidates), ",".join(sorted(candidates))


def ensure_skill_available(skill_name: str) -> tuple[bool, str]:
    """Download a skill from GitHub if it is not available locally.

    Looks up the skill in the JSONL catalog, clones the repo using
    sparse checkout, and copies the skill directory to
    SKILL_DYNAMIC_FETCH_ROOT.

    Args:
        skill_name: Name of the skill to fetch.

    Returns:
        Tuple of (success, message).
    """
    name = str(skill_name or "").strip()
    if not name:
        return False, "empty skill name"
    if has_local_skill_dir(name):
        return True, f"skill already available: {name}"
    if "/" in name or "\\" in name:
        return False, f"invalid skill name: {name!r}"
    if not SKILL_DYNAMIC_FETCH_ENABLED:
        return False, "dynamic skill fetch disabled"

    catalog_skills, by_name = _load_router_catalog_from_jsonl(SKILL_DYNAMIC_FETCH_CATALOG_JSONL)
    if not catalog_skills:
        return False, f"catalog unavailable: {SKILL_DYNAMIC_FETCH_CATALOG_JSONL}"
    preferred = _choose_catalog_entry(by_name.get(name) or [])
    if preferred is None:
        return False, f"skill {name!r} not found in catalog"

    github_url = str(preferred.get("githubUrl") or "").strip()
    parsed = _parse_github_tree_url(github_url)
    if parsed is None:
        return False, f"unsupported githubUrl: {github_url!r}"
    repo_url, ref, subpath = parsed
    allowed, normalized_candidates = _is_allowed_repo(repo_url)
    if not allowed:
        return (
            False,
            "blocked by SKILL_DYNAMIC_FETCH_ALLOWED_REPOS; "
            f"repo={repo_url} candidates={normalized_candidates}",
        )

    with _SKILL_FETCH_LOCK:
        if has_local_skill_dir(name):
            return True, f"skill already available: {name}"

        root = SKILL_DYNAMIC_FETCH_ROOT
        if not root.is_absolute():
            root = (PROJECT_ROOT / root).resolve()
        else:
            root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)

        dest = (root / name).resolve()
        if dest.exists():
            if (dest / "SKILL.md").exists():
                return True, f"skill already exists: {dest}"
            return False, f"target exists without SKILL.md: {dest}"

        with tempfile.TemporaryDirectory(prefix="memento-skill-") as tmp:
            repo_dir = Path(tmp) / "repo"
            ok, msg = _run_command_capture(
                ["git", "clone", "--filter=blob:none", "--no-checkout", repo_url, str(repo_dir)]
            )
            if not ok:
                return False, f"git clone failed: {msg}"

            if subpath:
                ok, msg = _run_command_capture(["git", "-C", str(repo_dir), "sparse-checkout", "init", "--cone"])
                if not ok:
                    return False, f"sparse-checkout init failed: {msg}"
                ok, msg = _run_command_capture(["git", "-C", str(repo_dir), "sparse-checkout", "set", subpath])
                if not ok:
                    return False, f"sparse-checkout set failed: {msg}"

            checkout_ref = ref if ref and ref != "HEAD" else "HEAD"
            ok, msg = _run_command_capture(["git", "-C", str(repo_dir), "checkout", checkout_ref])
            if not ok:
                return False, f"git checkout failed: {msg}"

            source_dir = _pick_skill_dir_from_checkout(repo_dir, subpath, name)
            if source_dir is None:
                return False, f"SKILL.md not found after checkout: repo={repo_url} subpath={subpath!r}"

            staging = root / f".{name}.tmp-{os.getpid()}-{int(time.time() * 1000)}"
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            shutil.copytree(source_dir, staging)
            skill_md = staging / "SKILL.md"
            if not skill_md.exists():
                shutil.rmtree(staging, ignore_errors=True)
                return False, f"cloned folder missing SKILL.md: {source_dir}"
            staging.rename(dest)

    if has_local_skill_dir(name):
        log_event(
            "skill_dynamic_fetch_success",
            skill=name,
            github_url=github_url,
            dest=str(dest),
        )
        return True, f"downloaded skill {name} -> {dest}"
    return False, f"download finished but skill is still unavailable: {name}"


def openskills_read(skill_name: str) -> str:
    """Read the SKILL.md for a skill.

    Prefers local SKILL.md if present to avoid stale `.agent/skills` copies
    and reduce dependency on `npx`/network during development. Falls back
    to `npx openskills read <skill_name>` if no local copy exists.

    Args:
        skill_name: Name of the skill whose SKILL.md to read.

    Returns:
        The contents of SKILL.md as a string.

    Raises:
        RuntimeError: If the skill cannot be read locally or via npx.
    """
    local_dir = _resolve_skill_dir(skill_name)
    if local_dir is not None:
        local_skill_md = local_dir / "SKILL.md"
        if local_skill_md.exists():
            return local_skill_md.read_text(encoding="utf-8")

    p = subprocess.run(
        ["npx", "openskills", "read", skill_name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        stdin=subprocess.DEVNULL,
        env=_NO_GIT_PROMPT_ENV,
    )
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "openskills read failed").strip())
    return p.stdout


def install_or_update_skill(skill_name: str) -> tuple[bool, str]:
    """Install or update one local skill with openskills and sync AGENTS.md.

    If the skill is already registered in `.agent/skills/<name>`, runs
    `npx openskills update`. Otherwise runs `npx openskills install`.
    After install/update, runs `npx openskills sync -y` to refresh AGENTS.md.

    Args:
        skill_name: Name of the skill to install or update.

    Returns:
        Tuple of (success, message).
    """
    name = str(skill_name or "").strip()
    if not name:
        return False, "empty skill name"
    skill_path = _resolve_skill_dir(name)
    if skill_path is None:
        return False, f"Skill folder not found for {name!r}"

    agent_skill_dir = PROJECT_ROOT / ".agent" / "skills" / name
    if agent_skill_dir.exists():
        cmd = ["npx", "openskills", "update", name]
    else:
        cmd = ["npx", "openskills", "install", str(skill_path), "--universal", "--yes"]

    ok, out = _run_command_capture(cmd, cwd=PROJECT_ROOT, timeout=max(120, SKILL_DYNAMIC_FETCH_TIMEOUT_SEC))
    if not ok:
        return False, out

    ok, sync_out = _run_command_capture(
        ["npx", "openskills", "sync", "-y"],
        cwd=PROJECT_ROOT,
        timeout=max(120, SKILL_DYNAMIC_FETCH_TIMEOUT_SEC),
    )
    if not ok:
        return False, sync_out
    return True, sync_out or "ok"
