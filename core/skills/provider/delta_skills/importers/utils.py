
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from core.config import g_settings
from core.config.logging import get_logger
from core.llm import LLM
from .adapter import SkillAdapter
from .openskills_importer import OpenSkillsImporter
from ..schema import Skill

logger = get_logger(__name__)
_llm = LLM()

_CJK_TO_ENGLISH_PROMPT = """Translate the following user intent to a short English phrase for searching (3-8 words).
Output ONLY the English phrase, no quotes or explanation.

Examples:
   → feishu document summary
   → meeting notes
   → weather forecast
   → deep research

User query: {query}"""


def _parse_cjk_response(query: str, raw_text: str) -> Optional[str]:
    text = raw_text.strip().strip('"\'`')
    if not text:
        logger.warning("CJK→EN: empty response, keeping original query")
        return None
    first_line = text.split("\n")[0].strip().rstrip(".,;:")
    if ":" in first_line:
        after_colon = first_line.split(":", 1)[-1].strip().rstrip(".,;:")
        if after_colon and not re.search(r"[\u4e00-\u9fff]", after_colon) and len(after_colon) < 80:
            first_line = after_colon
    if first_line and not re.search(r"[\u4e00-\u9fff]", first_line) and len(first_line) < 80:
        logger.info("CJK→EN: '%s' → '%s'", query[:40], first_line)
        return first_line
    logger.warning("CJK→EN: response not valid English (first_line=%s), keeping original", first_line[:50])
    return None


async def cjk_query_to_english(query: str) -> str:
    if not re.search(r"[\u4e00-\u9fff]", query):
        return query
    try:
        resp = await _llm.chat(
            messages=[{"role": "user", "content": _CJK_TO_ENGLISH_PROMPT.format(query=query)}],
            temperature=0,
            max_tokens=1024,
            timeout=15,
        )
        raw = resp.content or ""

        if not raw and resp.raw and hasattr(resp.raw, "choices") and resp.raw.choices:
            msg = resp.raw.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if reasoning:
                logger.info("CJK→EN: content is empty, trying reasoning_content")
                raw = reasoning

        result = _parse_cjk_response(query, raw)
        if result:
            return result
    except Exception as e:
        logger.warning("CJK→EN failed, keeping original query: %s", e)
    return query


def parse_github_tree_url(github_url: str) -> Optional[dict]:
    parsed = urlparse(github_url)
    if not parsed.hostname or "github.com" not in parsed.hostname:
        return None
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 4 or parts[2] != "tree":
        return None
    return {
        "owner": parts[0],
        "repo": parts[1],
        "branch": parts[3],
        "path": "/".join(parts[4:]) if len(parts) > 4 else "",
    }


def _github_headers() -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = getattr(g_settings, "github_token", None)
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _download_github_dir(
    owner: str, repo: str, branch: str, path: str, local_dir: Path,
    timeout: int = 30,
) -> bool:
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    headers = _github_headers()
    try:
        resp = requests.get(api_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("GitHub API failed for %s/%s/%s: %s", owner, repo, path, e)
        return False

    items = resp.json()
    if not isinstance(items, list):
        items = [items]

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = False

    for item in items:
        if item["type"] == "file":
            download_url = item.get("download_url")
            if not download_url:
                continue
            try:
                file_resp = requests.get(download_url, headers=headers, timeout=timeout)
                file_resp.raise_for_status()
                file_path = local_dir / item["name"]
                file_path.write_bytes(file_resp.content)
                downloaded = True
                logger.debug("Downloaded: %s (%d bytes)", item["path"], len(file_resp.content))
            except requests.RequestException as e:
                logger.warning("Failed to download %s: %s", item["path"], e)
        elif item["type"] == "dir":
            sub_dir = local_dir / item["name"]
            if _download_github_dir(owner, repo, branch, item["path"], sub_dir, timeout):
                downloaded = True

    return downloaded


def download_skill_from_github(github_url: str, skills_dir: Path) -> Optional[Path]:
    info = parse_github_tree_url(github_url)
    if not info:
        logger.warning("Cannot parse GitHub URL: %s", github_url)
        return None

    owner, repo, branch, path = info["owner"], info["repo"], info["branch"], info["path"]
    skill_name = path.rstrip("/").split("/")[-1] if path else repo
    target_dir = skills_dir / skill_name

    logger.info("Downloading skill '%s' from GitHub (%s/%s)...", skill_name, owner, repo)

    if _download_github_dir(owner, repo, branch, path, target_dir):
        if (target_dir / "SKILL.md").exists():
            logger.info("Skill '%s' downloaded to %s", skill_name, target_dir)
            return target_dir
        else:
            logger.warning("Downloaded '%s' but no SKILL.md found", skill_name)
            return target_dir
    else:
        logger.warning("No files downloaded for '%s'", skill_name)
        return None


def download_skill_batch_from_github(github_urls: list[str], skills_dir: Path = None) -> list[Path]:
    if skills_dir is None:
        skills_dir = g_settings.skills_directory

    results = []
    for url in github_urls:
        path = download_skill_from_github(url, skills_dir)
        if path:
            results.append(path)
        else:
            logger.error("Batch download failed for %s", url)
    return results


def _download_via_openskills(github_url: str, skills_dir: Path) -> Optional[Path]:
    if not shutil.which("npx"):
        logger.debug("npx not available, skipping openskills download")
        return None

    try:
        importer = OpenSkillsImporter()
        imported_paths = importer.import_skill(github_url, target_dir=skills_dir)
        if imported_paths:
            local_path = imported_paths[0]
            logger.info("npx openskills download succeeded: %s", local_path)
            return local_path
    except Exception as e:
        logger.warning("npx openskills download failed: %s", e)

    return None


def download_with_strategy(github_url: str, skills_dir: Path, skill_name: str) -> Optional[Path]:
    method = g_settings.skill_download_method

    if method == "npx":
        logger.info("Download method=npx: using openskills for '%s'", skill_name)
        return _download_via_openskills(github_url, skills_dir)

    if method == "github_api":
        logger.info("Download method=github_api: using GitHub Contents API for '%s'", skill_name)
        return download_skill_from_github(github_url, skills_dir)

    local_path = download_skill_from_github(github_url, skills_dir)
    if local_path:
        return local_path

    logger.info("GitHub API failed for '%s', auto-fallback to npx openskills...", skill_name)
    return _download_via_openskills(github_url, skills_dir)
