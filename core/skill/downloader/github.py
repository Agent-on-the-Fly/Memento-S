"""GitHub skill 下载器实现。"""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests

from core.utils.text import to_kebab_case
from utils.logger import get_logger

from .base import SkillDownloader
from .config import DownloadConfig

logger = get_logger(__name__)


class GitHubSkillDownloader(SkillDownloader):
    """GitHub skill 下载器。

    支持:
    - GitHub Contents API 下载
    - 镜像代理 fallback
    - Token 鉴权
    """

    def __init__(self, config: DownloadConfig | None = None):
        self._config = config or DownloadConfig()

    def can_handle(self, url: str) -> bool:
        """检查是否为 GitHub URL。"""
        parsed = urlparse(url)
        return parsed.hostname is not None and "github.com" in parsed.hostname

    def download(self, url: str, target_dir: Path, skill_name: str) -> Path | None:
        """从 GitHub 下载 skill。

        约定：入参 skill_name 是内部名（snake_case），落盘目录统一为 kebab-case。
        """
        info = self._parse_github_tree_url(url)
        if not info:
            logger.warning("Cannot parse GitHub URL: {}", url)
            return None

        owner, repo, branch, path = (
            info["owner"],
            info["repo"],
            info["branch"],
            info["path"],
        )

        storage_name = to_kebab_case(skill_name)

        # GitHub URL 中的路径名（用于临时下载目录）
        url_skill_name = path.rstrip("/").split("/")[-1] if path else storage_name
        # 最终目标目录（统一使用 kebab-case 目录名）
        final_target_dir = target_dir / storage_name
        # 临时下载目录
        temp_target_dir = target_dir / url_skill_name

        logger.info(
            "Downloading skill '{}' from GitHub ({}/{})...",
            skill_name,
            owner,
            repo,
        )

        if self._download_github_dir(
            owner, repo, branch, path, temp_target_dir, timeout=self._config.timeout
        ):
            # 下载完成后，如果目录名不同，重命名为 skill_name
            if temp_target_dir != final_target_dir:
                if final_target_dir.exists():
                    # 如果目标目录已存在，先删除
                    shutil.rmtree(final_target_dir)
                temp_target_dir.rename(final_target_dir)
                logger.info(
                    "Renamed directory from '{}' to '{}'", url_skill_name, skill_name
                )

            if not (final_target_dir / "SKILL.md").exists():
                logger.warning("Downloaded '{}' but no SKILL.md found", skill_name)
            else:
                logger.info("Skill '{}' downloaded to {}", skill_name, final_target_dir)
            return final_target_dir

        logger.warning("No files downloaded for '{}'", skill_name)
        return None

    def _parse_github_tree_url(self, github_url: str) -> dict | None:
        """解析 GitHub tree URL → owner, repo, branch, path。

        格式: https://github.com/{owner}/{repo}/tree/{branch}/{path...}
        """
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

    def _github_headers(self) -> dict:
        """构建 GitHub API 请求头。"""
        headers = {"Accept": "application/vnd.github.v3+json"}
        # 优先使用 github_token 属性，兼容 env 字典
        github_token = self._config.github_token
        if not github_token and self._config.env:
            github_token = self._config.env.get("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"
        return headers

    def _get_mirror_prefixes(self) -> list[str]:
        """返回镜像前缀列表，末尾追加空串表示直连兜底。"""
        prefixes = [m.rstrip("/") + "/" for m in self._config.github_mirrors if m]
        prefixes.append("")  # 直连兜底
        return prefixes

    def _download_github_dir(
        self,
        owner: str,
        repo: str,
        branch: str,
        path: str,
        local_dir: Path,
        timeout: int,
        _mirror_prefix: str | None = None,
    ) -> bool:
        """递归下载 GitHub 目录。

        Returns:
            True if at least one file was downloaded
        """
        api_url = (
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
        )
        headers = self._github_headers()

        # 确定可用的镜像前缀
        prefixes = (
            [_mirror_prefix]
            if _mirror_prefix is not None
            else self._get_mirror_prefixes()
        )

        resp = None
        used_prefix = ""
        for prefix in prefixes:
            try:
                real_url = f"{prefix}{api_url}" if prefix else api_url
                logger.debug("Trying GitHub API: {}", real_url)
                resp = requests.get(real_url, headers=headers, timeout=timeout)
                resp.raise_for_status()
                used_prefix = prefix
                if prefix:
                    logger.info("Mirror hit: {}", prefix)
                break
            except requests.RequestException as e:
                label = prefix or "direct"
                logger.warning(
                    "GitHub API via {} failed for {}/{}/{}: {}",
                    label,
                    owner,
                    repo,
                    path,
                    e,
                )
                resp = None

        if resp is None:
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
                real_download_url = (
                    f"{used_prefix}{download_url}" if used_prefix else download_url
                )
                try:
                    file_resp = requests.get(
                        real_download_url, headers=headers, timeout=timeout
                    )
                    file_resp.raise_for_status()
                    file_path = local_dir / item["name"]
                    file_path.write_bytes(file_resp.content)
                    downloaded = True
                    logger.debug(
                        "Downloaded: {} ({}) bytes",
                        item["path"],
                        len(file_resp.content),
                    )
                except requests.RequestException as e:
                    logger.warning("Failed to download {}: {}", item["path"], e)
            elif item["type"] == "dir":
                sub_dir = local_dir / item["name"]
                if self._download_github_dir(
                    owner,
                    repo,
                    branch,
                    item["path"],
                    sub_dir,
                    timeout,
                    used_prefix,
                ):
                    downloaded = True

        return downloaded
