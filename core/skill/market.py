"""Skill market service - search/install/uninstall skills via remote catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import httpx
import shutil
import tempfile

from core.skill.config import SkillConfig
from core.skill.downloader.factory import create_default_download_manager
from core.skill.downloader.github import GitHubSkillDownloader
from core.skill.retrieval.remote_recall import RemoteRecall
from core.skill.schema import Skill, SkillManifest, SkillGovernanceMeta, ExecutionMode
from core.skill.store import SkillStore
from core.utils.text import to_kebab_case, to_snake_case
from utils.logger import get_logger

logger = get_logger(__name__)

# Default skill parameters (copied from schema.py for standalone use)
DEFAULT_SKILL_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
}


class SkillMarket:
    def __init__(
        self,
        *,
        config: "SkillConfig",
        store: "SkillStore",
        remote_recall: Optional["RemoteRecall"] = None,
    ) -> None:
        self._config = config
        self._store = store
        self._remote_recall = remote_recall

    @classmethod
    async def from_config(cls, config: Optional["SkillConfig"] = None) -> "SkillMarket":
        """
        工厂方法：从配置创建（用于生产环境）

        Args:
            config: SkillConfig 配置

        Returns:
            SkillMarket 实例
        """
        if config is None:
            config = SkillConfig.from_global_config()

        # SkillStore.from_config 内部自动处理 embedding
        store = await SkillStore.from_config(config)

        remote_recall = RemoteRecall.from_config(config)

        return cls(
            config=config,
            store=store,
            remote_recall=remote_recall,
        )

    async def search(self, query: str, k: int = 5):
        """Search skills from remote catalog."""
        logger.info("[SkillMarket] Searching for query: '{}', k={}", query, k)

        if not self._remote_recall:
            logger.warning("[SkillMarket] Remote recall not available, cannot search")
            return []

        try:
            logger.debug("[SkillMarket] Calling remote search API...")
            results = await self._remote_recall.search(query, k=k)
            logger.info(
                "[SkillMarket] Search completed: found {} skills for query '{}'",
                len(results),
                query,
            )

            # Log each found skill
            for i, skill in enumerate(results):
                skill_name = (
                    skill.get("name", "unknown")
                    if isinstance(skill, dict)
                    else getattr(skill, "name", "unknown")
                )
                logger.debug("[SkillMarket] Result {}: {}", i + 1, skill_name)

            return results
        except Exception as e:
            logger.error("[SkillMarket] Search failed for '{}': {}", query, e)
            return []

    async def install(self, skill_name: str) -> Optional["Skill"]:
        """Download a cloud skill and persist it locally.

        内部传递统一使用 snake_case；落盘目录由 downloader 统一转 kebab-case。
        """
        if not self._remote_recall:
            return None

        internal_name = to_snake_case(skill_name)
        github_url = await self._get_cloud_skill_url(to_kebab_case(internal_name))
        if not github_url:
            return None

        try:
            download_manager = create_default_download_manager()
            local_path = download_manager.download(
                github_url,
                self._config.skills_dir,
                internal_name,
            )
            if not local_path:
                return None

            skill = await self._store.load_from_path(Path(local_path))
            return skill
        except Exception as e:
            logger.warning("Failed to install cloud skill '{}': {}", skill_name, e)
            return None

    async def install_from_local(self, local_path: str) -> Optional["Skill"]:
        """Install a local skill and persist it locally.

        流程：
        1. 验证源路径的 skill 是否有效
        2. 复制到 skills_dir
        3. 加载并同步到 DB + Vector
        """
        source_path = Path(local_path).resolve()

        try:
            # 1. 验证 skill 有效性（尝试加载但不保存）
            skill = self._store._loader.load_from_dir(source_path, full=True)
            if not skill:
                logger.warning("Invalid skill at '{}'", local_path)
                return None

            # 2. 复制到 skills_dir
            internal_name = to_snake_case(skill.name or source_path.name)
            storage_name = to_kebab_case(internal_name)
            target_path = Path(self._config.skills_dir) / storage_name

            if target_path.exists():
                logger.warning(
                    "Skill '{}' already exists, removing old version", internal_name
                )
                shutil.rmtree(target_path)

            shutil.copytree(source_path, target_path)
            logger.info(
                "Copied skill '{}' from {} to {}",
                internal_name,
                source_path,
                target_path,
            )

            # 3. 从目标路径加载并同步到 DB + Vector
            installed_skill = await self._store.load_from_path(target_path)
            return installed_skill

        except Exception as e:
            logger.warning("Failed to install local skill '{}': {}", local_path, e)
            return None

    async def install_from_url(self, url: str) -> Optional["Skill"]:
        """Install a skill from remote URL by downloading to tmp first.

        Supports:
        - GitHub tree URLs (e.g., https://github.com/owner/repo/tree/branch/path)
        - Archive files (zip, tar.gz, etc.)
        """
        if not url.startswith("http"):
            logger.warning("Invalid URL '{}': only http/https is supported", url)
            return None

        temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None

        try:
            temp_dir_obj = tempfile.TemporaryDirectory(dir=tempfile.gettempdir())
            temp_dir = Path(temp_dir_obj.name)

            # Try GitHub download first
            github_downloader = GitHubSkillDownloader()
            if github_downloader.can_handle(url):
                source_path = self._download_from_github(
                    url, temp_dir, github_downloader
                )
            else:
                source_path = self._download_archive(url, temp_dir)

            if source_path is None:
                return None

            return await self.install_from_local(str(source_path))

        except Exception as e:
            logger.warning("Failed to install skill from url '{}': {}", url, e)
            return None
        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()

    def _download_from_github(
        self, url: str, temp_dir: Path, downloader: GitHubSkillDownloader
    ) -> Path | None:
        """Download skill from GitHub tree URL."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        skill_name = path_parts[-1] if path_parts else "unknown"

        logger.info("Downloading skill '{}' from GitHub URL", skill_name)
        return downloader.download(url, temp_dir, skill_name)

    def _download_archive(self, url: str, temp_dir: Path) -> Path | None:
        """Download and extract skill from archive URL."""
        temp_download_path = temp_dir / "skill_package"
        logger.info("Downloading archive from {} to {}", url, temp_download_path)

        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            resp = client.get(url)
            resp.raise_for_status()

        temp_download_path.write_bytes(resp.content)

        extracted_dir = temp_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(temp_download_path), str(extracted_dir))

        if (extracted_dir / "SKILL.md").exists():
            return extracted_dir

        candidate_dirs = [
            p
            for p in extracted_dir.iterdir()
            if p.is_dir() and (p / "SKILL.md").exists()
        ]
        if len(candidate_dirs) == 1:
            return candidate_dirs[0]

        logger.error("Archive from '{}' does not contain a valid skill directory", url)
        return None

    async def uninstall(self, skill_name: str) -> bool:
        """Remove a skill from local disk, DB, and vector index."""
        try:
            return await self._store.remove_skill(skill_name)
        except Exception as e:
            logger.warning("Failed to uninstall skill '{}': {}", skill_name, e)
            return False

    async def list_skills(self) -> list[SkillManifest]:
        """List all local skills from the store.

        Returns:
            List of SkillManifest objects representing all local skills.
            Returns empty list if no skills found or on error.
        """
        try:
            skills = await self._store.list_all_skills()
            manifests = [
                self._to_manifest(skill, source="local") for skill in skills.values()
            ]
            manifests.sort(key=lambda m: m.name)
            logger.info("Listed {} local skills", len(manifests))
            return manifests
        except Exception as e:
            logger.warning("Failed to list local skills: {}", e)
            return []

    def _to_manifest(self, skill: Skill, source: str = "local") -> SkillManifest:
        """Convert a Skill object to SkillManifest format.

        Args:
            skill: The Skill object to convert
            source: Source identifier ("local" or "cloud")

        Returns:
            SkillManifest representation of the skill
        """
        exec_mode = skill.execution_mode or (
            ExecutionMode.PLAYBOOK if skill.is_playbook else ExecutionMode.KNOWLEDGE
        )

        return SkillManifest(
            name=skill.name,
            description=skill.description or "",
            parameters=skill.parameters or DEFAULT_SKILL_PARAMS,
            execution_mode=exec_mode,
            dependencies=skill.dependencies or [],
            governance=SkillGovernanceMeta(
                source="cloud" if source == "cloud" else "local",
            ),
        )

    async def _get_cloud_skill_url(self, skill_name: str) -> Optional[str]:
        if not self._remote_recall:
            return None

        try:
            base_url = self._remote_recall._base_url
            with httpx.Client() as client:
                print(
                    f"Requesting download URL for skill '{skill_name}' from {base_url}..."
                )
                resp = client.post(
                    f"{base_url}/api/v1/download",
                    json={"skill_name": skill_name},
                )
                print(f"Received response: status={resp.status_code}, body={resp.text}")
                if resp.status_code == 200:
                    return resp.json().get("github_url", "")
        except Exception as e:
            logger.warning("Failed to get cloud skill URL for '{}': {}", skill_name, e)

        return None
