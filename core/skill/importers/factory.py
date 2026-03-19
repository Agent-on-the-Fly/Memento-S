"""下载器工厂 —— 创建预配置的下载管理器。"""

from __future__ import annotations

from .config import DownloadConfig
from .github import GitHubSkillDownloader
from .manager import DownloadManager


def create_default_download_manager(
    config: DownloadConfig | None = None,
) -> DownloadManager:
    """创建默认的下载管理器，预注册 GitHub 下载器。

    Args:
        config: 下载配置，如果为 None 则使用默认配置

    Returns:
        配置好的 DownloadManager 实例

    Example:
        from core.skill.importers.factory import create_default_download_manager
        from core.skill.importers.config import DownloadConfig

        config = DownloadConfig(
            github_token="your-token",
            github_mirrors=["https://mirror.github.com/"],
            timeout=30
        )
        manager = create_default_download_manager(config)
        result = manager.download(url, target_dir, skill_name)
    """
    manager = DownloadManager()
    manager.register(GitHubSkillDownloader(config))
    return manager
