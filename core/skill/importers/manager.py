"""下载管理器 —— 管理多个下载器，自动选择合适的下载器。"""

from __future__ import annotations

from pathlib import Path

from utils.logger import get_logger

from .base import SkillDownloader

logger = get_logger(__name__)


class DownloadManager:
    """下载管理器 —— 管理多个下载器，自动选择合适的下载器。

    支持注册多个下载器，按注册顺序尝试，直到成功或全部失败。

    Example:
        manager = DownloadManager()
        manager.register(GitHubSkillDownloader(config))
        # 未来可以注册更多下载器
        # manager.register(GitLabSkillDownloader(config))
        # manager.register(GiteeSkillDownloader(config))

        result = manager.download(url, target_dir, skill_name)
    """

    def __init__(self, downloaders: list[SkillDownloader] | None = None):
        self._downloaders = downloaders or []

    def register(self, downloader: SkillDownloader) -> None:
        """注册新的下载器。

        Args:
            downloader: 要注册的下载器实例
        """
        self._downloaders.append(downloader)

    def download(self, url: str, target_dir: Path, skill_name: str) -> Path | None:
        """使用合适的下载器下载 skill。

        按注册顺序尝试每个下载器，直到成功或全部失败。

        Args:
            url: skill 仓库的远程地址
            target_dir: 本地目标目录
            skill_name: skill 名称

        Returns:
            下载后的本地路径，失败返回 None
        """
        for downloader in self._downloaders:
            if downloader.can_handle(url):
                try:
                    result = downloader.download(url, target_dir, skill_name)
                    if result:
                        return result
                except Exception as e:
                    logger.warning("Downloader failed for '{}': {}", skill_name, e)

        logger.error("No downloader can handle URL: {}", url)
        return None

    @property
    def registered_downloaders(self) -> list[SkillDownloader]:
        """获取已注册的下载器列表。"""
        return self._downloaders.copy()
