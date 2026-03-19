"""importers — 技能下载

此包提供技能下载功能。

主要组件:
- SkillDownloader: 技能下载器抽象基类
- GitHubSkillDownloader: GitHub 技能下载器实现
- DownloadConfig: 下载配置
- DownloadManager: 下载管理器
- create_default_download_manager: 工厂函数

示例:
    from core.skill.importers import (
        GitHubSkillDownloader,
        DownloadConfig,
        DownloadManager,
        create_default_download_manager,
    )

    # 创建下载器
    config = DownloadConfig(github_token="your-token")
    downloader = GitHubSkillDownloader(config)

    # 或使用管理器
    manager = create_default_download_manager(config)
    result = manager.download(url, target_dir, skill_name)
"""

from .base import SkillDownloader
from .config import DownloadConfig
from .factory import create_default_download_manager
from .github import GitHubSkillDownloader
from .manager import DownloadManager

__all__ = [
    "SkillDownloader",
    "GitHubSkillDownloader",
    "DownloadConfig",
    "DownloadManager",
    "create_default_download_manager",
]
