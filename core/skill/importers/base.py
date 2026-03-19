"""技能下载器抽象基类。

定义所有技能下载器必须实现的接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class SkillDownloader(ABC):
    """技能下载器抽象基类。

    所有技能下载器必须继承此类并实现抽象方法。
    """

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """检查是否能处理给定的 URL。

        Args:
            url: 技能仓库的远程地址

        Returns:
            如果能处理此 URL 返回 True
        """
        pass

    @abstractmethod
    def download(self, url: str, target_dir: Path, skill_name: str) -> Path | None:
        """下载 skill 到目标目录。

        Args:
            url: 技能仓库的远程地址
            target_dir: 本地目标目录
            skill_name: 技能名称

        Returns:
            下载后的本地路径，失败返回 None
        """
        pass
