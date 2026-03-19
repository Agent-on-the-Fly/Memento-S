"""下载配置。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DownloadConfig:
    """下载配置。

    Attributes:
        github_mirrors: GitHub 镜像代理列表
        timeout: 下载超时时间（秒）
        env: 环境变量字典，包含 GITHUB_TOKEN 等
    """

    github_mirrors: list[str] = field(default_factory=list)
    timeout: int = 30
    env: dict[str, str] | None = None
