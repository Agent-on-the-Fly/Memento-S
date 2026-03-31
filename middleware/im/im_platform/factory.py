"""
IM 平台工厂。

通过 get_platform() 获取当前配置的平台实例，无需手动 import 适配器。

扩展新平台步骤：
  1. 在 scripts/ 下新建适配器（如 slack.py），实现 IMPlatform 协议
  2. 在 _REGISTRY 中注册平台名称与类
  3. 在 SKILL.md 中说明新平台的环境变量要求

示例用法：
    from factory import get_platform

    platform = get_platform()                       # 从 IM_PLATFORM 环境变量自动检测
    platform = get_platform("feishu")               # 显式指定平台
    msg = await platform.send_message("xxx", "你好")
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .base import IMError, IMPlatform

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# 平台注册表
# ---------------------------------------------------------------------------


def _get_registry() -> dict[str, type]:
    """延迟导入各适配器，避免未安装依赖时导入失败。"""
    registry: dict[str, type] = {}

    try:
        from .feishu.platform import FeishuPlatform
        registry["feishu"] = FeishuPlatform
        registry["lark"] = FeishuPlatform
    except ImportError:
        pass

    try:
        from .dingtalk.platform import DingTalkPlatform
        registry["dingtalk"] = DingTalkPlatform
    except ImportError:
        pass

    try:
        from .wecom.platform import WecomPlatform
        registry["wecom"] = WecomPlatform
        registry["wxwork"] = WecomPlatform  # 别名
    except ImportError:
        pass

    # 未来平台预留（取消注释并新建对应适配器文件即可接入）
    # try:
    #     from slack import SlackPlatform
    #     registry["slack"] = SlackPlatform
    # except ImportError:
    #     pass

    return registry


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def get_platform(platform: str | None = None) -> IMPlatform:
    """
    获取 IM 平台实例。

    Args:
        platform: 平台名称（feishu | lark）。
                  为 None 时从 IM_PLATFORM 环境变量读取，默认 "feishu"。

    Returns:
        实现了 IMPlatform 协议的平台实例。

    Raises:
        IMError: 平台名称不支持或环境变量未配置时抛出。

    示例：
        platform = get_platform()
        msg = await platform.send_message("ou_xxx", "Hello")
    """
    name = (
        platform or os.environ.get("IM_PLATFORM", "feishu")
    ).lower().strip()
    registry = _get_registry()

    if name not in registry:
        supported = ", ".join(sorted(registry.keys())) or "（无可用平台，请检查依赖）"
        raise IMError(
            f"不支持的 IM 平台：{name!r}。当前支持：{supported}。"
            f"请检查 IM_PLATFORM 环境变量是否正确。",
        )

    return registry[name]()


def list_supported_platforms() -> list[str]:
    """列出当前可用的平台名称（需对应环境变量已配置）。"""
    return sorted(_get_registry().keys())
