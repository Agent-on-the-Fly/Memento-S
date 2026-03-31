"""企业微信桥接模块

提供企业微信平台的 Agent 桥接实现。
"""

from .bridge import (
    WecomBridge,
    get_wecom_bridge,
    start_wecom_bridge,
    stop_wecom_bridge,
)
from .cli import wecom_bridge_command

__all__ = [
    "WecomBridge",
    "get_wecom_bridge",
    "start_wecom_bridge",
    "stop_wecom_bridge",
    "wecom_bridge_command",
]
