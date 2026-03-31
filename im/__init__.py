"""IM Bridge - 即时通讯平台 Agent 桥接模块

顶层 IM 调用入口，与 gui/、cli/ 平级。
封装 MementoSAgent 调用逻辑，提供各 IM 平台的桥接实现。

结构:
  im/
  ├── bridge_base.py     # Agent 调用基类
  ├── session/           # Session 映射管理
  ├── feishu/            # 飞书桥接
  ├── dingtalk/          # 钉钉桥接
  ├── wecom/             # 企业微信桥接
  └── gateway/           # Gateway 模式桥接

调用关系:
  gui/cli → im/ → middleware/im/ → core/memento_s/
"""

from .bridge_base import IMBridgeBase

# 平台桥接
from im.feishu import FeishuBridge, feishu_bridge_command
from im.dingtalk import DingtalkBridge, dingtalk_bridge_command
from im.wecom import WecomBridge, wecom_bridge_command
from im.gateway import GatewayAgentWorker, gateway_worker_command

__all__ = [
    # 基类
    "IMBridgeBase",
    # 飞书
    "FeishuBridge",
    "feishu_bridge_command",
    # 钉钉
    "DingtalkBridge",
    "dingtalk_bridge_command",
    # 企业微信
    "WecomBridge",
    "wecom_bridge_command",
    # Gateway
    "GatewayAgentWorker",
    "gateway_worker_command",
]
