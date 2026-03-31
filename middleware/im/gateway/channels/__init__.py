"""
渠道适配器模块。

提供渠道适配器基类和注册机制。
"""

from .base import BaseChannelAdapter

# 导入所有适配器以触发装饰器注册
from . import feishu
from . import dingtalk
from . import wecom
from . import wechat_ilinkai

__all__ = [
    "BaseChannelAdapter",
    "feishu",
    "dingtalk",
    "wecom",
    "wechat_ilinkai",
]
