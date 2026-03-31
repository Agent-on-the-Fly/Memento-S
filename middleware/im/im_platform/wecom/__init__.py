"""企业微信（WeCom）平台包。"""
from .platform import WecomPlatform
from .receiver import WecomReceiver, start_wecom_receiver

__all__ = ["WecomPlatform", "WecomReceiver", "start_wecom_receiver"]
