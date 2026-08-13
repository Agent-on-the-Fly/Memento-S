# SPDX-License-Identifier: Apache-2.0
"""CLI commands for Memento-S."""

from .agent import agent_command
from .doctor import doctor_command
from .im_status import im_status_command
from .wechat import wechat_app
from .skills import skills_app

from server.endpoint.im.cli import (
    feishu_bridge_command,
    dingtalk_bridge_command,
    wecom_bridge_command,
    wechat_bridge_command,
    gateway_worker_command,
)

__all__ = [
    "agent_command",
    "doctor_command",
    "feishu_bridge_command",
    "dingtalk_bridge_command",
    "wecom_bridge_command",
    "wechat_bridge_command",
    "im_status_command",
    "gateway_worker_command",
    "wechat_app",
    "skills_app",
]
