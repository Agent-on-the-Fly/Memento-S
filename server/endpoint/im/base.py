# SPDX-License-Identifier: Apache-2.0
"""
server/endpoint/im/base.py
IM 接入端点的协议定义和基类。
"""

from typing import Protocol, Any
from abc import ABC


class IMEndpoint(Protocol):
    """IM 接入端点协议"""

    @property
    def account_id(self) -> str: ...

    @property
    def channel_type(self) -> str: ...

    @property
    def is_running(self) -> bool: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def send_message(self, chat_id: str, content: str, **kwargs: Any) -> str: ...

    def get_status(self) -> dict[str, Any]: ...


class BaseEndpoint(ABC):
    """IM 接入端点基类"""

    def __init__(self, account_id: str, channel_type: str):
        self._account_id = account_id
        self._channel_type = channel_type
        self._running = False

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def channel_type(self) -> str:
        return self._channel_type

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict[str, Any]:
        return {
            "account_id": self._account_id,
            "channel_type": self._channel_type,
            "running": self._running,
        }
