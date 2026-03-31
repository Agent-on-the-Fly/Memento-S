"""Event Bus - 简单的事件发布/订阅系统

用于跨层通信，保持架构清晰：
- middleware 层：发布事件，不关心谁来处理
- GUI 层：订阅事件，处理UI显示
- 其他层：可以发布或订阅

使用示例：
    # 在 GUI 层订阅事件
    from utils.event_bus import event_bus, EventType

    def on_im_started(data):
        show_snackbar(f"{data['platform']} 服务已启动")

    event_bus.subscribe(EventType.IM_SERVICE_STARTED, on_im_started)

    # 在 middleware 层发布事件
    event_bus.publish(EventType.IM_SERVICE_STARTED, {"platform": "飞书"})
"""

from __future__ import annotations

import logging
import threading
from enum import Enum, auto
from typing import Any, Callable, Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型定义"""

    # IM 服务事件
    IM_SERVICE_STARTED = auto()
    IM_SERVICE_STOPPED = auto()
    IM_SERVICE_START_FAILED = auto()
    IM_SERVICE_STOP_FAILED = auto()

    # Gateway 事件
    GATEWAY_STARTED = auto()
    GATEWAY_STOPPED = auto()

    # 认证事件
    AUTH_REQUIRED = auto()  # HTTP 401 触发，需要重新登录

    # 通用事件
    CONFIG_CHANGED = auto()
    ERROR_OCCURRED = auto()


@dataclass
class Event:
    """事件数据"""

    type: EventType
    data: Any = None
    source: str = ""


class EventBus:
    """事件总线 - 发布/订阅模式"""

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()

    def subscribe(
        self, event_type: EventType, callback: Callable[[Event], None]
    ) -> None:
        """订阅事件

        Args:
            event_type: 事件类型
            callback: 回调函数，接收 Event 对象
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

        logger.debug(f"[EventBus] Subscribed to {event_type.name}")

    def unsubscribe(
        self, event_type: EventType, callback: Callable[[Event], None]
    ) -> None:
        """取消订阅"""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    cb for cb in self._subscribers[event_type] if cb != callback
                ]

    def publish(
        self, event_type: EventType, data: Any = None, source: str = ""
    ) -> None:
        """发布事件

        Args:
            event_type: 事件类型
            data: 事件数据（任意类型）
            source: 事件来源标识
        """
        event = Event(type=event_type, data=data, source=source)

        with self._lock:
            callbacks = self._subscribers.get(event_type, []).copy()

        if not callbacks:
            logger.debug(f"[EventBus] No subscribers for {event_type.name}")
            return

        # 调用所有订阅者
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"[EventBus] Error in callback for {event_type.name}: {e}")

        logger.debug(
            f"[EventBus] Published {event_type.name} to {len(callbacks)} subscribers"
        )


# 全局事件总线实例
event_bus = EventBus()


# 便捷函数
def subscribe(event_type: EventType, callback: Callable[[Event], None]) -> None:
    """订阅事件（便捷函数）"""
    event_bus.subscribe(event_type, callback)


def publish(event_type: EventType, data: Any = None, source: str = "") -> None:
    """发布事件（便捷函数）"""
    event_bus.publish(event_type, data, source)
