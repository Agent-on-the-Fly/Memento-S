"""Notification Manager - 跨模块通知系统

用于从后台线程（如 Gateway）向 GUI 发送通知。
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Any
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """通知类型"""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Notification:
    """通知数据"""

    message: str
    notif_type: NotificationType = NotificationType.INFO
    duration: int = 3000  # 毫秒


class NotificationManager:
    """通知管理器 - 单例模式

    使用示例:
        # 在 GUI 初始化时设置回调
        from gui.utils.notification_manager import notification_manager
        notification_manager.set_callback(lambda msg, type: show_snackbar(msg))

        # 在任意地方发送通知（包括后台线程）
        notification_manager.notify("飞书已启动", NotificationType.SUCCESS)
    """

    _instance: NotificationManager | None = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._callback: Callable[[str, NotificationType], None] | None = None
        self._queue: Queue[Notification] = Queue()
        self._initialized = True

    def set_callback(self, callback: Callable[[str, NotificationType], None]) -> None:
        """设置通知回调函数（由 GUI 调用）

        Args:
            callback: 回调函数，接收 (message, notification_type)
        """
        self._callback = callback
        logger.info("[NotificationManager] Callback registered")

    def notify(
        self, message: str, notif_type: NotificationType = NotificationType.INFO
    ) -> None:
        """发送通知（线程安全）

        Args:
            message: 通知消息
            notif_type: 通知类型
        """
        notification = Notification(message=message, notif_type=notif_type)

        # 直接调用回调（如果可用）
        if self._callback:
            try:
                self._callback(message, notif_type)
                logger.debug(f"[NotificationManager] Notification sent: {message}")
            except Exception as e:
                logger.error(f"[NotificationManager] Failed to send notification: {e}")
        else:
            # 回调未设置，存入队列稍后处理
            self._queue.put(notification)
            logger.debug(f"[NotificationManager] Notification queued: {message}")

    def process_queue(self) -> None:
        """处理队列中的通知（由 GUI 主线程定期调用）"""
        if not self._callback:
            return

        while not self._queue.empty():
            try:
                notif = self._queue.get_nowait()
                self._callback(notif.message, notif.notif_type)
            except Empty:
                break
            except Exception as e:
                logger.error(
                    f"[NotificationManager] Error processing notification: {e}"
                )


# 全局单例实例
notification_manager = NotificationManager()


# 便捷函数
def notify_info(message: str) -> None:
    """发送信息通知"""
    notification_manager.notify(message, NotificationType.INFO)


def notify_success(message: str) -> None:
    """发送成功通知"""
    notification_manager.notify(message, NotificationType.SUCCESS)


def notify_warning(message: str) -> None:
    """发送警告通知"""
    notification_manager.notify(message, NotificationType.WARNING)


def notify_error(message: str) -> None:
    """发送错误通知"""
    notification_manager.notify(message, NotificationType.ERROR)
