"""
钉钉 Stream 模式消息接收器。

通过 dingtalk-stream SDK 建立持久 Stream 连接，接收钉钉机器人推送的实时消息。
这是让机器人**被动响应**用户消息的入口，与 dingtalk.py 的主动调用 API 互补。

使用方式：

  方式一：同步阻塞（最简单，适合独立进程）
    from scripts.dingtalk_receiver import DingTalkReceiver

    def on_message(msg: dict):
        print(f"收到消息：{msg['sender_id']}: {msg['content']}")

    receiver = DingTalkReceiver(on_message=on_message)
    receiver.start()   # 阻塞运行，Ctrl+C 退出

  方式二：后台线程（适合集成到现有服务）
    receiver = DingTalkReceiver(on_message=my_handler)
    t = receiver.start_in_background()
    # 主线程继续运行其他逻辑

配置方式（~/memento_s/config.json）：
  im.dingtalk.app_key = "dingxxxxxx"
  im.dingtalk.app_secret = "xxx"

依赖：
  pip install dingtalk-stream
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import threading
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path.home() / "memento_s" / "config.json"


def _load_dingtalk_config() -> dict:
    """从 ~/memento_s/config.json 加载钉钉配置，失败时返回空字典。"""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("im", {}).get("dingtalk", {})
    except Exception:
        return {}


class DingTalkReceiver:
    """
    钉钉 Stream 模式消息接收器。

    建立 Stream 连接，将收到的消息统一转换为 dict 后调用 on_message 回调。

    Args:
        on_message: 消息回调函数，接收一个 dict 参数（与 IMMessage 字段一致）。
                    可以是普通函数或 async 函数。
        app_key: 覆盖 DINGTALK_APP_KEY 环境变量
        app_secret: 覆盖 DINGTALK_APP_SECRET 环境变量
    """

    def __init__(
        self,
        on_message: Callable[[dict], None],
        app_key: str | None = None,
        app_secret: str | None = None,
    ) -> None:
        cfg = _load_dingtalk_config()
        self._app_key = app_key or cfg.get("app_key") or os.environ.get("DINGTALK_APP_KEY", "")
        self._app_secret = app_secret or cfg.get("app_secret") or os.environ.get("DINGTALK_APP_SECRET", "")
        self._on_message = on_message
        self._client = None
        self._stopped = False

        if not self._app_key or not self._app_secret:
            raise ValueError(
                f"钉钉 Stream 长链接需要 app_key 和 app_secret，"
                f"请在 {_CONFIG_PATH} 的 im.dingtalk 节填写。"
            )

    def _build_client(self):
        """构建 dingtalk-stream 客户端。"""
        try:
            import dingtalk_stream
        except ImportError:
            raise ImportError(
                "Stream 模式需要 dingtalk-stream，请先安装：pip install dingtalk-stream"
            )

        on_message = self._on_message

        class _Handler(dingtalk_stream.ChatbotHandler):
            async def process(self, callback: dingtalk_stream.CallbackMessage):
                try:
                    incoming = dingtalk_stream.ChatbotMessage.from_dict(callback.data)

                    # 构建统一消息格式（与飞书 receiver.py 保持一致）
                    content = ""
                    if incoming.text:
                        # 去掉 @机器人 的前缀（钉钉 SDK 已做基础处理，text.content 通常是干净内容）
                        content = (incoming.text.content or "").strip()

                    # conversation_type: "1" = 单聊, "2" = 群聊
                    chat_type = "p2p" if incoming.conversation_type == "1" else "group"

                    msg_dict = {
                        "id": incoming.message_id or "",
                        "chat_id": incoming.conversation_id or "",
                        "sender_id": incoming.sender_staff_id or incoming.sender_id or "",
                        "sender_name": incoming.sender_nick or "",
                        "content": content,
                        "msg_type": "text",
                        "create_time": str(incoming.create_at or ""),
                        "root_id": "",
                        "parent_id": "",
                        "chat_type": chat_type,
                        "mentions": [],
                        "platform": "dingtalk",
                    }

                    # 支持同步和异步回调
                    if inspect.iscoroutinefunction(on_message):
                        await on_message(msg_dict)
                    else:
                        on_message(msg_dict)

                except Exception as e:
                    logger.error(f"处理钉钉消息时出错：{e}", exc_info=True)

                return dingtalk_stream.AckMessage.STATUS_OK, "OK"

        credential = dingtalk_stream.Credential(self._app_key, self._app_secret)
        client = dingtalk_stream.DingTalkStreamClient(credential)
        client.register_callback_handler(dingtalk_stream.ChatbotMessage.TOPIC, _Handler())
        return client

    def start(self) -> None:
        """
        启动 Stream 长链接（阻塞）。

        Ctrl+C 退出。

        示例：
            receiver = DingTalkReceiver(on_message=my_handler)
            receiver.start()
        """
        logger.info("正在建立钉钉 Stream 长链接...")
        self._client = self._build_client()
        self._client.start_forever()

    def start_in_background(self) -> threading.Thread:
        """
        在后台线程中启动 Stream 长链接，立即返回。

        Returns:
            后台线程对象（daemon=True，主进程退出时自动结束）

        示例：
            receiver = DingTalkReceiver(on_message=my_handler)
            t = receiver.start_in_background()
            # 主线程继续运行其他逻辑
        """
        self._stopped = False
        self._thread_stopped = threading.Event()

        def _run() -> None:
            try:
                self._client = self._build_client()
                self._client.start_forever()
            except Exception as e:
                if not self._stopped:
                    logger.error(f"钉钉 Stream 长链接异常退出：{e}", exc_info=True)
            finally:
                self._thread_stopped.set()

        self._bg_thread = threading.Thread(target=_run, daemon=True, name="dingtalk-stream")
        self._bg_thread.start()
        logger.info("钉钉 Stream 长链接已在后台启动")
        return self._bg_thread

    def stop(self) -> None:
        """停止 Stream 长链接。"""
        self._stopped = True
        if self._client is not None:
            try:
                self._client.stop()
            except Exception as e:
                logger.warning(f"停止钉钉 Stream 长链接时出错：{e}")
            finally:
                self._client = None
        if hasattr(self, "_bg_thread"):
            self._bg_thread.join(timeout=5)
        logger.info("钉钉 Stream 长链接已停止")


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def start_dingtalk_receiver(
    on_message: Callable[[dict], None],
    background: bool = False,
) -> None | threading.Thread:
    """
    快速启动钉钉 Stream 消息接收器。

    Args:
        on_message: 消息回调（同步或 async 函数均可）
        background: True 时在后台线程运行（非阻塞），False 时阻塞运行

    Returns:
        background=True 时返回线程对象，否则返回 None
    """
    receiver = DingTalkReceiver(on_message=on_message)
    if background:
        return receiver.start_in_background()
    receiver.start()
    return None
