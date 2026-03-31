"""
飞书长链接（WebSocket）消息接收器。

通过 lark-oapi SDK 建立持久 WebSocket 连接，接收飞书推送的实时事件（消息、@机器人等）。
这是让机器人**被动响应**用户消息的入口，与 feishu.py 的主动调用 API 互补。

使用方式：

  方式一：同步阻塞（最简单，适合独立进程）
    from scripts.receiver import FeishuReceiver

    def on_message(msg: dict):
        print(f"收到消息：{msg['sender_id']}: {msg['content']}")

    receiver = FeishuReceiver(on_message=on_message)
    receiver.start()   # 阻塞运行，Ctrl+C 退出

  方式二：在已有 asyncio 事件循环中运行
    import asyncio
    from scripts.receiver import FeishuReceiver

    async def on_message(msg: dict):
        # 可以在这里调用 messaging.py 的函数进行回复
        from scripts.messaging import reply_to_message
        await reply_to_message(msg["id"], "收到，正在处理...")

    receiver = FeishuReceiver(on_message=on_message)
    await receiver.start_async()   # 非阻塞，在后台运行

配置方式（~/memento_s/config.json）：
  im.feishu.app_id = "cli_xxx"
  im.feishu.app_secret = "xxx"
  im.feishu.encrypt_key = "xxx"          # 可选，开放平台配置了加密时填写
  im.feishu.verification_token = "xxx"   # 可选
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import threading
from pathlib import Path
from typing import Callable

# 配置 Lark SDK 日志级别
# 抑制正常的 WebSocket 关闭日志（ConnectionClosedOK 是正常关闭，code=1000）
logging.getLogger("lark").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path.home() / "memento_s" / "config.json"


def _load_feishu_config() -> dict:
    """从 ~/memento_s/config.json 加载飞书配置，失败时返回空字典。"""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("im", {}).get("feishu", {})
    except Exception:
        return {}


class FeishuReceiver:
    """
    飞书长链接消息接收器。

    建立 WebSocket 连接，将收到的消息统一转换为 dict 后调用 on_message 回调。

    Args:
        on_message: 消息回调函数，接收一个 dict 参数（与 IMMessage 字段一致）。
                    可以是普通函数或 async 函数。
        app_id: 覆盖 FEISHU_APP_ID 环境变量
        app_secret: 覆盖 FEISHU_APP_SECRET 环境变量
    """

    def __init__(
        self,
        on_message: Callable[[dict], None],
        app_id: str | None = None,
        app_secret: str | None = None,
    ) -> None:
        cfg = _load_feishu_config()
        self._app_id = (
            app_id or cfg.get("app_id") or os.environ.get("FEISHU_APP_ID", "")
        )
        self._app_secret = (
            app_secret
            or cfg.get("app_secret")
            or os.environ.get("FEISHU_APP_SECRET", "")
        )
        self._encrypt_key = cfg.get("encrypt_key") or os.environ.get(
            "FEISHU_ENCRYPT_KEY", ""
        )
        self._verification_token = cfg.get("verification_token") or os.environ.get(
            "FEISHU_VERIFICATION_TOKEN", ""
        )
        self._on_message = on_message
        self._ws_client = None
        self._event_loop = None  # 存储事件循环引用，用于停止时
        self._bg_thread = None  # 初始化后台线程引用

        if not self._app_id or not self._app_secret:
            raise ValueError(
                f"飞书长链接需要 app_id 和 app_secret，"
                f"请在 {_CONFIG_PATH} 的 im.feishu 节填写。"
            )

    def _build_ws_client(self):
        """构建 lark-oapi WebSocket 客户端。"""
        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import P2ImMessageReceiveV1
        except ImportError:
            raise ImportError(
                "长链接模式需要 lark-oapi，请先安装：pip install lark-oapi"
            )

        def _handle(data: P2ImMessageReceiveV1) -> None:
            """将 lark-oapi 事件转换为统一 dict 格式并触发回调。"""
            logger.info("[feishu-receiver] _handle 被调用，开始解析消息")
            try:
                msg = data.event.message
                sender = data.event.sender

                # 调试日志：打印原始数据结构
                logger.info(
                    f"[feishu-receiver] Raw event data - message_id: {msg.message_id}, chat_id: {msg.chat_id}"
                )
                logger.info(
                    f"[feishu-receiver] Sender info - sender_id type: {type(sender.sender_id)},"
                    f"sender_id: {sender.sender_id}"
                )
                if sender.sender_id:
                    logger.info(
                        f"[feishu-receiver] sender_id.open_id: {sender.sender_id.open_id}"
                    )

                # 解析消息内容（飞书把 content 包在 JSON 里）
                content = ""
                try:
                    import json

                    parsed = json.loads(msg.content or "{}")
                    content = parsed.get("text", msg.content or "")
                except Exception:
                    content = msg.content or ""

                msg_dict = {
                    "id": msg.message_id or "",
                    "chat_id": msg.chat_id or "",
                    "sender_id": (sender.sender_id.open_id if sender.sender_id else ""),
                    "content": content,
                    "msg_type": msg.message_type or "text",
                    "create_time": str(msg.create_time or ""),
                    "root_id": msg.root_id or "",
                    "parent_id": msg.parent_id or "",
                    "chat_type": msg.chat_type or "",
                    "mentions": [
                        {
                            "key": m.key,
                            "name": m.name,
                            "id": m.id.open_id if m.id else "",
                        }
                        for m in (msg.mentions or [])
                    ],
                }

                logger.info(f"[feishu-receiver] Parsed message dict: {msg_dict}")

                # 支持同步和异步回调
                # _handle 由 lark SDK 在其事件循环线程中调用，
                # 不能用 asyncio.run()（会抛 RuntimeError: This event loop is already running）
                if inspect.iscoroutinefunction(self._on_message):
                    try:
                        loop = asyncio.get_running_loop()
                        # 在已有事件循环中调度协程（非阻塞，fire-and-forget）
                        loop.create_task(self._on_message(msg_dict))
                    except RuntimeError:
                        # 没有运行中的事件循环（极少情况），新建一个
                        asyncio.run(self._on_message(msg_dict))
                else:
                    self._on_message(msg_dict)

            except Exception as e:
                logger.error(f"处理飞书消息时出错：{e}", exc_info=True)

        # 正确用法：builder 方法直接注册回调，无需单独的 Handler 类
        handler = (
            lark.EventDispatcherHandler.builder(
                self._encrypt_key,
                self._verification_token,
                lark.LogLevel.INFO,
            )
            .register_p2_im_message_receive_v1(_handle)
            .register_p2_im_message_message_read_v1(
                lambda _: None
            )  # 忽略已读事件，避免 SDK 报错
            .build()
        )

        return lark.ws.Client(
            self._app_id,
            self._app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.INFO,
        )

    def _start_ws_in_thread(self) -> None:
        """在当前线程中构建并启动 ws.Client（供 run_in_executor 调用）。"""
        self._ws_client = self._build_ws_client()
        self._ws_client.start()

    def start(self) -> None:
        """
        启动长链接（阻塞）。

        在飞书开放平台配置完成后调用此方法建立连接。
        Ctrl+C 退出。

        示例：
            receiver = FeishuReceiver(on_message=my_handler)
            receiver.start()
        """
        logger.info("正在建立飞书长链接...")
        self._ws_client = self._build_ws_client()
        self._ws_client.start()

    async def start_async(self) -> None:
        """
        在 asyncio 事件循环中以非阻塞方式启动长链接。

        在后台线程运行 WebSocket 客户端，不阻塞当前协程。
        注意：ws.Client 须在后台线程内构建（避免与主线程事件循环冲突），
              此处直接调用 start_in_background() 并在后台线程中完成构建。

        示例：
            receiver = FeishuReceiver(on_message=my_async_handler)
            await receiver.start_async()
            # 继续执行其他任务...
        """
        loop = asyncio.get_event_loop()
        # 不加 await：只提交任务到线程池，不等待其返回，实现真正的非阻塞
        loop.run_in_executor(None, self._start_ws_in_thread)

    def start_in_background(self) -> threading.Thread:
        """
        在后台线程中启动长链接，立即返回。

        lark ws.Client 会在构造时捕获当前事件循环，必须在后台线程中构造，
        以避免与主线程的 asyncio.run() 事件循环冲突。

        Returns:
            后台线程对象（daemon=True，主进程退出时自动结束）

        示例：
            receiver = FeishuReceiver(on_message=my_handler)
            t = receiver.start_in_background()
            # 主线程继续运行其他逻辑
        """

        self._thread_stopped = threading.Event()

        def _run() -> None:
            logger.info("[FeishuReceiver] Background thread started")
            # 创建全新的事件循环，避免与主线程冲突
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._event_loop = loop  # 保存引用以便停止时使用

            try:
                # 强制重新导入 lark_oapi 模块，确保使用新的事件循环
                import sys
                import importlib

                # 移除已加载的 lark_oapi 模块（如果存在）
                modules_to_remove = [
                    key for key in sys.modules.keys() if key.startswith("lark_oapi")
                ]
                for mod in modules_to_remove:
                    del sys.modules[mod]
                    logger.debug(f"[FeishuReceiver] Removed module: {mod}")

                # 现在导入会使用新的事件循环
                import lark_oapi as lark
                from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

                logger.info(
                    "[FeishuReceiver] Lark modules imported fresh in background thread"
                )

                # 构建 handler（在后台线程内）
                def _handle(data: P2ImMessageReceiveV1) -> None:
                    """将 lark-oapi 事件转换为统一 dict 格式并触发回调。"""
                    logger.info("[feishu-receiver] _handle 被调用，开始解析消息")
                    try:
                        msg = data.event.message
                        sender = data.event.sender

                        # 解析消息内容（飞书把 content 包在 JSON 里）
                        content = ""
                        try:
                            import json

                            parsed = json.loads(msg.content or "{}")
                            content = parsed.get("text", msg.content or "")
                        except Exception:
                            content = msg.content or ""

                        msg_dict = {
                            "id": msg.message_id or "",
                            "chat_id": msg.chat_id or "",
                            "sender_id": (
                                sender.sender_id.open_id if sender.sender_id else ""
                            ),
                            "content": content,
                            "msg_type": msg.message_type or "text",
                            "create_time": str(msg.create_time or ""),
                            "root_id": msg.root_id or "",
                            "parent_id": msg.parent_id or "",
                            "chat_type": msg.chat_type or "",
                            "mentions": [
                                {
                                    "key": m.key,
                                    "name": m.name,
                                    "id": m.id.open_id if m.id else "",
                                }
                                for m in (msg.mentions or [])
                            ],
                        }

                        logger.info(
                            f"[feishu-receiver] Parsed message: {msg_dict.get('content', '')[:30]}..."
                        )

                        # 在后台线程的事件循环中调度回调
                        if inspect.iscoroutinefunction(self._on_message):
                            loop.create_task(self._on_message(msg_dict))
                        else:
                            # 使用线程池执行同步回调，避免阻塞事件循环
                            loop.run_in_executor(None, self._on_message, msg_dict)

                    except Exception as e:
                        logger.error(f"处理飞书消息时出错：{e}", exc_info=True)

                # 正确用法：builder 方法直接注册回调
                handler = (
                    lark.EventDispatcherHandler.builder(
                        self._encrypt_key,
                        self._verification_token,
                        lark.LogLevel.INFO,
                    )
                    .register_p2_im_message_receive_v1(_handle)
                    .register_p2_im_message_message_read_v1(
                        lambda _: None
                    )  # 忽略已读事件
                    .build()
                )

                # 创建 Client
                logger.info("[FeishuReceiver] Creating WebSocket client...")
                self._ws_client = lark.ws.Client(
                    self._app_id,
                    self._app_secret,
                    event_handler=handler,
                    log_level=lark.LogLevel.INFO,
                )

                logger.info("[FeishuReceiver] Starting WebSocket client...")
                self._ws_client.start()
            except Exception as e:
                # 区分正常关闭和真正的错误
                error_msg = str(e).lower()
                is_normal_shutdown = any(
                    [
                        "event loop stopped" in error_msg,
                        "connectionclosed" in error_msg,
                        "1000" in error_msg,  # WebSocket 正常关闭代码
                    ]
                )

                if is_normal_shutdown:
                    logger.debug(f"[FeishuReceiver] 飞书长链接已关闭: {e}")
                else:
                    logger.error(f"[FeishuReceiver] 飞书长链接启动失败: {e}")
            finally:
                self._thread_stopped.set()
                # 优雅关闭：取消所有待处理的任务
                try:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                loop.close()
                self._event_loop = None
                logger.info("[FeishuReceiver] Background thread finished")

        self._bg_thread = threading.Thread(target=_run, daemon=True)
        self._bg_thread.start()
        logger.info("[FeishuReceiver] 飞书长链接已在后台启动")
        return self._bg_thread

    def stop(self) -> None:
        """停止长链接。关闭 WebSocket 连接并等待后台线程退出。"""
        # 禁用自动重连
        if self._ws_client is not None:
            try:
                self._ws_client._auto_reconnect = False
            except Exception:
                pass

        # 断开连接
        if (
            self._ws_client is not None
            and self._event_loop
            and not self._event_loop.is_closed()
        ):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._ws_client._disconnect(), self._event_loop
                )
                future.result(timeout=3)
            except Exception:
                pass
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)

        self._ws_client = None
        self._event_loop = None

        # 等待后台线程结束
        if hasattr(self, "_bg_thread") and self._bg_thread:
            self._bg_thread.join(timeout=10)
            self._bg_thread = None


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def start_feishu_receiver(
    on_message: Callable[[dict], None],
    background: bool = False,
) -> None | threading.Thread:
    """
    快速启动飞书消息接收器。

    Args:
        on_message: 消息回调（同步或 async 函数均可）
        background: True 时在后台线程运行（非阻塞），False 时阻塞运行

    Returns:
        background=True 时返回线程对象，否则返回 None

    示例：
        # 阻塞模式（适合独立进程）
        def handle(msg):
            print(msg["content"])
        start_feishu_receiver(handle)

        # 后台模式（适合集成到现有服务）
        t = start_feishu_receiver(handle, background=True)
    """
    receiver = FeishuReceiver(on_message=on_message)
    if background:
        return receiver.start_in_background()
    receiver.start()
    return None
