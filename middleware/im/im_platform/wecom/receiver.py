"""
企业微信智能机器人长连接消息接收器。

通过 WebSocket 长连接接收企业微信智能机器人消息，同时支持通过同一连接回复。
无需公网 IP，无需 corp_id，只需机器人的 bot_id 和 secret。

连接流程：
  1. 连接到 wss://openws.work.weixin.qq.com
  2. 发送 aibot_subscribe（含 bot_id + secret）完成认证
  3. 持续监听 aibot_msg_callback（用户消息）和 aibot_event_callback（事件）
  4. 通过同一 WebSocket 连接发送 aibot_respond_msg 进行回复
  5. 每 30 秒发送 ping 心跳保持连接
  6. 收到 disconnected_event 或连接断开时自动重连

官方文档：
  https://developer.work.weixin.qq.com/document/path/101463

使用方式：

  方式一：同步阻塞（适合独立进程）
    from wecom.receiver import WecomReceiver

    def on_message(msg: dict):
        # 文字回复直接 return 字符串，或调用 msg["reply"]("回复内容")
        print(f"收到消息：{msg['sender_id']}: {msg['content']}")
        msg["reply"]("收到！")

    receiver = WecomReceiver(on_message=on_message)
    receiver.start()   # 阻塞运行，Ctrl+C 退出

  方式二：后台线程
    receiver = WecomReceiver(on_message=my_handler)
    t = receiver.start_in_background()

配置方式（~/memento_s/config.json）：
  im.wecom.bot_id  = "your_bot_id"
  im.wecom.secret  = "your_bot_secret"

依赖：
  pip install aiohttp
"""
from __future__ import annotations
import sys
import asyncio
import inspect
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable


logger = logging.getLogger(__name__)

_CONFIG_PATH = Path.home() / "memento_s" / "config.json"

_WS_URL = "wss://openws.work.weixin.qq.com"
_HEARTBEAT_INTERVAL = 30  # 秒


def _load_wecom_config() -> dict:
    """从 ~/memento_s/config.json 加载企业微信配置，失败时返回空字典。"""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("im", {}).get("wecom", {})
    except Exception:
        return {}


def _new_req_id() -> str:
    return uuid.uuid4().hex


class WecomReceiver:
    """
    企业微信智能机器人长连接消息接收器。

    连接到企业微信 WebSocket 服务器，接收用户消息并回调 on_message。
    on_message 收到的 dict 中包含 "reply" 键，值为异步回复函数。

    Args:
        on_message: 消息回调函数，接收 dict。可以是普通函数或 async 函数。
        bot_id:     覆盖 WECOM_BOT_ID 环境变量
        secret:     覆盖 WECOM_SECRET 环境变量
    """

    def __init__(
        self,
        on_message: Callable[[dict], None],
        bot_id: str | None = None,
        secret: str | None = None,
    ) -> None:
        cfg = _load_wecom_config()
        self._bot_id = bot_id or cfg.get("bot_id") or os.environ.get("WECOM_BOT_ID", "")
        self._secret = secret or cfg.get("secret") or os.environ.get("WECOM_SECRET", "")
        self._on_message = on_message
        self._stopped = False
        self._ws = None  # 当前 WebSocket 连接，用于主动发消息

        if not self._bot_id or not self._secret:
            raise ValueError(
                f"企业微信机器人长连接需要 bot_id 和 secret，"
                f"请在 {_CONFIG_PATH} 的 im.wecom 节填写。"
            )

    # -----------------------------------------------------------------------
    # 发送消息（通过 WebSocket）
    # -----------------------------------------------------------------------

    async def _ws_send(self, payload: dict) -> None:
        """向 WebSocket 发送 JSON 消息。"""
        if self._ws is None or self._ws.closed:
            raise RuntimeError(
                f"WebSocket 连接未建立或已关闭（ws={self._ws}, "
                f"closed={getattr(self._ws, 'closed', '?')}），无法发送消息"
            )
        logger.info(f"[wecom-receiver] 发送 cmd={payload.get('cmd')} ...")
        await self._ws.send_str(json.dumps(payload, ensure_ascii=False))
        logger.info(f"[wecom-receiver] 发送完成 cmd={payload.get('cmd')}")

    async def send_text(
        self,
        chat_id: str,
        text: str,
        to_user_id: str = "",
    ) -> None:
        """
        主动向指定会话或用户发送文字消息（aibot_send_msg）。

        Args:
            chat_id:     群聊 chatid，单聊时可传空字符串并填 to_user_id
            text:        消息正文
            to_user_id:  单聊时的 userid（群聊时忽略）
        """
        body: dict = {
            "aibotid": self._bot_id,
            "msgtype": "text",
            "text": {"content": text},
        }
        if chat_id:
            body["chatid"] = chat_id
        elif to_user_id:
            body["touserid"] = to_user_id

        await self._ws_send({
            "cmd": "aibot_send_msg",
            "headers": {"req_id": _new_req_id()},
            "body": body,
        })

    async def _respond_msg(
        self,
        req_id: str,
        chat_id: str,
        text: str,
        finish: bool = True,
    ) -> None:
        """
        回复指定回调消息（aibot_respond_msg）。

        Args:
            req_id:  原始 aibot_msg_callback 中的 headers.req_id
            chat_id: 目标会话 chatid（群聊必填，单聊可为空）
            text:    回复文字
            finish:  是否结束流式回复（True = 完成，False = 流式中间帧）
        """
        # 使用 stream 格式回复（官方推荐格式）
        body: dict = {
            "req_id": req_id,
            "aibotid": self._bot_id,
            "msgtype": "stream",
            "stream": {
                "id": _new_req_id(),
                "content": text,
                "finish": finish,
            },
        }
        # 群聊需要 chatid；单聊靠 req_id 关联，chatid 留空即可
        if chat_id and chat_id != "":
            body["chatid"] = chat_id
        await self._ws_send({
            "cmd": "aibot_respond_msg",
            "headers": {"req_id": _new_req_id()},
            "body": body,
        })

    # -----------------------------------------------------------------------
    # 消息解析
    # -----------------------------------------------------------------------

    def _parse_msg_callback(self, body: dict, req_id: str) -> dict:
        """将 aibot_msg_callback body 转换为统一 dict 格式。"""
        msg_type = body.get("msgtype", "text")
        content = ""

        if msg_type == "text":
            text_obj = body.get("text", {})
            content = text_obj.get("content", "") if isinstance(text_obj, dict) else str(text_obj)
        elif msg_type == "image":
            content = body.get("image", {}).get("url", "") if isinstance(body.get("image"), dict) else ""
        elif msg_type == "file":
            file_obj = body.get("file", {}) or {}
            content = file_obj.get("filename", "")
        else:
            content = str(body.get(msg_type, ""))

        sender = body.get("from", {}) or {}
        sender_id = sender.get("userid", "")
        raw_chat_id = body.get("chatid", "")  # 群聊才有真实 chatid
        chat_type = body.get("chattype", "single")
        # 单聊时 chatid 可能等于 userid 或为空，回复时不传 chatid 靠 req_id 关联
        is_group = chat_type != "single"
        chat_id = raw_chat_id if is_group else raw_chat_id or sender_id  # 展示用

        # 构建 reply 快捷函数（绑定本次回调的上下文）
        _receiver = self
        _req_id = req_id
        _reply_chat_id = raw_chat_id if is_group else ""  # 单聊不传 chatid

        async def _reply(text: str, finish: bool = True) -> None:
            try:
                body: dict = {
                    "msgtype": "stream",
                    "stream": {
                        "id": _new_req_id(),
                        "content": text,
                        "finish": finish,
                    },
                }
                if _reply_chat_id:
                    body["chatid"] = _reply_chat_id
                payload = {
                    "cmd": "aibot_respond_msg",
                    "headers": {"req_id": _req_id},  # 原始消息的 req_id 放在 headers
                    "body": body,
                }
                await _receiver._ws_send(payload)
            except BaseException as exc:
                print(f"[wecom-receiver DEBUG] _reply FAILED: {type(exc).__name__}: {exc}", flush=True, file=sys.stderr)
                raise

        return {
            "id": body.get("msgid", ""),
            "chat_id": chat_id,
            "sender_id": sender_id,
            "sender_name": sender.get("name", ""),
            "content": content.strip(),
            "msg_type": msg_type,
            "create_time": str(body.get("create_time", int(time.time() * 1000))),
            "root_id": "",
            "parent_id": "",
            "chat_type": "p2p" if chat_type == "single" else "group",
            "mentions": [],
            "platform": "wecom",
            "reply": _reply,   # 异步回复函数
            "raw": body,
        }

    # -----------------------------------------------------------------------
    # WebSocket 核心循环
    # -----------------------------------------------------------------------

    async def _run_once(self) -> None:
        """建立一次 WebSocket 连接，直到断线或主动停止。"""
        try:
            from aiohttp import ClientSession, WSMsgType
        except ImportError:
            raise ImportError("长连接模式需要 aiohttp，请先安装：pip install aiohttp")

        async with ClientSession() as session:
            async with session.ws_connect(_WS_URL) as ws:
                self._ws = ws
                logger.info(f"[wecom-receiver] WebSocket 已连接：{_WS_URL}")

                # Step 1: 发送订阅认证
                await ws.send_str(json.dumps({
                    "cmd": "aibot_subscribe",
                    "headers": {"req_id": _new_req_id()},
                    "body": {
                        "bot_id": self._bot_id,
                        "secret": self._secret,
                    },
                }, ensure_ascii=False))
                logger.info("[wecom-receiver] aibot_subscribe 已发送，等待认证响应...")

                # Step 2: 启动心跳任务
                heartbeat_task = asyncio.create_task(self._heartbeat(ws))

                try:
                    async for msg in ws:
                        if msg.type == WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                            except json.JSONDecodeError:
                                continue
                            await self._dispatch(data)

                        elif msg.type == WSMsgType.PING:
                            await ws.pong(msg.data)

                        elif msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                            logger.warning(f"[wecom-receiver] 连接关闭：{msg.type}")
                            break
                finally:
                    heartbeat_task.cancel()
                    self._ws = None

    async def _heartbeat(self, ws) -> None:
        """定时发送 ping 心跳。"""
        while not ws.closed:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            if not ws.closed:
                try:
                    await ws.ping()
                except Exception:
                    break

    async def _dispatch(self, data: dict) -> None:
        """根据 cmd 字段分发消息。"""
        cmd = data.get("cmd", "")
        headers = data.get("headers", {}) or {}
        body = data.get("body", {}) or {}
        req_id = headers.get("req_id", "")

        if cmd == "aibot_subscribe":
            # 认证响应
            errcode = body.get("errcode", 0)
            if errcode != 0:
                raise RuntimeError(
                    f"aibot_subscribe 认证失败（code={errcode}）："
                    f"{body.get('errmsg', '')}。"
                    f"请检查 bot_id 和 secret 是否正确。"
                )
            logger.info(f"[wecom-receiver] 认证成功，机器人已上线：{self._bot_id}")

        elif cmd == "aibot_msg_callback":
            # 用户消息
            msg_dict = self._parse_msg_callback(body, req_id)
            if not msg_dict.get("content"):
                return
            try:
                if inspect.iscoroutinefunction(self._on_message):
                    await self._on_message(msg_dict)
                else:
                    self._on_message(msg_dict)
            except Exception as e:
                logger.error(f"[wecom-receiver] 处理消息时出错：{e}", exc_info=True)

        elif cmd == "aibot_event_callback":
            # 事件（进入会话、连接断开等）
            event_type = (body.get("event") or {}).get("eventtype", "")
            logger.debug(f"[wecom-receiver] 收到事件：{event_type}")
            if event_type == "disconnected_event":
                logger.info("[wecom-receiver] 服务器推送 disconnected_event，即将重连")
                if self._ws and not self._ws.closed:
                    await self._ws.close()

        else:
            logger.debug(f"[wecom-receiver] 未处理的 cmd：{cmd}")

    async def _run_forever(self) -> None:
        """持续运行，自动指数退避重连。"""
        retry_delay = 5
        while not self._stopped:
            try:
                await self._run_once()
            except asyncio.CancelledError:
                break
            except RuntimeError as e:
                # 认证失败属于配置错误，不再重连
                if "认证失败" in str(e) or "aibot_subscribe" in str(e):
                    logger.error(f"[wecom-receiver] {e}")
                    self._stopped = True
                    break
                if self._stopped:
                    break
                logger.error(
                    f"[wecom-receiver] 连接异常：{e}，{retry_delay}s 后重连...",
                    exc_info=True,
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
                continue
            except Exception as e:
                if self._stopped:
                    break
                logger.error(
                    f"[wecom-receiver] 连接异常：{e}，{retry_delay}s 后重连...",
                    exc_info=True,
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
                continue
            retry_delay = 5

        logger.info("[wecom-receiver] 已停止")

    # -----------------------------------------------------------------------
    # 启动/停止接口
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """启动长连接（阻塞）。Ctrl+C 退出。"""
        logger.info("正在建立企业微信智能机器人长连接...")
        self._stopped = False
        try:
            asyncio.run(self._run_forever())
        except KeyboardInterrupt:
            self._stopped = True

    def start_in_background(self) -> threading.Thread:
        """
        在后台线程中启动长连接，立即返回。

        Returns:
            后台线程对象（daemon=True）
        """
        self._stopped = False
        self._loop_ready = threading.Event()
        self._bg_loop: asyncio.AbstractEventLoop | None = None

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._bg_loop = loop
            self._loop_ready.set()
            try:
                loop.run_until_complete(self._run_forever())
            except Exception as e:
                if not self._stopped:
                    logger.error(f"[wecom-receiver] 后台线程异常：{e}", exc_info=True)
            finally:
                loop.close()

        self._bg_thread = threading.Thread(target=_run, daemon=True, name="wecom-receiver")
        self._bg_thread.start()
        self._loop_ready.wait(timeout=5)
        logger.info("企业微信智能机器人长连接已在后台启动")
        return self._bg_thread

    def stop(self) -> None:
        """停止长连接。"""
        self._stopped = True
        if hasattr(self, "_bg_loop") and self._bg_loop and not self._bg_loop.is_closed():
            self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        if hasattr(self, "_bg_thread"):
            self._bg_thread.join(timeout=5)
        logger.info("[wecom-receiver] 已停止")


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def start_wecom_receiver(
    on_message: Callable[[dict], None],
    background: bool = False,
) -> None | threading.Thread:
    """
    快速启动企业微信智能机器人长连接接收器。

    Args:
        on_message: 消息回调（同步或 async 函数均可）
        background: True 时后台线程运行（非阻塞），False 时阻塞运行

    Returns:
        background=True 时返回线程对象，否则返回 None
    """
    receiver = WecomReceiver(on_message=on_message)
    if background:
        return receiver.start_in_background()
    receiver.start()
    return None
