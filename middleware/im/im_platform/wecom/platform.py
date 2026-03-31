"""
企业微信（WeCom）Open API 适配器。

支持两种认证模式（优先从 ~/memento_s/config.json 读取配置）：
  1. 应用凭证模式：im.wecom.corp_id + im.wecom.secret + im.wecom.agent_id → 完整 Open API 权限
  2. Webhook 模式：im.wecom.webhook_url                                       → 仅发送消息到群（自定义机器人）

Token 管理：
  - 内存缓存 access_token，过期前 60 秒自动刷新
  - 企业微信 access_token 默认有效期 7200 秒
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from ..base import (
    IMAuthError,
    IMChat,
    IMError,
    IMMessage,
    IMNotFoundError,
    IMUser,
    IMWebhookOnlyError,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://qyapi.weixin.qq.com"
TOKEN_REFRESH_BEFORE_EXPIRE = 60  # 提前 60 秒刷新

_CONFIG_PATH = Path.home() / "memento_s" / "config.json"


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def _load_wecom_config() -> dict:
    """从 ~/memento_s/config.json 加载企业微信配置，失败时返回空字典。"""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("im", {}).get("wecom", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 企业微信适配器
# ---------------------------------------------------------------------------

class WecomPlatform:
    """
    企业微信 Open API 适配器。

    优先从 ~/memento_s/config.json 的 im.wecom 节读取配置，
    缺失时回退到同名环境变量（WECOM_CORP_ID / WECOM_SECRET / WECOM_AGENT_ID / WECOM_WEBHOOK_URL）。
    """

    def __init__(self) -> None:
        cfg = _load_wecom_config()
        self._corp_id = cfg.get("corp_id") or os.environ.get("WECOM_CORP_ID", "")
        self._secret = cfg.get("secret") or os.environ.get("WECOM_SECRET", "")
        self._agent_id = cfg.get("agent_id") or os.environ.get("WECOM_AGENT_ID", "")
        self._webhook_url = cfg.get("webhook_url") or os.environ.get("WECOM_WEBHOOK_URL", "")
        self._base_url = (cfg.get("base_url") or os.environ.get("WECOM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")

        self._use_webhook_only = bool(self._webhook_url and not (self._corp_id and self._secret))

        if not self._corp_id and not self._webhook_url:
            raise IMAuthError(
                f"企业微信配置缺失：请在 {_CONFIG_PATH} 的 im.wecom 节填写 "
                "corp_id + secret + agent_id（完整模式）或 webhook_url（仅发送模式）",
                platform="wecom",
            )

        # Token 缓存
        self._token: str = ""
        self._token_expire_at: float = 0.0
        self._token_lock = asyncio.Lock()

    # -----------------------------------------------------------------------
    # Token 管理
    # -----------------------------------------------------------------------

    async def _get_token(self) -> str:
        """获取有效的 access_token（带缓存和自动刷新）。"""
        if self._use_webhook_only:
            raise IMWebhookOnlyError("Webhook 模式不支持此操作，请配置 corp_id 和 secret")

        async with self._token_lock:
            if time.time() < self._token_expire_at - TOKEN_REFRESH_BEFORE_EXPIRE:
                return self._token
            await self._refresh_token()
            return self._token

    async def _refresh_token(self) -> None:
        url = f"{self._base_url}/cgi-bin/gettoken"
        params = {"corpid": self._corp_id, "corpsecret": self._secret}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
        data = resp.json()
        if data.get("errcode", 0) != 0:
            raise IMAuthError(
                f"企业微信 Token 获取失败：{data.get('errmsg', '')} (code={data.get('errcode')})",
                code=data.get("errcode", 0),
                platform="wecom",
            )
        self._token = data["access_token"]
        self._token_expire_at = time.time() + data.get("expires_in", 7200)

    # -----------------------------------------------------------------------
    # HTTP 工具
    # -----------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        files: dict | None = None,
        stream_to: str | None = None,
        retry: bool = True,
    ) -> dict:
        """发起已认证的 API 请求，自动处理 token 过期重试。"""
        token = await self._get_token()
        url = f"{self._base_url}{path}"
        req_params = {"access_token": token, **(params or {})}

        async with httpx.AsyncClient(timeout=30) as client:
            if stream_to:
                async with client.stream(method, url, params=req_params) as resp:
                    resp.raise_for_status()
                    with open(stream_to, "wb") as f:
                        async for chunk in resp.aiter_bytes():
                            f.write(chunk)
                return {"errcode": 0, "errmsg": "ok"}

            if files:
                resp = await client.request(method, url, params=req_params, files=files)
            elif json_body is not None:
                resp = await client.request(method, url, params=req_params, json=json_body)
            else:
                resp = await client.request(method, url, params=req_params)

        data = resp.json()

        # 处理企业微信 API 错误
        if isinstance(data, dict):
            errcode = data.get("errcode", 0)
            errmsg = data.get("errmsg", "")

            # access_token 过期
            if errcode in (40014, 42001) and retry:
                async with self._token_lock:
                    await self._refresh_token()
                return await self._request(
                    method, path,
                    params=params, json_body=json_body,
                    files=files, stream_to=stream_to, retry=False,
                )

            if errcode != 0:
                if errcode in (40013, 40014, 42001):
                    raise IMAuthError(f"企业微信认证失败：{errmsg}", code=errcode, platform="wecom")
                if errcode == 46004:
                    raise IMNotFoundError(f"成员不存在：{errmsg}", code=errcode, platform="wecom")
                raise IMError(f"企业微信 API 错误 {errcode}：{errmsg}", code=errcode, platform="wecom")

        return data

    # -----------------------------------------------------------------------
    # 消息
    # -----------------------------------------------------------------------

    async def send_message(
        self,
        receive_id: str,
        content: str,
        msg_type: str = "text",
        receive_id_type: str = "touser",
    ) -> IMMessage:
        """
        发送消息。

        Args:
            receive_id: 接收者 ID（userid | chatid | 部门 id）
            content: 消息内容（text 时为纯文字；markdown 时为 MD 文字；其他类型为 JSON 字符串）
            msg_type: text | markdown | image | file | news（图文）
            receive_id_type: touser（成员 userid）| toparty（部门 id）|
                             totag（标签 id）| chatid（appchat 群 id）
        """
        if self._use_webhook_only:
            return await self._send_via_webhook(content, msg_type)

        if receive_id_type == "chatid":
            return await self._send_appchat(receive_id, content, msg_type)

        payload = _build_app_message(
            receive_id=receive_id,
            receive_id_type=receive_id_type,
            content=content,
            msg_type=msg_type,
            agent_id=int(self._agent_id) if self._agent_id else 0,
        )

        data = await self._request("POST", "/cgi-bin/message/send", json_body=payload)
        return IMMessage(
            id=data.get("msgid", ""),
            chat_id=receive_id,
            sender_id="bot",
            content=content,
            msg_type=msg_type,
            create_time=str(int(time.time() * 1000)),
            raw=data,
        )

    async def _send_appchat(self, chatid: str, content: str, msg_type: str) -> IMMessage:
        """发送消息到应用群聊（appchat）。"""
        payload = _build_appchat_message(chatid=chatid, content=content, msg_type=msg_type)
        data = await self._request("POST", "/cgi-bin/appchat/send", json_body=payload)
        return IMMessage(
            id=data.get("msgid", ""),
            chat_id=chatid,
            sender_id="bot",
            content=content,
            msg_type=msg_type,
            create_time=str(int(time.time() * 1000)),
            raw=data,
        )

    async def _send_via_webhook(self, content: str, msg_type: str) -> IMMessage:
        """通过自定义机器人 Webhook 发送消息。"""
        if msg_type == "markdown":
            payload = {"msgtype": "markdown", "markdown": {"content": content}}
        else:
            payload = {
                "msgtype": "text",
                "text": {"content": content, "mentioned_list": [], "mentioned_mobile_list": []}
                }

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(self._webhook_url, json=payload)
        data = resp.json()
        if data.get("errcode", 0) != 0:
            raise IMError(
                f"企业微信 Webhook 发送失败：{data.get('errmsg', '')}",
                platform="wecom",
            )
        return IMMessage(
            id="", chat_id="", sender_id="bot",
            content=content, msg_type=msg_type,
            create_time=str(int(time.time() * 1000)),
        )

    async def reply_message(
        self,
        message_id: str,
        content: str,
        msg_type: str = "text",
        receive_id_type: str = "touser",
    ) -> IMMessage:
        """
        回复消息（企业微信不支持直接回复，通过 send_message 发回给原来源）。

        Args:
            message_id:      接收者 ID（userid 或 chatid，由 receive_id_type 决定）
            receive_id_type: touser（单聊，默认）| chatid（群聊）
        """
        return await self.send_message(
            receive_id=message_id,
            content=content,
            msg_type=msg_type,
            receive_id_type=receive_id_type,
        )

    async def get_message(self, message_id: str) -> IMMessage:
        """企业微信不支持按 message_id 获取消息。"""
        raise IMError("企业微信 API 不支持直接按 message_id 获取消息", platform="wecom")

    async def list_messages(
        self,
        container_id: str,
        container_id_type: str = "chatid",
        page_size: int = 20,
        start_time: str = "",
        end_time: str = "",
    ) -> list[IMMessage]:
        """企业微信不提供消息列表 API。"""
        raise IMError("企业微信 API 不支持获取历史消息列表", platform="wecom")

    # -----------------------------------------------------------------------
    # 群组/会话
    # -----------------------------------------------------------------------

    async def get_chat(self, chat_id: str) -> IMChat:
        """获取应用群聊信息。"""
        data = await self._request("GET", "/cgi-bin/appchat/get", params={"chatid": chat_id})
        chat_info = data.get("chat_info", data)
        return IMChat(
            id=chat_info.get("chatid", chat_id),
            name=chat_info.get("name", ""),
            chat_type="group",
            description=chat_info.get("owner", ""),
            member_count=len(chat_info.get("memberlist", [])),
            owner_id=chat_info.get("owner", ""),
            raw=chat_info,
        )

    async def list_chat_members(self, chat_id: str, page_size: int = 50) -> list[IMUser]:
        """列出群聊成员。"""
        chat = await self.get_chat(chat_id)
        member_ids = [m for m in chat.raw.get("memberlist", [])]
        # 批量获取成员信息
        users = []
        for uid in member_ids[:page_size]:
            try:
                user = await self.get_user(uid)
                users.append(user)
            except Exception:
                users.append(IMUser(id=uid, name=uid))
        return users

    # -----------------------------------------------------------------------
    # 用户
    # -----------------------------------------------------------------------

    async def get_user(self, user_id: str, id_type: str = "userid") -> IMUser:
        """获取成员信息。"""
        data = await self._request("GET", "/cgi-bin/user/get", params={"userid": user_id})
        return _parse_user(data)

    async def search_users(self, query: str, page_size: int = 10) -> list[IMUser]:
        """搜索成员（按姓名模糊匹配部门一级列表）。"""
        # 企业微信没有直接搜索 API，获取根部门成员列表做本地过滤
        data = await self._request(
            "GET",
            "/cgi-bin/user/list",
            params={"department_id": 1, "fetch_child": 1},
        )
        user_list = data.get("userlist", [])
        query_lower = query.lower()
        results = [
            _parse_user(u) for u in user_list
            if query_lower in (u.get("name", "") + u.get("userid", "") + u.get("email", "")).lower()
        ]
        return results[:page_size]

    # -----------------------------------------------------------------------
    # 文件/资源
    # -----------------------------------------------------------------------

    async def upload_image(self, file_path: str) -> str:
        """上传图片，返回 media_id。"""
        path = Path(file_path)
        with open(path, "rb") as f:
            files = {"media": (path.name, f, "image/jpeg")}
            data = await self._request(
                "POST",
                "/cgi-bin/media/upload",
                params={"type": "image"},
                files=files,
            )
        return data.get("media_id", "")

    async def upload_file(self, file_path: str, file_type: str = "file") -> str:
        """上传文件，返回 media_id。"""
        path = Path(file_path)
        with open(path, "rb") as f:
            files = {"media": (path.name, f, "application/octet-stream")}
            data = await self._request(
                "POST",
                "/cgi-bin/media/upload",
                params={"type": file_type},
                files=files,
            )
        return data.get("media_id", "")

    async def download_file(self, file_key: str, save_path: str) -> str:
        """下载媒体文件到本地。"""
        await self._request(
            "GET",
            "/cgi-bin/media/get",
            params={"media_id": file_key},
            stream_to=save_path,
        )
        return save_path


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _build_app_message(
    receive_id: str,
    receive_id_type: str,
    content: str,
    msg_type: str,
    agent_id: int,
) -> dict[str, Any]:
    """构建应用消息发送 payload。"""
    payload: dict[str, Any] = {
        receive_id_type: receive_id,
        "agentid": agent_id,
        "safe": 0,
    }

    if msg_type == "text":
        payload["msgtype"] = "text"
        payload["text"] = {"content": content}
    elif msg_type == "markdown":
        payload["msgtype"] = "markdown"
        payload["markdown"] = {"content": content}
    elif msg_type == "image":
        payload["msgtype"] = "image"
        payload["image"] = {"media_id": content}
    elif msg_type == "file":
        payload["msgtype"] = "file"
        payload["file"] = {"media_id": content}
    elif msg_type == "news":
        try:
            articles = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            articles = [{"title": "消息", "description": content, "url": ""}]
        payload["msgtype"] = "news"
        payload["news"] = {"articles": articles if isinstance(articles, list) else [articles]}
    else:
        # 默认 text
        payload["msgtype"] = "text"
        payload["text"] = {"content": content}

    return payload


def _build_appchat_message(chatid: str, content: str, msg_type: str) -> dict[str, Any]:
    """构建 appchat 群消息 payload。"""
    payload: dict[str, Any] = {"chatid": chatid, "safe": 0}

    if msg_type == "text":
        payload["msgtype"] = "text"
        payload["text"] = {"content": content}
    elif msg_type == "markdown":
        payload["msgtype"] = "markdown"
        payload["markdown"] = {"content": content}
    elif msg_type == "image":
        payload["msgtype"] = "image"
        payload["image"] = {"media_id": content}
    elif msg_type == "file":
        payload["msgtype"] = "file"
        payload["file"] = {"media_id": content}
    else:
        payload["msgtype"] = "text"
        payload["text"] = {"content": content}

    return payload


def _parse_user(raw: dict) -> IMUser:
    return IMUser(
        id=raw.get("userid", ""),
        name=raw.get("name", ""),
        open_id=raw.get("userid", ""),
        email=raw.get("email", raw.get("biz_mail", "")),
        mobile=raw.get("mobile", ""),
        department=str(raw.get("department", [""])[0]) if raw.get("department") else "",
        raw=raw,
    )
