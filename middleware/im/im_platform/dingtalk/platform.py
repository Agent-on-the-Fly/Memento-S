"""
钉钉（DingTalk）Open API 适配器。

支持两种认证模式（优先从 ~/memento_s/config.json 读取配置）：
  1. 应用凭证模式：im.dingtalk.app_key + im.dingtalk.app_secret  → 完整 Open API 权限
  2. Webhook 模式：im.dingtalk.webhook_url                       → 仅发送消息到群（自定义机器人）

Token 管理：
  - 内存缓存 access_token，过期前 60 秒自动刷新
  - Webhook 签名：若配置了 webhook_secret，自动生成 timestamp+sign
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from base64 import b64encode
from pathlib import Path
from typing import Any
from urllib.parse import quote

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

DEFAULT_BASE_URL = "https://api.dingtalk.com"
TOKEN_REFRESH_BEFORE_EXPIRE = 60  # 提前 60 秒刷新

_CONFIG_PATH = Path.home() / "memento_s" / "config.json"


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def _load_dingtalk_config() -> dict:
    """从 ~/memento_s/config.json 加载钉钉配置，失败时返回空字典。"""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("im", {}).get("dingtalk", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 钉钉适配器
# ---------------------------------------------------------------------------

class DingTalkPlatform:
    """
    钉钉 Open API 适配器。

    优先从 ~/memento_s/config.json 的 im.dingtalk 节读取配置，
    缺失时回退到同名环境变量（DINGTALK_APP_KEY / DINGTALK_APP_SECRET / DINGTALK_WEBHOOK_URL）。
    """

    def __init__(self) -> None:
        cfg = _load_dingtalk_config()
        self._app_key = cfg.get("app_key") or os.environ.get("DINGTALK_APP_KEY", "")
        self._app_secret = cfg.get("app_secret") or os.environ.get("DINGTALK_APP_SECRET", "")
        self._webhook_url = cfg.get("webhook_url") or os.environ.get("DINGTALK_WEBHOOK_URL", "")
        self._webhook_secret = cfg.get("webhook_secret") or os.environ.get("DINGTALK_WEBHOOK_SECRET", "")
        self._base_url = (cfg.get("base_url") or os.environ.get("DINGTALK_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")

        self._use_webhook_only = bool(self._webhook_url and not (self._app_key and self._app_secret))

        if not self._app_key and not self._webhook_url:
            raise IMAuthError(
                f"钉钉配置缺失：请在 {_CONFIG_PATH} 的 im.dingtalk 节填写 "
                "app_key + app_secret（完整模式）或 webhook_url（仅发送模式）",
                platform="dingtalk",
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
            raise IMWebhookOnlyError("Webhook 模式不支持此操作，请配置 DINGTALK_APP_KEY 和 DINGTALK_APP_SECRET")

        async with self._token_lock:
            if time.time() < self._token_expire_at - TOKEN_REFRESH_BEFORE_EXPIRE:
                return self._token
            await self._refresh_token()
            return self._token

    async def _refresh_token(self) -> None:
        url = f"{self._base_url}/v1.0/oauth2/accessToken"
        payload = {"appKey": self._app_key, "appSecret": self._app_secret}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
        data = resp.json()
        if "accessToken" not in data:
            raise IMAuthError(
                f"钉钉 Token 获取失败：{data}",
                platform="dingtalk",
            )
        self._token = data["accessToken"]
        self._token_expire_at = time.time() + data.get("expireIn", 7200)

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
        headers = {
            "x-acs-dingtalk-access-token": token,
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}{path}"

        async with httpx.AsyncClient(timeout=30) as client:
            if stream_to:
                async with client.stream(method, url, headers=headers, params=params) as resp:
                    resp.raise_for_status()
                    with open(stream_to, "wb") as f:
                        async for chunk in resp.aiter_bytes():
                            f.write(chunk)
                return {"success": True}

            if files:
                del headers["Content-Type"]
                resp = await client.request(method, url, headers=headers, files=files, params=params)
            elif json_body is not None:
                resp = await client.request(method, url, headers=headers, json=json_body, params=params)
            else:
                resp = await client.request(method, url, headers=headers, params=params)

        data = resp.json()

        # 处理钉钉 v2 API 错误格式
        if isinstance(data, dict):
            code = data.get("code", "")
            # InvalidAuthentication 表示 token 过期
            if code == "InvalidAuthentication" and retry:
                async with self._token_lock:
                    await self._refresh_token()
                return await self._request(method, path, params=params, json_body=json_body,
                                           files=files, stream_to=stream_to, retry=False)

            if "errcode" in data and data["errcode"] != 0:
                errcode = data["errcode"]
                errmsg = data.get("errmsg", "unknown error")
                if errcode in (40001, 40014, 42001):
                    raise IMAuthError(f"钉钉 API 权限不足：{errmsg}", code=errcode, platform="dingtalk")
                if errcode == 40078:
                    raise IMNotFoundError(f"资源不存在：{errmsg}", code=errcode, platform="dingtalk")
                raise IMError(f"钉钉 API 错误 {errcode}：{errmsg}", code=errcode, platform="dingtalk")

            if code and code != "0" and "message" in data:
                msg = data.get("message", "unknown error")
                raise IMError(f"钉钉 API 错误 {code}：{msg}", platform="dingtalk")

        return data

    # -----------------------------------------------------------------------
    # 消息
    # -----------------------------------------------------------------------

    async def send_message(
        self,
        receive_id: str,
        content: str,
        msg_type: str = "text",
        receive_id_type: str = "staffId",
    ) -> IMMessage:
        """
        发送消息。

        Args:
            receive_id: 接收者 ID（staffId、unionId 或 openConversationId）
            content: 消息内容（text 时为纯文字，其他类型为 JSON 字符串）
            msg_type: sampleText | sampleMarkdown | sampleImageMsg | sampleFile
            receive_id_type: staffId | unionId | openConversationId（群聊）
        """
        if self._use_webhook_only:
            return await self._send_via_webhook(content, msg_type)

        msg_key, msg_param = _build_msg_param(content, msg_type)

        if receive_id_type == "openConversationId":
            # 群聊消息
            data = await self._request(
                "POST",
                "/v1.0/robot/groupMessages/send",
                json_body={
                    "robotCode": self._app_key,
                    "openConversationId": receive_id,
                    "msgKey": msg_key,
                    "msgParam": msg_param,
                },
            )
        else:
            # 1:1 消息
            data = await self._request(
                "POST",
                "/v1.0/robot/oToMessages/batchSend",
                json_body={
                    "robotCode": self._app_key,
                    "userIds": [receive_id],
                    "msgKey": msg_key,
                    "msgParam": msg_param,
                },
            )

        return IMMessage(
            id=data.get("processQueryKey", ""),
            chat_id=receive_id,
            sender_id="bot",
            content=content,
            msg_type=msg_type,
            create_time=str(int(time.time() * 1000)),
            raw=data,
        )

    async def _send_via_webhook(self, content: str, msg_type: str) -> IMMessage:
        """通过自定义机器人 Webhook 发送消息（支持签名）。"""
        url = self._webhook_url

        # 生成签名（如果配置了 webhook_secret）
        if self._webhook_secret:
            ts = str(int(time.time() * 1000))
            sign = _calc_webhook_sign(ts, self._webhook_secret)
            url = f"{url}&timestamp={ts}&sign={quote(sign)}"

        if msg_type in ("sampleMarkdown", "markdown"):
            payload = {
                "msgtype": "markdown",
                "markdown": {"title": "消息", "text": content},
            }
        else:
            payload = {"msgtype": "text", "text": {"content": content}}

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
        data = resp.json()
        if data.get("errcode", 0) != 0:
            raise IMError(
                f"Webhook 发送失败：{data.get('errmsg', '')}",
                platform="dingtalk",
            )
        return IMMessage(id="", chat_id="", sender_id="bot", content=content,
                         msg_type=msg_type, create_time=str(int(time.time() * 1000)))

    async def reply_message(
        self,
        message_id: str,
        content: str,
        msg_type: str = "text",
    ) -> IMMessage:
        """
        回复指定消息（钉钉通过会话 ID 回复，message_id 此处为 openConversationId）。
        """
        return await self.send_message(
            receive_id=message_id,
            content=content,
            msg_type=msg_type,
            receive_id_type="openConversationId",
        )

    async def get_message(self, message_id: str) -> IMMessage:
        """获取单条消息（钉钉不直接支持，返回占位对象）。"""
        raise IMError("钉钉 API 不支持直接按 message_id 获取消息", platform="dingtalk")

    async def list_messages(
        self,
        container_id: str,
        container_id_type: str = "openConversationId",
        page_size: int = 20,
        start_time: str = "",
        end_time: str = "",
    ) -> list[IMMessage]:
        """列出会话消息历史（需要高级权限）。"""
        # 钉钉 v1.0 query 接口统一用 POST + JSON body，不能用 GET + query params
        json_body: dict[str, Any] = {
            "openConversationId": container_id,
            "maxResults": min(page_size, 100),
        }
        if start_time:
            json_body["queryStartTime"] = start_time
        if end_time:
            json_body["queryEndTime"] = end_time

        data = await self._request("POST", "/v1.0/im/conversations/messages/query", json_body=json_body)
        items = data.get("messageList", []) or []
        return [_parse_message(item) for item in items]

    # -----------------------------------------------------------------------
    # 群组/会话
    # -----------------------------------------------------------------------

    async def get_chat(self, chat_id: str) -> IMChat:
        """获取群组信息。"""
        data = await self._request(
            "POST",
            "/v1.0/im/conversations/get",
            json_body={"openConversationId": chat_id},
        )
        return _parse_chat(data)

    async def search_chats(self, query: str, page_size: int = 20) -> list[IMChat]:
        """搜索群组（需要企业授权）。"""
        raise IMError("钉钉 API 暂不支持群组搜索，请使用 openConversationId 直接访问", platform="dingtalk")

    async def list_chat_members(self, chat_id: str, page_size: int = 50) -> list[IMUser]:
        """列出群成员。"""
        data = await self._request(
            "POST",
            "/v1.0/im/conversations/members/query",
            json_body={
                "openConversationId": chat_id,
                "maxResults": min(page_size, 100),
            },
        )
        members = data.get("memberInfoList", []) or []
        return [
            IMUser(
                id=m.get("staffId", ""),
                name=m.get("nick", ""),
                open_id=m.get("unionId", ""),
                raw=m,
            )
            for m in members
        ]

    # -----------------------------------------------------------------------
    # 用户
    # -----------------------------------------------------------------------

    async def get_user(self, user_id: str, id_type: str = "staffId") -> IMUser:
        """获取用户信息。"""
        if user_id == "me":
            # /me 接口获取当前调用者信息，不需要额外 body
            data = await self._request("GET", "/v1.0/contact/users/me")
            return _parse_user(data.get("result", data))

        # 按 staffId 批量查询，取第一个结果
        data = await self._request(
            "POST",
            "/v1.0/contact/users",
            json_body={"staffIds": [user_id]},
        )
        users = data.get("result", {}).get("userList", [])
        if not users:
            raise IMNotFoundError(f"用户不存在：{user_id}", platform="dingtalk")
        return _parse_user(users[0])

    async def search_users(self, query: str, page_size: int = 10) -> list[IMUser]:
        """搜索用户。"""
        data = await self._request(
            "POST",
            "/v1.0/contact/users/search",
            json_body={"query": query, "maxResults": min(page_size, 50)},
        )
        users = data.get("result", []) or []
        return [_parse_user(u) for u in users]

    # -----------------------------------------------------------------------
    # 文件/资源
    # -----------------------------------------------------------------------

    async def upload_image(self, file_path: str) -> str:
        """上传图片，返回 mediaId。"""
        path = Path(file_path)
        with open(path, "rb") as f:
            files = {"media": (path.name, f, "image/jpeg")}
            data = await self._request(
                "POST",
                "/v1.0/robot/mediaFiles/upload",
                params={"robotCode": self._app_key, "mediaType": "image"},
                files=files,
            )
        return data.get("mediaId", "")

    async def upload_file(self, file_path: str, file_type: str = "file") -> str:
        """上传文件，返回 mediaId。"""
        path = Path(file_path)
        with open(path, "rb") as f:
            files = {"media": (path.name, f, "application/octet-stream")}
            data = await self._request(
                "POST",
                "/v1.0/robot/mediaFiles/upload",
                params={"robotCode": self._app_key, "mediaType": file_type},
                files=files,
            )
        return data.get("mediaId", "")

    async def download_file(self, file_key: str, save_path: str) -> str:
        """下载文件资源，保存到 save_path。"""
        await self._request(
            "GET",
            f"/v1.0/robot/mediaFiles/{file_key}/download",
            stream_to=save_path,
        )
        return save_path


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _calc_webhook_sign(timestamp: str, secret: str) -> str:
    """计算钉钉 Webhook 签名（HMAC-SHA256）。"""
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    return b64encode(hmac_code).decode("utf-8")


def _build_msg_param(content: str, msg_type: str) -> tuple[str, str]:
    """构建钉钉机器人消息的 msgKey 和 msgParam（JSON 字符串）。"""
    if msg_type in ("text", "sampleText"):
        return "sampleText", json.dumps({"content": content}, ensure_ascii=False)
    if msg_type in ("markdown", "sampleMarkdown"):
        try:
            parsed = json.loads(content)
            return "sampleMarkdown", json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return "sampleMarkdown", json.dumps({"title": "消息", "text": content}, ensure_ascii=False)
    # 其他类型：用户自行传完整 msgKey+JSON
    try:
        parsed = json.loads(content)
        key = parsed.pop("msgKey", msg_type)
        return key, json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return "sampleText", json.dumps({"content": content}, ensure_ascii=False)


def _parse_message(raw: dict) -> IMMessage:
    body = raw.get("messageBody", raw)
    content = body.get("content", raw.get("text", {}).get("content", ""))
    if isinstance(content, dict):
        content = content.get("content", str(content))
    return IMMessage(
        id=raw.get("messageId", ""),
        chat_id=raw.get("conversationId", raw.get("openConversationId", "")),
        sender_id=raw.get("senderId", raw.get("senderStaffId", "")),
        content=content,
        msg_type=raw.get("messageType", "text"),
        create_time=str(raw.get("sendTime", int(time.time() * 1000))),
        raw=raw,
    )


def _parse_chat(raw: dict) -> IMChat:
    return IMChat(
        id=raw.get("openConversationId", ""),
        name=raw.get("title", raw.get("name", "")),
        chat_type="group" if raw.get("conversationType") == "2" else "p2p",
        description=raw.get("description", ""),
        member_count=raw.get("memberCount", 0),
        owner_id=raw.get("ownerUserId", ""),
        raw=raw,
    )


def _parse_user(raw: dict) -> IMUser:
    return IMUser(
        id=raw.get("staffId", raw.get("userid", "")),
        name=raw.get("name", raw.get("nick", "")),
        open_id=raw.get("unionId", ""),
        union_id=raw.get("unionId", ""),
        email=raw.get("email", raw.get("orgEmail", "")),
        mobile=raw.get("mobile", ""),
        department=str(raw.get("deptIdList", [""])[0]) if raw.get("deptIdList") else "",
        raw=raw,
    )
