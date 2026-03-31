"""
飞书（Lark）Open API 适配器。

支持两种认证模式（优先从 ~/memento_s/config.json 读取配置）：
  1. 应用凭证模式：im.feishu.app_id + im.feishu.app_secret  → 完整 Open API 权限
  2. Webhook 模式：im.feishu.webhook_url                    → 仅发送消息到群/机器人

Token 管理：
  - 内存缓存 tenant_access_token，过期前 60 秒自动刷新
  - 遇到 99991671（token 过期）自动刷新后重试一次
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
    IMRateLimitError,
    IMUser,
    IMWebhookOnlyError,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://open.feishu.cn"
TOKEN_REFRESH_BEFORE_EXPIRE = 60  # 提前 60 秒刷新

_CONFIG_PATH = Path.home() / "memento_s" / "config.json"


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def _load_feishu_config() -> dict:
    """从 ~/memento_s/config.json 加载飞书配置，失败时返回空字典。"""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("im", {}).get("feishu", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 飞书适配器
# ---------------------------------------------------------------------------

class FeishuPlatform:
    """
    飞书 Open API 适配器。

    优先从 ~/memento_s/config.json 的 im.feishu 节读取配置，
    缺失时回退到同名环境变量（FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_WEBHOOK_URL）。
    """

    def __init__(self) -> None:
        cfg = _load_feishu_config()
        self._app_id = cfg.get("app_id") or os.environ.get("FEISHU_APP_ID", "")
        self._app_secret = cfg.get("app_secret") or os.environ.get("FEISHU_APP_SECRET", "")
        self._webhook_url = cfg.get("webhook_url") or os.environ.get("FEISHU_WEBHOOK_URL", "")
        self._base_url = (cfg.get("base_url") or os.environ.get("FEISHU_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")

        self._use_webhook_only = bool(self._webhook_url and not (self._app_id and self._app_secret))

        if not self._app_id and not self._webhook_url:
            raise IMAuthError(
                f"飞书配置缺失：请在 {_CONFIG_PATH} 的 im.feishu 节填写 "
                "app_id + app_secret（完整模式）或 webhook_url（仅发送模式）",
                platform="feishu",
            )

        # Token 缓存
        self._token: str = ""
        self._token_expire_at: float = 0.0
        self._token_lock = asyncio.Lock()

    # -----------------------------------------------------------------------
    # Token 管理
    # -----------------------------------------------------------------------

    async def _get_token(self) -> str:
        """获取有效的 tenant_access_token（带缓存和自动刷新）。"""
        if self._use_webhook_only:
            raise IMWebhookOnlyError("Webhook 模式不支持此操作，请配置 FEISHU_APP_ID 和 FEISHU_APP_SECRET")

        async with self._token_lock:
            if time.time() < self._token_expire_at - TOKEN_REFRESH_BEFORE_EXPIRE:
                return self._token
            await self._refresh_token()
            return self._token

    async def _refresh_token(self) -> None:
        url = f"{self._base_url}/open-apis/auth/v3/tenant_access_token/internal"
        payload = {"app_id": self._app_id, "app_secret": self._app_secret}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
        data = resp.json()
        if data.get("code") != 0:
            raise IMAuthError(
                f"飞书 Token 获取失败：{data.get('msg', '')}",
                code=data.get("code", -1),
                platform="feishu",
            )
        self._token = data["tenant_access_token"]
        self._token_expire_at = time.time() + data.get("expire", 7200)

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
        headers = {"Authorization": f"Bearer {token}"}
        url = f"{self._base_url}{path}"

        async with httpx.AsyncClient(timeout=30) as client:
            if stream_to:
                async with client.stream(method, url, headers=headers, params=params) as resp:
                    resp.raise_for_status()
                    with open(stream_to, "wb") as f:
                        async for chunk in resp.aiter_bytes():
                            f.write(chunk)
                return {"code": 0}

            if files:
                resp = await client.request(method, url, headers=headers, files=files, params=params)
            elif json_body is not None:
                resp = await client.request(method, url, headers=headers, json=json_body, params=params)
            else:
                resp = await client.request(method, url, headers=headers, params=params)

        data = resp.json()
        code = data.get("code", -1)

        # Token 过期，刷新后重试一次
        if code == 99991671 and retry:
            async with self._token_lock:
                await self._refresh_token()
            return await self._request(method, path, params=params, json_body=json_body,
                                       files=files, stream_to=stream_to, retry=False)

        if code != 0:
            msg = data.get("msg", "unknown error")
            if code in (99991663, 99991664, 99991665):
                raise IMAuthError(f"飞书 API 权限不足：{msg}", code=code, platform="feishu")
            if code == 1300007:
                raise IMNotFoundError(f"资源不存在：{msg}", code=code, platform="feishu")
            if code == 99991400:
                raise IMRateLimitError(f"请求频率超限：{msg}", code=code, platform="feishu")
            raise IMError(f"飞书 API 错误 {code}：{msg}", code=code, platform="feishu")

        return data

    # -----------------------------------------------------------------------
    # 消息
    # -----------------------------------------------------------------------

    async def send_message(
        self,
        receive_id: str,
        content: str,
        msg_type: str = "text",
        receive_id_type: str = "open_id",
    ) -> IMMessage:
        """发送消息（两种认证模式均支持 text 类型）。"""
        if self._use_webhook_only:
            return await self._send_via_webhook(content, msg_type)

        # 构建 content 结构
        built_content = _build_content(content, msg_type)

        data = await self._request(
            "POST",
            "/open-apis/im/v1/messages",
            params={"receive_id_type": receive_id_type},
            json_body={
                "receive_id": receive_id,
                "msg_type": msg_type,
                "content": built_content,
            },
        )
        return _parse_message(data["data"])

    async def _send_via_webhook(self, content: str, msg_type: str) -> IMMessage:
        """通过 Webhook 发送消息（仅支持 text 和 interactive）。"""
        if msg_type == "interactive":
            payload = {"msg_type": "interactive", "card": json.loads(content)}
        else:
            payload = {"msg_type": "text", "content": {"text": content}}

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(self._webhook_url, json=payload)
        data = resp.json()
        if data.get("StatusCode", data.get("code", -1)) not in (0, 200):
            raise IMError(
                f"Webhook 发送失败：{data.get('StatusMessage', data.get('msg', ''))}",
                platform="feishu",
            )
        # Webhook 不返回消息 ID，返回占位对象
        return IMMessage(id="", chat_id="", sender_id="bot", content=content,
                         msg_type=msg_type, create_time=str(int(time.time() * 1000)))

    async def reply_message(
        self,
        message_id: str,
        content: str,
        msg_type: str = "text",
    ) -> IMMessage:
        """回复指定消息。"""
        built_content = _build_content(content, msg_type)
        data = await self._request(
            "POST",
            f"/open-apis/im/v1/messages/{message_id}/reply",
            json_body={"msg_type": msg_type, "content": built_content},
        )
        return _parse_message(data["data"])

    async def get_message(self, message_id: str) -> IMMessage:
        """获取单条消息。"""
        data = await self._request("GET", f"/open-apis/im/v1/messages/{message_id}")
        items = data.get("data", {}).get("items", [])
        if not items:
            raise IMNotFoundError(f"消息不存在：{message_id}", platform="feishu")
        return _parse_message(items[0])

    async def list_messages(
        self,
        container_id: str,
        container_id_type: str = "chat_id",
        page_size: int = 20,
        start_time: str = "",
        end_time: str = "",
    ) -> list[IMMessage]:
        """列出会话消息历史。"""
        params: dict[str, Any] = {
            "container_id_type": container_id_type,
            "container_id": container_id,
            "page_size": min(page_size, 50),
            "sort_type": "ByCreateTimeDesc",
        }
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        data = await self._request("GET", "/open-apis/im/v1/messages", params=params)
        items = data.get("data", {}).get("items", []) or []
        return [_parse_message(item) for item in items]

    # -----------------------------------------------------------------------
    # 群组/会话
    # -----------------------------------------------------------------------

    async def get_chat(self, chat_id: str) -> IMChat:
        """获取群组信息。"""
        data = await self._request("GET", f"/open-apis/im/v1/chats/{chat_id}")
        return _parse_chat(data["data"])

    async def search_chats(
        self,
        query: str,
        page_size: int = 20,
    ) -> list[IMChat]:
        """按名称搜索机器人所在的群组。"""
        params = {"search_key": query, "page_size": min(page_size, 100)}
        data = await self._request(
            "GET", "/open-apis/im/v1/chats", params=params
        )
        items = data.get("data", {}).get("items", []) or []
        return [_parse_chat(item) for item in items]

    async def list_chat_members(
        self,
        chat_id: str,
        page_size: int = 50,
    ) -> list[IMUser]:
        """列出群组成员。"""
        params = {"page_size": min(page_size, 100)}
        data = await self._request("GET", f"/open-apis/im/v1/chats/{chat_id}/members", params=params)
        members = data.get("data", {}).get("items", []) or []
        return [
            IMUser(
                id=m.get("member_id", ""),
                name=m.get("name", ""),
                open_id=m.get("member_id", "") if m.get("member_id_type") == "open_id" else "",
                raw=m,
            )
            for m in members
        ]

    # -----------------------------------------------------------------------
    # 用户
    # -----------------------------------------------------------------------

    async def get_user(
        self,
        user_id: str,
        id_type: str = "open_id",
    ) -> IMUser:
        """获取用户信息。"""
        params = {"user_id_type": id_type}
        data = await self._request("GET", f"/open-apis/contact/v3/users/{user_id}", params=params)
        return _parse_user(data.get("data", {}).get("user", {}))

    async def search_users(
        self,
        query: str,
        page_size: int = 10,
    ) -> list[IMUser]:
        """搜索用户（需要 contact:user.employee_id 等权限）。"""
        params = {"query": query, "page_size": min(page_size, 50)}
        data = await self._request("GET", "/open-apis/contact/v3/users/search", params=params)
        users = data.get("data", {}).get("users", []) or []
        return [_parse_user(u) for u in users]

    # -----------------------------------------------------------------------
    # 文件/资源
    # -----------------------------------------------------------------------

    async def upload_image(self, file_path: str) -> str:
        """上传图片，返回 image_key。"""
        path = Path(file_path)
        with open(path, "rb") as f:
            files = {
                "image_type": (None, "message"),
                "image": (path.name, f, "application/octet-stream"),
            }
            data = await self._request("POST", "/open-apis/im/v1/images", files=files)
        return data["data"]["image_key"]

    async def upload_file(
        self,
        file_path: str,
        file_type: str = "stream",
    ) -> str:
        """上传文件，返回 file_key。file_type: opus|mp4|pdf|doc|xls|ppt|stream"""
        path = Path(file_path)
        with open(path, "rb") as f:
            files = {
                "file_type": (None, file_type),
                "file_name": (None, path.name),
                "file": (path.name, f, "application/octet-stream"),
            }
            data = await self._request("POST", "/open-apis/im/v1/files", files=files)
        return data["data"]["file_key"]

    async def download_file(
        self,
        file_key: str,
        save_path: str,
    ) -> str:
        """下载文件资源（图片或文件），保存到 save_path。"""
        # 先尝试 images 端点，若失败再尝试 resources 端点
        save_path = str(Path(save_path))
        try:
            await self._request(
                "GET",
                f"/open-apis/im/v1/images/{file_key}",
                stream_to=save_path,
            )
        except IMError:
            await self._request(
                "GET",
                f"/open-apis/im/v1/resources/{file_key}/content",
                params={"type": "file"},
                stream_to=save_path,
            )
        return save_path


# ---------------------------------------------------------------------------
# 解析工具函数
# ---------------------------------------------------------------------------

def _build_content(content: str, msg_type: str) -> str:
    """将用户输入内容包装为飞书 API 需要的 JSON 字符串。"""
    if msg_type == "text":
        return json.dumps({"text": content}, ensure_ascii=False)
    if msg_type == "interactive":
        # 用户直接传卡片 JSON 字符串或 dict
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        return content
    # rich_text / image / file 等类型用户自行传完整 JSON
    return content


def _parse_message(raw: dict) -> IMMessage:
    body = raw.get("body", {})
    sender = raw.get("sender", {})
    content = body.get("content", "")
    # 尝试解析 text 内容
    try:
        parsed = json.loads(content)
        content = parsed.get("text", content)
    except (json.JSONDecodeError, AttributeError):
        pass

    return IMMessage(
        id=raw.get("message_id", ""),
        chat_id=raw.get("chat_id", ""),
        sender_id=sender.get("id", ""),
        content=content,
        msg_type=raw.get("msg_type", "text"),
        create_time=raw.get("create_time", ""),
        root_id=raw.get("root_id", ""),
        parent_id=raw.get("parent_id", ""),
        raw=raw,
    )


def _parse_chat(raw: dict) -> IMChat:
    return IMChat(
        id=raw.get("chat_id", ""),
        name=raw.get("name", ""),
        chat_type=raw.get("chat_type", ""),
        description=raw.get("description", ""),
        member_count=raw.get("member_count", 0),
        owner_id=raw.get("owner_id", ""),
        raw=raw,
    )


def _parse_user(raw: dict) -> IMUser:
    return IMUser(
        id=raw.get("user_id", raw.get("open_id", "")),
        name=raw.get("name", ""),
        open_id=raw.get("open_id", ""),
        union_id=raw.get("union_id", ""),
        email=raw.get("enterprise_email", raw.get("email", "")),
        mobile=raw.get("mobile", ""),
        department=raw.get("department_ids", [""])[0] if raw.get("department_ids") else "",
        raw=raw,
    )
