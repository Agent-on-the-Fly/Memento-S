"""
IM 平台抽象协议与公共数据模型。

新平台适配器只需实现 IMPlatform 协议中定义的方法，即可接入统一工厂。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class IMMessage:
    """统一消息模型。"""
    id: str
    chat_id: str
    sender_id: str
    content: str                    # 纯文本摘要或原始 content JSON
    msg_type: str                   # text | rich_text | image | file | interactive
    create_time: str                # 毫秒时间戳字符串
    root_id: str = ""               # 话题根消息 ID（可选）
    parent_id: str = ""             # 回复目标消息 ID（可选）
    raw: dict = field(default_factory=dict)   # 原始平台响应（调试用）


@dataclass
class IMUser:
    """统一用户模型。"""
    id: str                         # 平台内部用户 ID
    name: str
    open_id: str = ""               # 飞书 open_id / Slack user_id 等
    union_id: str = ""              # 跨应用唯一 ID（飞书 union_id）
    email: str = ""
    mobile: str = ""
    department: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class IMChat:
    """统一群组/会话模型。"""
    id: str
    name: str
    chat_type: str                  # p2p | group
    description: str = ""
    member_count: int = 0
    owner_id: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class IMFile:
    """统一文件/资源模型。"""
    key: str                        # 平台文件 key
    name: str = ""
    file_type: str = ""             # image | video | audio | pdf | doc | ...
    size: int = 0
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 平台协议
# ---------------------------------------------------------------------------

@runtime_checkable
class IMPlatform(Protocol):
    """
    IM 平台统一接口。

    各平台适配器实现此协议，通过 factory.get_platform() 获取实例。
    Webhook 模式的平台仅需实现 send_message；完整 API 模式需实现所有方法。
    """

    # ---- 消息 ----

    async def send_message(
        self,
        receive_id: str,
        content: str,
        msg_type: str = "text",
        receive_id_type: str = "open_id",
    ) -> IMMessage:
        """
        发送消息。

        Args:
            receive_id: 接收方 ID（用户 open_id / chat_id / user_id 等）
            content: 消息内容。text 类型传纯文本；interactive 类型传 card JSON 字符串
            msg_type: 消息类型，text | rich_text | image | file | interactive
            receive_id_type: ID 类型，open_id | user_id | union_id | email | chat_id
        """
        ...

    async def reply_message(
        self,
        message_id: str,
        content: str,
        msg_type: str = "text",
    ) -> IMMessage:
        """回复指定消息。"""
        ...

    async def get_message(self, message_id: str) -> IMMessage:
        """获取单条消息详情。"""
        ...

    async def list_messages(
        self,
        container_id: str,
        container_id_type: str = "chat_id",
        page_size: int = 20,
        start_time: str = "",
        end_time: str = "",
    ) -> list[IMMessage]:
        """
        列出会话消息历史。

        Args:
            container_id: chat_id 或 thread_id
            container_id_type: chat_id | thread_id
            page_size: 每页数量（最大 50）
            start_time: 开始时间戳（秒，可选）
            end_time: 结束时间戳（秒，可选）
        """
        ...

    # ---- 群组/会话 ----

    async def get_chat(self, chat_id: str) -> IMChat:
        """获取群组/会话信息。"""
        ...

    async def list_chat_members(
        self,
        chat_id: str,
        page_size: int = 50,
    ) -> list[IMUser]:
        """列出群组成员。"""
        ...

    # ---- 用户 ----

    async def get_user(
        self,
        user_id: str,
        id_type: str = "open_id",
    ) -> IMUser:
        """
        获取用户信息。

        Args:
            user_id: 用户 ID
            id_type: ID 类型，open_id | user_id | union_id | email | mobile
        """
        ...

    async def search_users(
        self,
        query: str,
        page_size: int = 10,
    ) -> list[IMUser]:
        """按名称/邮箱搜索用户。"""
        ...

    # ---- 文件/资源 ----

    async def upload_image(self, file_path: str) -> str:
        """
        上传图片，返回 image_key。

        Args:
            file_path: 本地图片文件路径
        """
        ...

    async def upload_file(
        self,
        file_path: str,
        file_type: str = "stream",
    ) -> str:
        """
        上传文件，返回 file_key。

        Args:
            file_path: 本地文件路径
            file_type: 飞书文件类型，opus | mp4 | pdf | doc | xls | ppt | stream
        """
        ...

    async def download_file(
        self,
        file_key: str,
        save_path: str,
    ) -> str:
        """
        下载文件资源到本地。

        Args:
            file_key: 平台文件 key（image_key 或 file_key）
            save_path: 本地保存路径

        Returns:
            实际保存路径
        """
        ...


# ---------------------------------------------------------------------------
# 异常类型
# ---------------------------------------------------------------------------

class IMError(Exception):
    """IM 平台操作异常基类。"""
    def __init__(self, message: str, code: int = 0, platform: str = ""):
        super().__init__(message)
        self.code = code
        self.platform = platform


class IMAuthError(IMError):
    """认证/授权失败（Token 无效、权限不足等）。"""


class IMNotFoundError(IMError):
    """资源不存在（消息、用户、群组等）。"""


class IMRateLimitError(IMError):
    """请求频率超限。"""


class IMWebhookOnlyError(IMError):
    """仅 Webhook 模式时调用了需要 App 凭证的 API。"""
