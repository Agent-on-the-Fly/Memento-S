"""
用户与群组操作封装函数。

提供对 IMPlatform 用户/群组相关方法的高层封装，返回可序列化的 dict。

示例：
    from scripts.users import get_user_info, list_group_members
    import asyncio

    user = asyncio.run(get_user_info("ou_xxx"))
    members = asyncio.run(list_group_members("oc_xxx"))
"""
from __future__ import annotations

from .base import IMChat, IMUser
from .factory import get_platform


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _user_to_dict(user: IMUser) -> dict:
    return {
        "id": user.id,
        "name": user.name,
        "open_id": user.open_id,
        "union_id": user.union_id,
        "email": user.email,
        "mobile": user.mobile,
        "department": user.department,
    }


def _chat_to_dict(chat: IMChat) -> dict:
    return {
        "id": chat.id,
        "name": chat.name,
        "chat_type": chat.chat_type,
        "description": chat.description,
        "member_count": chat.member_count,
        "owner_id": chat.owner_id,
    }


# ---------------------------------------------------------------------------
# 用户操作
# ---------------------------------------------------------------------------

async def get_user_info(
    user_id: str,
    id_type: str = "open_id",
    platform: str | None = None,
) -> dict:
    """
    获取用户详细信息。

    Args:
        user_id: 用户 ID
        id_type: ID 类型，open_id | user_id | union_id | email | mobile
        platform: 指定平台（None 时从 IM_PLATFORM 环境变量读取）

    Returns:
        用户信息 dict，含 id、name、open_id、union_id、email、mobile、department

    示例：
        user = await get_user_info("ou_xxx")
        user = await get_user_info("user@company.com", id_type="email")
    """
    p = get_platform(platform)
    user = await p.get_user(user_id, id_type=id_type)
    return _user_to_dict(user)


async def search_users(
    query: str,
    count: int = 10,
    platform: str | None = None,
) -> list[dict]:
    """
    按名称或邮箱搜索用户。

    Args:
        query: 搜索关键词（姓名、邮箱等）
        count: 返回结果数量（最大 50）
        platform: 指定平台

    Returns:
        用户信息列表

    示例：
        users = await search_users("张三")
        users = await search_users("zhang@company.com")
    """
    p = get_platform(platform)
    users = await p.search_users(query, page_size=min(count, 50))
    return [_user_to_dict(u) for u in users]


# ---------------------------------------------------------------------------
# 群组操作
# ---------------------------------------------------------------------------

async def search_chats(
    query: str,
    count: int = 20,
    platform: str | None = None,
) -> list[dict]:
    """
    按名称搜索机器人所在的群组。

    Args:
        query: 群名称关键词
        count: 返回结果数量（最大 100）
        platform: 指定平台

    Returns:
        群组信息列表，每条含 id、name、chat_type、member_count 等字段

    示例：
        chats = await search_chats("技术交流")
        for c in chats:
            print(f"{c['name']} - {c['id']}")
    """
    p = get_platform(platform)
    chats = await p.search_chats(query, page_size=min(count, 100))
    return [_chat_to_dict(c) for c in chats]


async def list_group_members(
    chat_id: str,
    count: int = 50,
    platform: str | None = None,
) -> list[dict]:
    """
    列出群组成员。

    Args:
        chat_id: 群组 ID（oc_xxx 格式）
        count: 获取成员数量（最大 100）
        platform: 指定平台

    Returns:
        成员用户信息列表，每条含 id、name、open_id

    示例：
        members = await list_group_members("oc_xxx")
        for m in members:
            print(m["name"])
    """
    p = get_platform(platform)
    users = await p.list_chat_members(chat_id, page_size=min(count, 100))
    return [_user_to_dict(u) for u in users]
