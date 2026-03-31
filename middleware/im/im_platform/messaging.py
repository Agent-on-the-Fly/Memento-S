"""
消息操作封装函数。

提供对 IMPlatform.send_message / reply_message / list_messages 的高层封装，
LLM 生成的代码可直接调用这些函数，无需关心平台实例创建。

所有函数均为 async，返回可序列化的 dict 或 list[dict]，
便于 LLM 直接 print 查看结果。

示例：
    from scripts.messaging import send_text_message, get_recent_messages
    import asyncio

    result = asyncio.run(send_text_message("ou_xxx", "你好！"))
    print(result)
"""
from __future__ import annotations
import json
from .base import IMMessage
from .factory import get_platform


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _message_to_dict(msg: IMMessage) -> dict:
    return {
        "id": msg.id,
        "chat_id": msg.chat_id,
        "sender_id": msg.sender_id,
        "content": msg.content,
        "msg_type": msg.msg_type,
        "create_time": msg.create_time,
        "root_id": msg.root_id,
        "parent_id": msg.parent_id,
    }


# ---------------------------------------------------------------------------
# 发送消息
# ---------------------------------------------------------------------------

async def send_text_message(
    receive_id: str,
    text: str,
    receive_id_type: str = "open_id",
    platform: str | None = None,
) -> dict:
    """
    发送纯文本消息。

    Args:
        receive_id: 接收方 ID（用户 open_id / chat_id 等）
        text: 消息文本内容
        receive_id_type: ID 类型，open_id | user_id | union_id | email | chat_id
        platform: 指定平台（None 时从 IM_PLATFORM 环境变量读取）

    Returns:
        消息详情 dict，含 id、chat_id、sender_id、content、create_time 等字段

    示例：
        result = await send_text_message("ou_xxx", "你好！")
        result = await send_text_message("oc_xxx", "群消息", receive_id_type="chat_id")
    """
    p = get_platform(platform)
    msg = await p.send_message(receive_id, text, msg_type="text", receive_id_type=receive_id_type)
    return _message_to_dict(msg)


async def send_card_message(
    receive_id: str,
    card: dict | str,
    receive_id_type: str = "open_id",
    platform: str | None = None,
) -> dict:
    """
    发送飞书交互卡片（Interactive Card）。

    Args:
        receive_id: 接收方 ID
        card: 卡片 JSON（dict 或 JSON 字符串），遵循飞书卡片 schema
        receive_id_type: ID 类型
        platform: 指定平台

    Returns:
        消息详情 dict

    示例：
        card = {
            "config": {"wide_screen_mode": True},
            "elements": [{"tag": "div", "text": {"content": "Hello", "tag": "lark_md"}}]
        }
        result = await send_card_message("ou_xxx", card)
    """
    p = get_platform(platform)
    content = json.dumps(card, ensure_ascii=False) if isinstance(card, dict) else card
    msg = await p.send_message(receive_id, content, msg_type="interactive", receive_id_type=receive_id_type)
    return _message_to_dict(msg)


async def send_to_chat_by_name(
    chat_name: str,
    text: str,
    platform: str | None = None,
) -> dict:
    """
    按群名称发送消息（自动搜索 chat_id）。

    先搜索与 chat_name 匹配的群，若找到唯一精确匹配则直接发送；
    若有多个匹配，优先选择名称完全相同的群；
    若无匹配则返回错误信息。

    Args:
        chat_name: 群名称（支持模糊匹配，但优先精确匹配）
        text: 消息文本内容
        platform: 指定平台

    Returns:
        成功时返回消息详情 dict；失败时返回含 error 和 candidates 字段的 dict

    示例：
        result = await send_to_chat_by_name("技术交流群", "大家好！")
    """
    from users import search_chats

    chats = await search_chats(chat_name, platform=platform)
    if not chats:
        return {"error": f"未找到名称包含「{chat_name}」的群组，请确认机器人已加入该群"}

    # 优先精确匹配
    exact = [c for c in chats if c["name"] == chat_name]
    target = exact[0] if exact else chats[0]

    if not exact and len(chats) > 1:
        return {
            "error": f"找到 {len(chats)} 个匹配群组，请提供更精确的群名",
            "candidates": [{"name": c["name"], "id": c["id"]} for c in chats],
        }

    result = await send_text_message(
        target["id"], text, receive_id_type="chat_id", platform=platform
    )
    result["chat_name"] = target["name"]
    return result


async def reply_to_message(
    message_id: str,
    text: str,
    msg_type: str = "text",
    platform: str | None = None,
) -> dict:
    """
    回复指定消息。

    Args:
        message_id: 被回复的消息 ID（om_xxx 格式）
        text: 回复内容（text 类型传文本，interactive 类型传卡片 JSON）
        msg_type: 消息类型，text | interactive
        platform: 指定平台

    Returns:
        新消息详情 dict

    示例：
        result = await reply_to_message("om_xxx", "收到！")
    """
    p = get_platform(platform)
    msg = await p.reply_message(message_id, text, msg_type=msg_type)
    return _message_to_dict(msg)


# ---------------------------------------------------------------------------
# 查询消息
# ---------------------------------------------------------------------------

async def get_message(
    message_id: str,
    platform: str | None = None,
) -> dict:
    """
    获取单条消息详情。

    Args:
        message_id: 消息 ID（om_xxx 格式）
        platform: 指定平台

    Returns:
        消息详情 dict

    示例：
        msg = await get_message("om_xxx")
        print(msg["content"])
    """
    p = get_platform(platform)
    msg = await p.get_message(message_id)
    return _message_to_dict(msg)


async def get_recent_messages(
    chat_id: str,
    count: int = 20,
    start_time: str = "",
    end_time: str = "",
    platform: str | None = None,
) -> list[dict]:
    """
    获取群组/会话的最近消息列表。

    Args:
        chat_id: 群组 chat_id（oc_xxx 格式）
        count: 获取消息数量（最大 50）
        start_time: 开始时间戳（秒，可选）
        end_time: 结束时间戳（秒，可选）
        platform: 指定平台

    Returns:
        消息列表（按时间倒序），每条消息含 id、sender_id、content、create_time 等字段

    示例：
        msgs = await get_recent_messages("oc_xxx", count=10)
        for m in msgs:
            print(f"{m['sender_id']}: {m['content']}")
    """
    p = get_platform(platform)
    messages = await p.list_messages(
        container_id=chat_id,
        container_id_type="chat_id",
        page_size=min(count, 50),
        start_time=start_time,
        end_time=end_time,
    )
    return [_message_to_dict(m) for m in messages]


async def get_thread_messages(
    thread_id: str,
    count: int = 20,
    platform: str | None = None,
) -> list[dict]:
    """
    获取话题（Thread）中的消息列表。

    Args:
        thread_id: 话题 ID
        count: 获取消息数量（最大 50）
        platform: 指定平台

    Returns:
        消息列表（按时间倒序）
    """
    p = get_platform(platform)
    messages = await p.list_messages(
        container_id=thread_id,
        container_id_type="thread_id",
        page_size=min(count, 50),
    )
    return [_message_to_dict(m) for m in messages]
