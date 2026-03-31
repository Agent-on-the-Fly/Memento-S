"""IM 平台操作入口脚本（Playbook entry_script）。

Agent 通过 execute_skill 调用时指定 args 参数：
    args: ["<operation>", "<arg1>", "<arg2>", ...]

支持的操作：
    send_message     <receive_id> <text> [receive_id_type]
    send_to_chat     <chat_name> <text>
    get_user_info    <user_id> [id_type]
    search_users     <query> [count]
    search_chats     <query> [count]
    get_chat_info    <chat_id>
    list_members     <chat_id> [count]
    get_messages     <chat_id> [count]
    get_message      <message_id>
    reply            <message_id> <text>

示例（Agent 调用）：
    execute_skill("im-platform", "发消息给技术群", args=["send_to_chat", "技术交流群", "大家好"])
    execute_skill("im-platform", "搜索用户", args=["search_users", "张三"])
"""
from __future__ import annotations

import json
import os
import sys
import asyncio
# 确保 scripts/ 目录在 PYTHONPATH（playbook 从 output_dir 运行）
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def _out(data) -> None:
    print(json.dumps(data, ensure_ascii=False, default=str))


def main() -> None:
    args = sys.argv[1:]

    if not args:
        _out({
            "error": "未指定操作",
            "operations": [
                "send_message <receive_id> <text> [receive_id_type]",
                "send_to_chat <chat_name> <text>",
                "get_user_info <user_id> [id_type]",
                "search_users <query> [count]",
                "search_chats <query> [count]",
                "get_chat_info <chat_id>",
                "list_members <chat_id> [count]",
                "get_messages <chat_id> [count]",
                "get_message <message_id>",
                "reply <message_id> <text>",
            ],
        })
        sys.exit(1)

    op = args[0]
    params = args[1:]

    try:
        if op == "send_message":
            from messaging import send_text_message
            receive_id = params[0]
            text = params[1]
            id_type = params[2] if len(params) > 2 else "open_id"
            result = asyncio.run(
                send_text_message(receive_id, text, receive_id_type=id_type)
            )
            _out(result)

        elif op == "send_to_chat":
            from messaging import send_to_chat_by_name
            chat_name = params[0]
            text = params[1]
            result = asyncio.run(send_to_chat_by_name(chat_name, text))
            _out(result)

        elif op == "get_user_info":
            from users import get_user_info
            user_id = params[0]
            id_type = params[1] if len(params) > 1 else "open_id"
            result = asyncio.run(get_user_info(user_id, id_type=id_type))
            _out(result)

        elif op == "search_users":
            from users import search_users
            query = params[0]
            count = int(params[1]) if len(params) > 1 else 10
            result = asyncio.run(search_users(query, count=count))
            _out(result)

        elif op == "search_chats":
            from users import search_chats
            query = params[0]
            count = int(params[1]) if len(params) > 1 else 20
            result = asyncio.run(search_chats(query, count=count))
            _out(result)

        elif op == "get_chat_info":
            from users import get_chat_info
            result = asyncio.run(get_chat_info(params[0]))
            _out(result)

        elif op == "list_members":
            from users import list_group_members
            chat_id = params[0]
            count = int(params[1]) if len(params) > 1 else 50
            result = asyncio.run(list_group_members(chat_id, count=count))
            _out(result)

        elif op == "get_messages":
            from messaging import get_recent_messages
            chat_id = params[0]
            count = int(params[1]) if len(params) > 1 else 20
            result = asyncio.run(get_recent_messages(chat_id, count=count))
            _out(result)

        elif op == "get_message":
            from messaging import get_message
            result = asyncio.run(get_message(params[0]))
            _out(result)

        elif op == "reply":
            from messaging import reply_to_message
            result = asyncio.run(reply_to_message(params[0], params[1]))
            _out(result)

        else:
            _out({"error": f"未知操作: {op}"})
            sys.exit(1)

    except IndexError:
        _out({"error": f"操作 '{op}' 缺少必要参数，请检查 args"})
        sys.exit(1)
    except Exception as e:
        _out({"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
