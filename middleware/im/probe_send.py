"""
IM 平台连通性探针。

直接运行，向指定目标发送"你好"测试消息，验证凭证和 API 是否正常。

用法：
    # 飞书：发给用户 open_id
    python middleware/im/probe_send.py feishu ou_xxxxxxxxxx

    # 飞书：发给群 chat_id
    python middleware/im/probe_send.py feishu oc_xxxxxxxxxx chat_id

    # 钉钉：发给群会话 openConversationId
    python middleware/im/probe_send.py dingtalk cid_xxxxxxxxxx openConversationId

    # 钉钉：发给员工 staffId
    python middleware/im/probe_send.py dingtalk staff_xxxxxx staffId

    # 企微：发给用户 userid（应用模式）
    python middleware/im/probe_send.py wecom zhangsan touser

    # 企微：发给应用群 chatid
    python middleware/im/probe_send.py wecom wrChatId_xxxx chatid

不传 receive_id 时，只检查配置是否加载成功（不发消息）。
"""
from __future__ import annotations

import asyncio
import json
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_CONFIG_PATH = os.path.expanduser("~/memento_s/config.json")


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"[ERROR] config.json 格式错误：{e}")
        return {}


def _show_config_status(cfg: dict):
    """打印各平台配置检查结果。"""
    im = cfg.get("im", {})
    platforms = {
        "feishu":   ["app_id", "app_secret"],
        "dingtalk": ["app_key", "app_secret"],
        "wecom":    ["corp_id", "secret", "agent_id"],
    }
    print("\n── 配置状态 ──────────────────────────────")
    for name, keys in platforms.items():
        section = im.get(name, {})
        filled = [k for k in keys if section.get(k)]
        missing = [k for k in keys if not section.get(k)]

        # Webhook 也算有效
        webhook = section.get("webhook_url", "")
        bot_mode = name == "wecom" and section.get("bot_id")

        if filled == keys:
            status = "✓ 完整模式"
        elif webhook:
            status = "~ Webhook 模式（仅发送）"
        elif bot_mode:
            status = "~ 智能机器人模式"
        elif filled:
            status = f"✗ 缺少 {missing}"
        else:
            status = "✗ 未配置"

        print(f"  {name:10s}  {status}")
    print("──────────────────────────────────────────\n")


async def _probe(platform: str, receive_id: str, receive_id_type: str):
    """实际发送探针消息。"""
    from middleware.im.im_platform.factory import get_platform

    print(f"[{platform}] 初始化平台...")
    p = get_platform(platform)
    print(f"[{platform}] 平台实例已创建")

    msg_text = "你好！这是来自 probe_send.py 的连通性测试消息 👋"
    print(f"[{platform}] 发送到 {receive_id_type}={receive_id!r} ...")

    result = await p.send_message(
        receive_id=receive_id,
        content=msg_text,
        msg_type="text",
        receive_id_type=receive_id_type,
    )

    print(f"[{platform}] ✓ 发送成功！")
    print(f"  message_id : {result.id}")
    print(f"  chat_id    : {result.chat_id}")
    print(f"  create_time: {result.create_time}")


def main():
    args = sys.argv[1:]
    cfg = _load_config()
    _show_config_status(cfg)

    if not args:
        print("用法：python probe_send.py <platform> <receive_id> [receive_id_type]")
        print("示例：python probe_send.py feishu ou_xxxx")
        print("      python probe_send.py dingtalk cid_xxxx openConversationId")
        print("      python probe_send.py wecom zhangsan touser")
        return

    platform = args[0].lower()
    if len(args) < 2:
        print("[INFO] 只传了 platform，跳过发送测试（配置检查已完成）")
        return

    receive_id = args[1]

    # 默认 receive_id_type
    defaults = {
        "feishu":   "open_id",
        "dingtalk": "staffId",
        "wecom":    "touser",
    }
    receive_id_type = args[2] if len(args) > 2 else defaults.get(platform, "open_id")

    try:
        asyncio.run(_probe(platform, receive_id, receive_id_type))
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
