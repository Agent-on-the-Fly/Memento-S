# SPDX-License-Identifier: Apache-2.0
"""
飞书机器人 × Agent 桥接脚本（带 DB 持久化）。

飞书用户发消息 → Agent 处理 → 飞书回复
每个用户拥有独立的 DB Session，对话历史跨重启保留。

用法：
    cd /path/to/opc_memento_s
    python daemon/im_platform/feishu/bridge.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from shared.chat import ChatManager
from core.memento_s.agent import MementoSAgent
from middleware.config import g_config
from messaging import send_text_message
from feishu.receiver import FeishuReceiver
from server.schema.uni_response import UniResponse

# 将项目根目录和 scripts/ 目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

# 进程内缓存：feishu sender_id → DB session_id
_sender_to_session: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# 映射文件（workspace/feishu_sessions.json）                                   #
# --------------------------------------------------------------------------- #


def _mapping_path() -> Path:
    workspace = Path(g_config.paths.workspace_dir).expanduser().resolve()
    return workspace / "feishu_sessions.json"


def _load_mapping() -> dict[str, str]:
    p = _mapping_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_mapping(mapping: dict[str, str]) -> None:
    p = _mapping_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Session 管理                                                                  #
# --------------------------------------------------------------------------- #


async def get_or_create_session(sender_id: str) -> str:
    """获取或创建飞书用户对应的 DB Session，返回 DB session_id。"""
    if sender_id in _sender_to_session:
        db_sid = _sender_to_session[sender_id]
        if await ChatManager.exists(db_sid):
            return db_sid
        del _sender_to_session[sender_id]

    session = await ChatManager.create_session(
        title=f"飞书: {sender_id}",
        metadata={"feishu_sender_id": sender_id, "source": "feishu"},
    )
    db_sid = session.id
    _sender_to_session[sender_id] = db_sid
    _save_mapping(_sender_to_session)
    print(f"[Session] 为用户 {sender_id} 创建新会话: {db_sid}")
    return db_sid


# --------------------------------------------------------------------------- #
# 消息处理                                                                      #
# --------------------------------------------------------------------------- #


def build_agent() -> MementoSAgent:
    return MementoSAgent()


async def handle_message(
    msg: dict,
    agent: MementoSAgent,
) -> None:
    """收到飞书消息后，交给 Agent 处理并回复，同时将对话写入 DB。"""
    sender_id = msg["sender_id"]
    content = msg["content"].strip()
    if not content:
        return

    print(f"\n[飞书→Agent] {sender_id}: {content}")

    session_id = await get_or_create_session(sender_id)

    user_title = content[:50] + "..." if len(content) > 50 else content
    # 构建用户消息事件（用于前端展示）
    user_event = {
        "type": "USER_INPUT",
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "conversation_id": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
        "role": "user",
        "content": content,
        "event_id": f"msg_{uuid.uuid4().hex[:12]}",
        "payload": {"messages": "", "content": content},
    }
    user_content_detail = UniResponse.from_event(user_event).model_dump(mode="json", exclude_none=True)

    user_conv = await ChatManager.create_conversation(
        session_id=session_id,
        role="user",
        title=user_title,
        content=content,
        content_detail=user_content_detail,
        meta_info={
            "channel": "feishu",
            "sender_id": sender_id,
            "im_message_id": msg.get("message_id", ""),
            "chat_id": msg.get("chat_id", ""),
            "chat_type": msg.get("chat_type", ""),
            "msg_type": msg.get("msg_type", "text"),
            "media_urls": msg.get("media_urls", []),
            "raw_metadata": msg,
        },
    )

    final_text = ""
    conversation_id = user_content_detail.get("conversation_id", "")
    _text_start_event: dict | None = None
    _text_buf: list[str] = []
    _text_run_id: str | None = None

    async def _save_event(event: dict, role: str, title: str, text_content: str, tokens: int = 0) -> None:
        """Save a single conversation event to DB."""
        content_detail = UniResponse.from_event(event).model_dump(mode="json", exclude_none=True)
        await ChatManager.create_conversation(
            session_id=session_id,
            role=role,
            title=title,
            content=text_content,
            content_detail=content_detail,
            meta_info={
                "reply_to": user_conv.id,
                "channel": "feishu",
                "sender_id": sender_id,
                "chat_id": msg.get("chat_id", ""),
            },
            tokens=tokens,
        )

    async for event in agent.reply_stream(session_id=session_id, user_content=content):
        event_type = event.get("type")

        # 管理 run_id
        if event_type == "TEXT_MESSAGE_START":
            _text_run_id = str(uuid.uuid4())
        elif event_type in ("TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END"):
            pass  # 复用同一个 _text_run_id
        else:
            _text_run_id = str(uuid.uuid4())

        # 更新事件中的 run_id 和 conversation_id
        if _text_run_id:
            event["run_id"] = _text_run_id
        event["conversation_id"] = conversation_id

        # TEXT_MESSAGE 三段式：累积后在 END 时统一持久化
        if event_type == "TEXT_MESSAGE_START":
            _text_start_event = event
            _text_buf = []
            print(event.get("delta", ""), end="", flush=True)
            continue

        if event_type == "TEXT_MESSAGE_CONTENT":
            delta = event.get("delta", "")
            _text_buf.append(delta)
            print(delta, end="", flush=True)
            continue

        if event_type == "TEXT_MESSAGE_END":
            full_text = "".join(_text_buf)

            # 持久化 START
            if _text_start_event:
                await _save_event(
                    _text_start_event,
                    role="assistant",
                    title="",
                    text_content="",
                )

            # 持久化 CONTENT（payload.content 放完整文本）
            content_event = {
                **event,
                "type": "TEXT_MESSAGE_CONTENT",
                "delta": full_text,
                "payload": {"messages": "", "content": full_text},
            }
            await _save_event(
                content_event,
                role="assistant",
                title=full_text[:50] + "..." if len(full_text) > 50 else full_text,
                text_content=full_text,
            )

            # 持久化 END
            await _save_event(
                event,
                role="assistant",
                title="",
                text_content="",
            )

            final_text = full_text
            _text_start_event = None
            _text_buf = []
            continue

        # 其他事件（TOOL_CALL_START 等）
        if event_type == "TOOL_CALL_START":
            print(f"\n  [调用工具: {event.get('toolName', '')}]", end="", flush=True)

        # RUN_FINISHED / RUN_ERROR
        if event_type in ("RUN_FINISHED", "RUN_ERROR"):
            output_text = event.get("outputText") or ""
            if not final_text and output_text:
                final_text = output_text
            # RUN_FINISHED 的 payload.messages 放完整文本，content 为空
            run_event = {
                **event,
                "payload": {"messages": output_text, "content": ""},
            }
            await _save_event(
                run_event,
                role="assistant",
                title=output_text[:50] + "..." if len(output_text) > 50 else output_text if output_text else "RUN_FINISHED",
                text_content=output_text,
            )

        if event_type == "RUN_ERROR":
            final_text = f"处理出错：{event.get('message', '')}"

    print()

    if final_text:
        print(f"[Agent→飞书] 回复：{final_text[:80]}...")
        await send_text_message(sender_id, final_text)


# --------------------------------------------------------------------------- #
# 入口                                                                          #
# --------------------------------------------------------------------------- #


def main() -> None:
    global _sender_to_session
    _sender_to_session = _load_mapping()

    agent = build_agent()

    print(f"Agent 初始化完成，已加载 {len(_sender_to_session)} 个飞书会话映射")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def on_message(msg: dict) -> None:
        future = asyncio.run_coroutine_threadsafe(
            handle_message(msg, agent),
            loop,
        )
        future.result()

    receiver = FeishuReceiver(on_message=on_message)
    receiver.start_in_background()

    print("飞书长链接已在后台启动，等待消息... (Ctrl+C 退出)")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
