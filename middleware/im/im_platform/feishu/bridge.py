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
from pathlib import Path
from shared.chat import ChatManager
from core.memento_s.agent import MementoSAgent
from core.memento_s.stream_output import (
    AGUIEventPipeline,
    AGUIEventType,
    PersistenceSink,
)
from middleware.config import g_config
from messaging import send_text_message
from feishu.receiver import FeishuReceiver

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
    user_conv = await ChatManager.create_conversation(
        session_id=session_id,
        role="user",
        title=user_title,
        content=content,
        meta_info={},
    )

    final_text = ""

    async def _persist_reply(text: str) -> None:
        nonlocal final_text
        final_text = text
        reply_title = text[:50] + "..." if len(text) > 50 else text
        await ChatManager.create_conversation(
            session_id=session_id,
            role="assistant",
            title=reply_title,
            content=text,
            meta_info={"reply_to": user_conv.id},
        )

    pipeline = AGUIEventPipeline()
    pipeline.add_sink(PersistenceSink(callback=_persist_reply))

    async for event in agent.reply_stream(session_id=session_id, user_content=content):
        await pipeline.emit(event)
        etype = event.get("type")
        if etype == AGUIEventType.TEXT_MESSAGE_CONTENT:
            print(event.get("delta", ""), end="", flush=True)
        elif etype == AGUIEventType.TOOL_CALL_START:
            print(f"\n  [调用工具: {event.get('toolName', '')}]", end="", flush=True)
        elif etype == AGUIEventType.RUN_ERROR:
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
