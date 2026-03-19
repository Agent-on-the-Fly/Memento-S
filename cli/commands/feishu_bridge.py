"""Feishu WebSocket bridge command for Memento-S.

启动一个独立进程，与飞书保持 WebSocket 长链接：
  - 用户在飞书发消息 → Agent 处理 → 通过机器人回复
  - 每个用户拥有独立的 DB Session，对话历史跨重启保留

用法：
    memento feishu
    python cli/main.py feishu
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FEISHU_SCRIPTS = (
    _PROJECT_ROOT / "builtin" / "skills" / "im-platform" / "scripts"
)

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_FEISHU_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_FEISHU_SCRIPTS))

from core.manager import ConversationManager, SessionManager  # noqa: E402
from core.memento_s.agent import MementoSAgent  # noqa: E402
from core.memento_s.stream_output import (  # noqa: E402
    AGUIEventPipeline,
    AGUIEventType,
    PersistenceSink,
)
from middleware.config import g_config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

console = Console()
logger = get_logger(__name__)

# 进程内缓存：feishu sender_id → DB session_id (UUID)
_sender_sessions: dict[str, str] = {}

# 当前活跃的 FeishuReceiver 实例（供外部停止用）
_active_receiver: Any = None
# bridge 后台线程的 asyncio event loop（供外部停止用）
_bridge_loop: Any = None


# --------------------------------------------------------------------------- #
# 映射文件持久化
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
    text = json.dumps(mapping, ensure_ascii=False, indent=2)
    p.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Session 管理
# --------------------------------------------------------------------------- #

async def _get_or_create_session(
    sender_id: str, session_manager: SessionManager
) -> str:
    """获取或创建飞书用户对应的 DB Session，返回 DB session_id (UUID)。"""
    if sender_id in _sender_sessions:
        db_sid = _sender_sessions[sender_id]
        if await session_manager.exists(db_sid):
            return db_sid
        del _sender_sessions[sender_id]

    session = await session_manager.create_session(
        title=f"飞书: {sender_id}",
        metadata={"feishu_sender_id": sender_id, "source": "feishu"},
    )
    db_sid = session["id"]
    _sender_sessions[sender_id] = db_sid
    _save_mapping(_sender_sessions)
    logger.info(f"[feishu-bridge] 为用户 {sender_id} 创建新会话: {db_sid}")
    return db_sid


# --------------------------------------------------------------------------- #
# 消息处理
# --------------------------------------------------------------------------- #

async def _handle_message(
    msg: dict[str, Any],
    agent: MementoSAgent,
    session_manager: SessionManager,
    conversation_manager: ConversationManager,
) -> None:
    """接收飞书消息，交由 Agent 处理，持久化对话并回复。"""
    from messaging import send_text_message

    sender_id: str = msg.get("sender_id", "")
    content: str = (msg.get("content") or "").strip()

    if not sender_id or not content:
        return

    console.print(f"\n[cyan][飞书→Agent][/cyan] {sender_id}: {content[:80]}")

    session_id = await _get_or_create_session(sender_id, session_manager)

    # 保存用户消息到 DB
    user_title = content[:50] + "..." if len(content) > 50 else content
    user_conv = await conversation_manager.create_conversation(
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
        await conversation_manager.create_conversation(
            session_id=session_id,
            role="assistant",
            title=reply_title,
            content=text,
            meta_info={"reply_to": user_conv.id},
        )

    pipeline = AGUIEventPipeline()
    pipeline.add_sink(PersistenceSink(callback=_persist_reply))

    try:
        # history=None → agent 从 DB 自动加载历史
        async for event in agent.reply_stream(
            session_id=session_id, user_content=content
        ):
            await pipeline.emit(event)
            if event.get("type") == AGUIEventType.TOOL_CALL_START:
                console.print(
                    f"  [yellow]{event.get('toolName', '')}[/yellow] "
                    f"[dim]{str(event.get('arguments', ''))[:60]}[/dim]"
                )
    except Exception as e:
        logger.error(f"[feishu-bridge] Agent 处理出错: {e}", exc_info=True)
        await send_text_message(sender_id, "处理出错，请稍后重试。")
        return

    if final_text:
        console.print(f"[green][Agent→飞书][/green] {final_text[:80]}...")
        await send_text_message(sender_id, final_text)
    else:
        logger.info("[feishu-bridge] Agent 无文字回复（可能已执行任务），不发送消息")


async def _bridge_main(agent: MementoSAgent) -> None:
    """主协程：启动飞书 WebSocket 接收器，持续等待消息直到被中断。"""
    global _active_receiver
    from receiver import FeishuReceiver

    session_manager = SessionManager()
    conversation_manager = ConversationManager()

    loop = asyncio.get_running_loop()

    def on_message(msg: dict[str, Any]) -> None:
        """由 WebSocket 后台线程调用，线程安全地在主事件循环中创建独立 Task。"""

        def _schedule() -> None:
            coro = _handle_message(
                msg, agent, session_manager, conversation_manager
            )
            task = loop.create_task(coro)
            task.add_done_callback(
                lambda t: logger.error(
                    f"[feishu-bridge] 未捕获的任务异常: {t.exception()}",
                )
                if not t.cancelled() and t.exception()
                else None
            )

        loop.call_soon_threadsafe(_schedule)

    receiver = FeishuReceiver(on_message=on_message)
    _active_receiver = receiver
    receiver.start_in_background()

    console.print("[dim]飞书长链接已建立，等待消息... (Ctrl+C 退出)[/dim]\n")

    stop = asyncio.Event()
    await stop.wait()


def feishu_bridge_command() -> None:
    """启动飞书 WebSocket 长链接，接收消息并由 Agent 处理。"""
    # 防止 bootstrap 自动启动后，手动运行 `memento feishu` 再次重复启动
    try:
        import bootstrap as _bs
        if _bs._feishu_bridge_started:
            console.print(
                "[dim]飞书长链接已由启动项自动建立，保持进程运行... (Ctrl+C 退出)[/dim]"
            )
            try:
                asyncio.run(asyncio.Event().wait())  # 阻塞直到 Ctrl+C
            except KeyboardInterrupt:
                console.print("\n[dim]正在退出...[/dim]")
            return
    except Exception:
        pass

    global _sender_sessions
    _sender_sessions = _load_mapping()

    console.print("[bold cyan]Memento-S × 飞书 Bridge[/bold cyan]")
    console.print(f"[dim]已加载 {len(_sender_sessions)} 个飞书会话映射[/dim]")

    agent = MementoSAgent()

    try:
        asyncio.run(_bridge_main(agent))
    except KeyboardInterrupt:
        console.print("\n[dim]正在退出...[/dim]")
