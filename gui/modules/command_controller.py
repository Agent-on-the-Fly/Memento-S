from __future__ import annotations
from gui.i18n import t
import sys
from pathlib import Path

from middleware.config import g_config

# 将飞书脚本目录加入 sys.path，供 /feishu 命令使用
_FEISHU_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "builtin"
    / "skills"
    / "im-platform"
    / "scripts"
)
if str(_FEISHU_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_FEISHU_SCRIPTS))


class CommandController:
    """Handle slash commands from chat input."""

    def __init__(self, app):
        self.app = app
        self._feishu_receiver = None  # FeishuReceiver 实例（运行中时不为 None）
        self._feishu_loop = None  # bridge 后台线程的 asyncio event loop

    async def handle_command(self, cmd: str):
        parts = cmd.split(maxsplit=1)
        base_cmd = parts[0].lower()
        cmd_arg = parts[1].strip() if len(parts) > 1 else ""

        if base_cmd == "/clear":
            self.app._on_clear_chat()
            self.app._add_system_message(t("commands.feedback.cleared"))

        elif base_cmd == "/context":
            try:
                model_name = g_config.llm.current.model if g_config else "Unknown"
            except Exception:
                model_name = "Unknown"
            info = f"""**{t("context.info_title")}**

• {t("context.tokens", current=self.app.total_tokens)}
• {t("context.messages", count=len(self.app.messages))}
• {t("context.model", model=model_name)}
• {t("context.skills", count=len(self.app._skill_gateway.discover()) if self.app._skill_gateway else 0)}
• {t("context.session", id=self.app.current_session_id or "None")}"""
            self.app._add_system_message(info)

        elif base_cmd == "/compress":
            agent = self.app._agent
            if agent and hasattr(agent, "context_manager") and agent.context_manager:
                self.app._add_system_message(t("context.compressing"))
                self.app.page.update()
                try:
                    old_tokens, new_tokens, preview = (
                        await agent.context_manager.force_compact_now()
                    )
                    if old_tokens == 0:
                        self.app._add_system_message(
                            t("commands.feedback.compress_no_history")
                        )
                    else:
                        self.app.total_tokens = new_tokens
                        self.app._update_token_display(new_tokens)
                        msg = t(
                            "status.context_compressed_tokens",
                            old_tokens=old_tokens,
                            new_tokens=new_tokens,
                        )
                        if preview:
                            msg += f"\n\n**摘要预览:**\n{preview[:500]}"
                        self.app._add_system_message(msg)
                except Exception as e:
                    self.app._add_system_message(
                        t("context.compress_failed") + f": {e}"
                    )
            else:
                self.app._add_system_message(
                    t("commands.feedback.compress_no_session")
                )

        elif base_cmd == "/reset":
            self.app.messages.clear()
            self.app.total_tokens = 0
            self.app._update_token_display()
            self.app._on_clear_chat()
            self.app._add_system_message(t("commands.feedback.reset"))

        elif base_cmd == "/skills":
            if self.app._skill_gateway:
                manifests = self.app._skill_gateway.discover()
                skills = [m.name for m in manifests]
                skills_text = f"**{t('commands.help_title')} ({len(skills)})**\n\n"
                skills_text += "\n".join(f"• {s}" for s in skills[:30])
                if len(skills) > 30:
                    skills_text += (
                        f"\n\n... {t('chat.item_count', count=len(skills) - 30)}"
                    )
            else:
                skills_text = t("commands.descriptions.skills")
            self.app._add_system_message(skills_text)

        elif base_cmd == "/reload":
            self.app._add_system_message(t("commands.feedback.reloading"))
            self.app._add_system_message(t("commands.feedback.reloaded"))

        elif base_cmd == "/new":
            await self.app.conversation_controller.on_new_chat()
            self.app._add_system_message(t("commands.feedback.new_created"))

        elif base_cmd == "/save":
            # Auto-save is always on, just confirm
            self.app._add_system_message(t("commands.feedback.auto_saved"))

        elif base_cmd == "/load":
            if cmd_arg:
                await self.app._on_select_session(cmd_arg)
            else:
                self.app._add_system_message(t("commands.feedback.load_usage"))

        elif base_cmd == "/rename":
            if cmd_arg:
                # Rename the most recent conversation
                if self.app.messages:
                    # This would need to be implemented based on your needs
                    self.app._add_system_message(
                        t("commands.feedback.rename_feature", name=cmd_arg)
                    )
                else:
                    self.app._add_system_message(t("commands.feedback.no_conversation"))
            else:
                self.app._add_system_message(t("commands.feedback.rename_usage"))

        elif base_cmd == "/delete":
            if cmd_arg:
                await self.app._on_delete_conversation(cmd_arg)
            else:
                self.app._add_system_message(t("commands.feedback.delete_usage"))

        elif base_cmd == "/feishu":
            await self._handle_feishu_command(cmd_arg)

        elif base_cmd == "/exit":
            self.app.exit_app()

        elif base_cmd == "/help":
            help_text = self._build_help_text()
            self.app._add_system_message(help_text)

        else:
            self.app._add_system_message(t("commands.feedback.unknown", cmd=base_cmd))

        self.app.page.update()

    def _build_help_text(self) -> str:
        """Build translated help text."""
        commands = [
            ("/clear", t("commands.descriptions.clear")),
            ("/context", t("commands.descriptions.context")),
            ("/compress", t("commands.descriptions.compress")),
            ("/reset", t("commands.descriptions.reset")),
            ("/skills", t("commands.descriptions.skills")),
            ("/reload", t("commands.descriptions.reload")),
            ("/new", t("commands.descriptions.new")),
            ("/save", t("commands.descriptions.save")),
            ("/load", t("commands.descriptions.load")),
            ("/rename", t("commands.descriptions.rename")),
            ("/delete", t("commands.descriptions.delete")),
            ("/feishu start", "启动飞书长链接"),
            ("/feishu stop", "停止飞书长链接"),
            ("/feishu status", "查看飞书连接状态"),
            ("/exit", t("commands.descriptions.exit")),
            ("/help", t("commands.descriptions.help")),
        ]

        shortcuts = [
            t("commands.shortcuts.enter_send"),
            t("commands.shortcuts.shift_enter"),
            t("commands.shortcuts.esc_interrupt"),
            t("commands.shortcuts.ctrl_new"),
            t("commands.shortcuts.ctrl_clear"),
            t("commands.shortcuts.ctrl_exit"),
        ]

        cmd_lines = "\n".join([f"• `{cmd}` - {desc}" for cmd, desc in commands])
        shortcut_lines = "\n".join([f"• {s}" for s in shortcuts])

        return f"""**{t("commands.help_title")}**

{cmd_lines}

**{t("commands.shortcuts_title")}**

{shortcut_lines}"""

    async def _handle_feishu_command(self, sub: str) -> None:
        """处理 /feishu start|stop|status 子命令。"""
        sub = sub.strip().lower() or "status"

        import bootstrap as _bs

        if sub == "start":
            if self._feishu_receiver is not None or _bs._feishu_bridge_started:
                self.app._add_system_message("飞书长链接已在运行中，无需重复启动")
                return
            try:
                import threading
                import asyncio as _asyncio
                import cli.commands.feishu_bridge as _fb
                from core.memento_s.agent import MementoSAgent

                _fb._sender_sessions = _fb._load_mapping()
                agent = MementoSAgent()

                _loop_ref: list = [None]
                _ready = threading.Event()

                async def _tracked_bridge(a: MementoSAgent) -> None:
                    _loop_ref[0] = _asyncio.get_running_loop()
                    _ready.set()
                    await _fb._bridge_main(a)

                def _run() -> None:
                    _asyncio.run(_tracked_bridge(agent))

                t = threading.Thread(target=_run, daemon=True, name="feishu-bridge-gui")
                t.start()
                _ready.wait(timeout=5)
                self._feishu_loop = _loop_ref[0]
                _bs._feishu_bridge_started = True
                self.app._add_system_message("✓ 飞书长链接已启动，等待消息中...")
            except Exception as e:
                self.app._add_system_message(f"启动飞书长链接失败：{e}")

        elif sub == "stop":
            if self._feishu_receiver is None and not _bs._feishu_bridge_started:
                self.app._add_system_message("飞书长链接当前未运行")
                return
            import cli.commands.feishu_bridge as _fb

            # 1. 停止 lark WS receiver（断开连接，禁止重连）
            recv = _fb._active_receiver or self._feishu_receiver
            if recv is not None:
                try:
                    recv.stop()
                except Exception:
                    pass
                _fb._active_receiver = None
                self._feishu_receiver = None
            # 2. 停掉 bridge 的 asyncio event loop，让线程退出
            loop = self._feishu_loop or _fb._bridge_loop
            if loop is not None:
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except Exception:
                    pass
                self._feishu_loop = None
                _fb._bridge_loop = None
            _bs._feishu_bridge_started = False
            self.app._add_system_message("✓ 飞书长链接已停止")

        elif sub == "status":
            running = self._feishu_receiver is not None or _bs._feishu_bridge_started
            if running:
                src = (
                    "由启动项自动建立"
                    if _bs._feishu_bridge_started
                    else "由 /feishu start 建立"
                )
                self.app._add_system_message(f"飞书长链接状态：**运行中** ✓（{src}）")
            else:
                self.app._add_system_message(
                    "飞书长链接状态：未运行（使用 `/feishu start` 启动）"
                )

        else:
            self.app._add_system_message(
                "未知子命令: "
                + sub
                + "\n用法: `/feishu start` | `/feishu stop` | `/feishu status`"
            )
