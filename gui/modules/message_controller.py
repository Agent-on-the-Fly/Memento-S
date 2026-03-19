"""Message controller - handles sending messages and AG-UI event orchestration."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from core.memento_s.stream_output import (
    AGUIEventPipeline,
    AGUIEventType,
    PersistenceSink,
    RunAccumulator,
)
from gui.modules.message_renderer import MessageRenderer
from gui.widgets.chat_message import ChatMessage
from gui.i18n import t
from utils.token_utils import count_tokens_messages


class MessageController:
    """Handle message sending, stream orchestration, and persistence."""

    def __init__(self, app, logger):
        self.app = app
        self.logger = logger

    def _update_tokens_display(self):
        """Calculate and update token display using accurate token counting."""
        try:
            total = count_tokens_messages(self.app.messages)
            self.app.total_tokens = total
            self.app._update_token_display(total)
            self.logger.debug(f"[MSG] Updated token display: {total} tokens")
        except Exception as e:
            self.logger.warning(f"[MSG] Failed to update token display: {e}")

    async def send_current_message(self):
        if self.app.is_processing:
            return

        text = self.app.message_input.value.strip()
        if not text:
            return

        self.app.message_input.value = ""
        self.app.page.update()

        if text.startswith("/"):
            await self.app._handle_command(text)
            return

        try:
            user_conv = await self.app.conversation_controller.create_user_conversation(
                text
            )
            self.logger.info(f"[MSG] Created user conversation: {user_conv['id']}")

            self.logger.info(f"[MSG] Created user conversation: 0")

            self.app.chat_list.controls.append(
                ChatMessage(
                    text,
                    is_user=True,
                    max_width=self.app.page.width - 400 if self.app.page.width else 700,
                    timestamp=datetime.now(),
                )
            )
            self.logger.info(f"[MSG] Created user conversation: 1")

            self.app.messages.append(
                {
                    "role": "user",
                    "content": text,
                    "conversation_id": user_conv["id"],
                    "timestamp": datetime.now().isoformat(),
                }
            )

            self.logger.info(f"[MSG] Created user conversation: 2")

            self._update_tokens_display()

            self.app.page.update()
            await self.app.chat_list.scroll_to(offset=-1, duration=300)

            self.app._current_task = asyncio.create_task(
                self.process_message(text, user_conv["id"])
            )
            try:
                await self.app._current_task
            except asyncio.CancelledError:
                self.logger.info("[MSG] Processing cancelled by user")
                return
            finally:
                self.app._current_task = None
                self.app.is_processing = False

            if len(self.app.messages) >= 2:
                await self.app._generate_conversation_title()

        except Exception as e:
            self.logger.error(f"[MSG] Error sending message: {e}")
            self.app._show_error(t("status.send_message_failed", error=str(e)))

    async def process_message(self, text: str, user_conv_id: str):
        self.app.is_processing = True
        self.app.send_button.visible = False
        self.app.stop_button.visible = True
        self.app.loading_indicator.visible = True
        self.app._set_status(t("status.ai_thinking"))
        self.app.page.update()

        renderer = MessageRenderer(self.app, self.logger, flush_interval=0.1)
        run_accumulator: RunAccumulator | None = None
        ai_conv_id = None

        try:
            if not self.app._agent or not self.app.current_session_id:
                raise ValueError("Agent or session not initialized")

            renderer.start(text)
            start_time = datetime.now()
            step_count = 0

            async def _persist_assistant_output(
                content: str, usage: dict[str, Any] | None
            ):
                nonlocal ai_conv_id, step_count
                if not content:
                    return

                # Calculate duration
                duration = (datetime.now() - start_time).total_seconds()
                tokens = usage.get("total_tokens") if usage else None

                ai_conv = await self.app.conversation_controller.create_assistant_conversation(
                    content=content,
                    reply_to=user_conv_id,
                    steps=step_count,
                    duration_seconds=duration,
                    tokens=tokens,
                )
                ai_conv_id = ai_conv["id"]
                self.app.messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "conversation_id": ai_conv_id,
                        "reply_to": user_conv_id,
                        "timestamp": datetime.now().isoformat(),
                        "steps": step_count,
                        "duration_seconds": duration,
                    }
                )

            pipeline = AGUIEventPipeline()
            pipeline.add_sink(PersistenceSink(callback=_persist_assistant_output))

            async for event in self.app._agent.reply_stream(
                session_id=self.app.current_session_id,
                user_content=text,
            ):
                if self.app._current_task and self.app._current_task.cancelled():
                    raise asyncio.CancelledError()

                await pipeline.emit(event)
                renderer.add_event(event)
                event_type = event.get("type")

                if event_type == AGUIEventType.RUN_STARTED:
                    run_accumulator = RunAccumulator(
                        run_id=event.get("runId", ""),
                        thread_id=event.get(
                            "threadId", self.app.current_session_id or ""
                        ),
                    )
                    renderer.add_system_message("🚀 AI 开始思考...", "system")
                    self.app._set_status(t("status.ai_analyzing"))

                if run_accumulator is not None:
                    run_accumulator.consume(event)

                if event_type == AGUIEventType.STEP_STARTED:
                    step = int(event.get("step", 0))
                    step_count = max(step_count, step)  # Track max step count
                    renderer.on_step_started(step)
                    self.app._set_status(t("status.thinking_step", step=step))

                elif event_type == AGUIEventType.TEXT_MESSAGE_CONTENT:
                    renderer.on_text_delta(event.get("delta", ""))
                    # Ensure visible streaming even when chunks are tiny and sparse.
                    renderer.flush()

                elif event_type == AGUIEventType.TOOL_CALL_START:
                    tool_call_id = event.get("toolCallId", "")
                    tool_name = event.get("toolName", t("status.tool"))
                    renderer.on_tool_start(
                        tool_call_id, tool_name, event.get("arguments", {})
                    )
                    self.app._set_status(t("status.using_tool", tool_name=tool_name))

                elif event_type == AGUIEventType.TOOL_CALL_RESULT:
                    tool_call_id = event.get("toolCallId", "")
                    tool_name = event.get("toolName", t("status.tool"))
                    renderer.on_tool_result(
                        tool_call_id, tool_name, event.get("result", "")
                    )
                    self.app._set_status(
                        t("status.tool_completed", tool_name=tool_name)
                    )

                elif event_type == AGUIEventType.STEP_FINISHED:
                    step = int(event.get("step", 0))
                    status = event.get("status", "unknown")
                    renderer.on_step_finished(step, status)

                elif event_type == AGUIEventType.RUN_FINISHED:
                    reason = event.get("reason", "")
                    if reason == "final_answer_generated":
                        renderer.add_system_message("✓ 回复生成完成", "success")
                    elif reason == "max_iterations_reached":
                        renderer.add_system_message("⚠ 达到最大迭代次数", "warning")

                    final_text = event.get("outputText", "")
                    if run_accumulator and run_accumulator.final_text:
                        final_text = run_accumulator.final_text
                    renderer.finalize_text(final_text)
                    self.app._set_status(t("status.completed"))

                    ctx_tokens = event.get("contextTokens")
                    if ctx_tokens is not None:
                        self.app.total_tokens = ctx_tokens
                        self.app._update_token_display(ctx_tokens)

                elif event_type == AGUIEventType.RUN_ERROR:
                    message = event.get("message", t("status.unknown_error"))
                    renderer.add_system_message(f"❌ 错误: {message}", "error")
                    self.app._set_status(
                        t("status.error_with_message", message=message[:50] + "...")
                    )
                    raise RuntimeError(message)

                renderer.flush()

            renderer.finalize_message_bubble()
            renderer.flush(force=True)

            # Calculate final duration
            total_duration = (datetime.now() - start_time).total_seconds()

            # Replace streaming message with properly formatted ChatMessage (like refresh)
            if renderer.full_response and renderer.msg_row:
                try:
                    msg_row_index = self.app.chat_list.controls.index(renderer.msg_row)
                    # Use the same dynamic width calculation as the streaming message
                    page_width = self.app.page.width or 1200
                    sidebar_width = 280
                    margins = 120
                    available_width = page_width - sidebar_width - margins
                    max_width = max(500, min(int(available_width * 0.95), 900))

                    formatted_msg = ChatMessage(
                        renderer.full_response,
                        is_user=False,
                        max_width=max_width,
                        steps=step_count,
                        duration_seconds=total_duration,
                    )
                    self.app.chat_list.controls[msg_row_index] = formatted_msg
                    self.app.page.update()
                    # Scroll to bottom after replacing with formatted message
                    await self.app.chat_list.scroll_to(offset=-1, duration=200)
                    self.logger.info(
                        f"[MSG] Replaced streaming message with ChatMessage at index {msg_row_index} (steps={step_count}, duration={total_duration:.2f}s)"
                    )
                except ValueError:
                    self.logger.warning(
                        "[MSG] Could not find streaming message in controls, keeping original"
                    )

            self.logger.info(f"[MSG] Created AI conversation: {ai_conv_id}")

        except asyncio.CancelledError:
            self.logger.info("[MSG] Stream cancelled")
            raise

        except Exception as e:
            self.logger.error(f"[MSG] Processing error: {e}")
            renderer.show_error(str(e))
            renderer.flush(force=True)

        finally:
            self.app.is_processing = False
            self.app.send_button.visible = True
            self.app.stop_button.visible = False
            self.app.loading_indicator.visible = False
            self.app.page.update()

    def stop_generation(self):
        if self.app._current_task and not self.app._current_task.done():
            self.app._current_task.cancel()
            self.app._set_status(t("status.stopped"))
            self.logger.info("[MSG] User stopped generation")
