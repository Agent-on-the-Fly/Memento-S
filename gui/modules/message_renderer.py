from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import flet as ft

from gui.i18n import t
from gui.widgets.skill_call_widget import SkillCallWidget, SystemMessageWidget


class MessageRenderer:
    """Render AG-UI events into rich chat widgets with throttled updates."""

    def __init__(self, app, logger, flush_interval: float = 0.03):
        self.app = app
        self.logger = logger
        self.flush_interval = flush_interval
        self.min_chars_per_flush = 12

        self.message_content_column: ft.Column | None = None
        self.assistant_container: ft.Container | None = None
        self.msg_row: ft.Row | None = None
        self.metadata_row: ft.Row | None = None
        self._content_area: ft.Container | None = None

        self.full_response: str = ""
        self.pending_text: str = ""
        self.last_flush_time: float = 0.0
        self._dirty: bool = False

        self.step_widgets: dict[int, ft.Container] = {}
        self.active_tool_widgets: dict[str, SkillCallWidget] = {}

        self._markdown_control: ft.Markdown | None = None

    def start(self, user_text: str):
        self.message_content_column = ft.Column(
            spacing=10,
            tight=True,
            horizontal_alignment=ft.CrossAxisAlignment.START,
        )

        # Use Markdown for both streaming and final display
        def on_link_tap(e):
            """Open link in browser when clicked."""
            import webbrowser

            webbrowser.open(e.data)

        self._markdown_control = ft.Markdown(
            "",
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            on_tap_link=on_link_tap,
        )

        self._content_area = ft.Container(
            content=self._markdown_control,
            bgcolor=ft.Colors.GREY_850
            if hasattr(ft.Colors, "GREY_850")
            else ft.Colors.GREY_800,
            border_radius=ft.BorderRadius.all(8),
            padding=10,
        )

        # Add content area at the end so it always appears after tools/steps
        # Tools and steps will be inserted at index 0
        self.message_content_column.controls.append(self._content_area)

        # Calculate dynamic width based on window size
        # Sidebar is 280px, plus margins/padding (~100px), avatar (~40px), spacing (~12px)
        page_width = self.app.page.width or 1200  # Default to 1200 if not available
        sidebar_width = 280
        margins = 120  # Total margins/padding/avatar/spacing
        available_width = page_width - sidebar_width - margins

        # Use most of available space, but with reasonable min/max
        # Min: 500px, Max: 900px or 85% of available width
        calculated_width = int(available_width * 0.95)  # 95% of available space
        adaptive_width = max(500, min(calculated_width, 900))

        # Store calculated width to keep it consistent during streaming
        self._container_width = adaptive_width

        self.assistant_container = ft.Container(
            content=self.message_content_column,
            padding=16,
            bgcolor=ft.Colors.GREY_800,
            border_radius=ft.BorderRadius.only(
                top_left=4, top_right=16, bottom_left=16, bottom_right=16
            ),
            width=self._container_width,
        )

        # Metadata row for timestamp, steps and duration
        self.metadata_row = ft.Row(
            [
                ft.Text(
                    datetime.now().strftime("%H:%M"),
                    size=10,
                    color=ft.Colors.GREY_500,
                )
            ],
            spacing=2,
            alignment=ft.MainAxisAlignment.START,
        )

        self.msg_row = ft.Row(
            [
                ft.CircleAvatar(
                    content=ft.Icon(ft.icons.Icons.SMART_TOY, color=ft.Colors.WHITE),
                    bgcolor=ft.Colors.GREEN_700,
                    radius=20,
                ),
                ft.Column(
                    [self.assistant_container, self.metadata_row],
                    spacing=4,
                    horizontal_alignment=ft.CrossAxisAlignment.START,
                ),
            ],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

        self.app.chat_list.controls.append(self.msg_row)
        self._mark_dirty()

    def add_event(self, event: dict[str, Any]):
        """No-op: event timeline display disabled by product decision."""
        return

    def _insert_before_content(self, control):
        """Insert a control before the content area (text area)."""
        if self.message_content_column is None:
            return
        # Find content area index and insert before it
        if self._content_area in self.message_content_column.controls:
            idx = self.message_content_column.controls.index(self._content_area)
            self.message_content_column.controls.insert(idx, control)
        else:
            self.message_content_column.controls.append(control)

    def add_system_message(self, message: str, msg_type: str = "system"):
        if self.message_content_column is None:
            return
        # System messages go before content
        self._insert_before_content(SystemMessageWidget(message, msg_type))
        self._mark_dirty()

    def on_step_started(self, step: int):
        if self.message_content_column is None:
            return
        step_widget = ft.Container(
            content=ft.Text(
                t("chat.step_progress", step=step), size=12, color=ft.Colors.BLUE_300
            ),
            bgcolor=ft.Colors.GREY_800,
            padding=ft.Padding.symmetric(horizontal=12, vertical=8),
            border_radius=ft.BorderRadius.all(6),
            border=ft.Border.only(left=ft.BorderSide(3, ft.Colors.BLUE_400)),
            margin=ft.margin.only(left=20, top=4, bottom=4),
        )
        self.step_widgets[step] = step_widget
        # Insert before content area so steps appear before text
        self._insert_before_content(step_widget)
        self._mark_dirty()

    def on_step_finished(self, step: int, status: str):
        widget = self.step_widgets.get(step)
        if not widget or not isinstance(widget.content, ft.Text):
            return
        widget.content.value = t("chat.step_finalize", step=step, status=status)
        widget.border = ft.Border.only(
            left=ft.BorderSide(
                3, ft.Colors.GREEN_400 if status == "finalize" else ft.Colors.BLUE_300
            )
        )
        self._mark_dirty()

    def on_tool_start(
        self, tool_call_id: str, tool_name: str, arguments: dict[str, Any] | None
    ):
        if self.message_content_column is None:
            return
        widget = SkillCallWidget(tool_name, status="running")
        if arguments and isinstance(widget.content, ft.Column):
            try:
                preview = json.dumps(arguments, ensure_ascii=False)
            except Exception:
                preview = str(arguments)
            widget.content.controls.append(
                ft.Text(
                    t("chat.skill_params", params=preview[:220]),
                    size=10,
                    color=ft.Colors.GREY_400,
                    font_family="Consolas, monospace",
                )
            )
        self.active_tool_widgets[tool_call_id] = widget
        # Insert before content area so tools appear before text
        self._insert_before_content(widget)
        self._mark_dirty()

    def on_tool_result(self, tool_call_id: str, tool_name: str, result: Any):
        old_widget = self.active_tool_widgets.get(tool_call_id)
        if not old_widget or self.message_content_column is None:
            return

        pretty_result, parsed_ok = self._format_tool_result_for_display(
            tool_name, result
        )
        status = "error" if self._is_error_tool_result(result, parsed_ok) else "success"
        new_widget = SkillCallWidget(tool_name, status=status, result=pretty_result)

        try:
            idx = self.message_content_column.controls.index(old_widget)
            self.message_content_column.controls[idx] = new_widget
            self.active_tool_widgets[tool_call_id] = new_widget
        except ValueError:
            self.message_content_column.controls.append(new_widget)
        self._mark_dirty()

    def _format_tool_result_for_display(
        self, tool_name: str, result: Any
    ) -> tuple[str, bool]:
        """Extract concise, human-readable fields from tool JSON result."""
        if not isinstance(result, str):
            return str(result), False

        text = result.strip()
        if not text:
            return "", False

        try:
            data = json.loads(text)
        except Exception:
            return result, False

        if not isinstance(data, dict):
            return result, True

        summary = str(data.get("summary", "")).strip()
        output = data.get("output")
        status = str(data.get("status", "")).strip().lower()
        error_code = data.get("error_code")

        lines: list[str] = []

        # 1) Human-readable title
        if summary and summary.lower() not in {
            "skill document loaded",
            "skill executed",
        }:
            lines.append(summary)
        elif tool_name == "read_skill":
            skill_name = str(data.get("skill_name", "")).strip()
            lines.append(f"已读取技能文档{f'：{skill_name}' if skill_name else ''}")
        elif tool_name == "skill_list":
            lines.append(summary or "技能列表已加载")
        elif tool_name == "skill_install":
            lines.append(summary or "技能安装已完成")
        else:
            if summary:
                lines.append(summary)

        # 2) Output formatting by tool type
        if tool_name == "read_skill" and isinstance(output, str):
            parsed = self._extract_skill_doc_preview(output)
            lines.extend(parsed)
        elif tool_name == "skill_list" and isinstance(output, list):
            lines.extend(self._format_skill_list_preview(output))
        elif isinstance(output, str):
            clean = output.strip()
            if clean and clean != summary:
                max_len = 240
                preview = clean[:max_len] + ("..." if len(clean) > max_len else "")
                lines.append(preview)
        elif isinstance(output, list):
            count = len(output)
            if count:
                if output and isinstance(output[0], dict) and "name" in output[0]:
                    names = [
                        str(x.get("name", "")).strip()
                        for x in output[:5]
                        if isinstance(x, dict)
                    ]
                    names = [n for n in names if n]
                    if names:
                        lines.append(
                            f"共 {count} 项："
                            + "、".join(names)
                            + (" ..." if count > 5 else "")
                        )
                    else:
                        lines.append(f"共 {count} 项")
                else:
                    lines.append(f"共 {count} 项")
        elif isinstance(output, dict):
            keys = list(output.keys())
            if keys:
                preview_keys = "、".join(keys[:6])
                lines.append(
                    f"输出字段：{preview_keys}" + (" ..." if len(keys) > 6 else "")
                )

        if error_code:
            lines.append(t("chat.error_code", code=error_code))

        if not lines:
            fallback_parts = []
            if status:
                fallback_parts.append(f"status={status}")
            if error_code:
                fallback_parts.append(f"error_code={error_code}")
            if fallback_parts:
                lines.append(", ".join(fallback_parts))
            else:
                lines.append(t("chat.execution_complete"))

        return "\n".join(lines), True

    def _extract_skill_doc_preview(self, output: str) -> list[str]:
        """Parse SKILL.md-like text and return compact readable preview."""
        lines: list[str] = []
        text = output.strip()
        if not text:
            return lines

        import re

        # Extract YAML frontmatter fields if present
        name_match = re.search(r"\nname:\s*([^\n]+)", "\n" + text)
        desc_match = re.search(r"\ndescription:\s*([^\n]+)", "\n" + text)

        if name_match:
            lines.append(t("chat.skill_name", name=name_match.group(1).strip()))
        if desc_match:
            desc = desc_match.group(1).strip()
            if len(desc) > 140:
                desc = desc[:140] + "..."
            lines.append(t("chat.skill_desc", desc=desc))

        # Extract first markdown heading as title
        heading_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        if heading_match:
            lines.append(t("chat.skill_title", title=heading_match.group(1).strip()))

        # Required params from frontmatter
        req_match = re.search(r"required:\s*\n((?:\s*-\s*[^\n]+\n?)*)", text)
        if req_match:
            req_block = req_match.group(1)
            reqs = re.findall(r"-\s*([^\n]+)", req_block)
            reqs = [r.strip() for r in reqs if r.strip()]
            if reqs:
                params_text = "、".join(reqs[:6]) + (" ..." if len(reqs) > 6 else "")
                lines.append(t("chat.required_params", params=params_text))

        if not lines:
            preview = text[:220] + ("..." if len(text) > 220 else "")
            lines.append(preview)

        return lines

    def _format_skill_list_preview(self, output: list[Any]) -> list[str]:
        lines: list[str] = []
        count = len(output)
        lines.append(t("chat.skill_count", count=count))
        if count:
            names: list[str] = []
            for item in output[:8]:
                if isinstance(item, dict):
                    name = str(item.get("name", "")).strip()
                    if name:
                        names.append(name)
            if names:
                skills_text = "、".join(names) + (" ..." if count > 8 else "")
                lines.append(t("chat.skill_name", name=skills_text))
        return lines

    def _is_error_tool_result(self, result: Any, parsed_ok: bool) -> bool:
        if isinstance(result, str) and result.startswith("Error:"):
            return True
        if not parsed_ok or not isinstance(result, str):
            return False
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                ok = data.get("ok")
                status = str(data.get("status", "")).lower()
                return ok is False or status in {
                    "failed",
                    "error",
                    "blocked",
                    "timeout",
                }
        except Exception:
            return False
        return False

    def on_text_delta(self, delta: str):
        if not delta or self._markdown_control is None:
            return

        self.full_response += delta
        self.pending_text += delta

        # Always mark dirty so flush() will trigger page.update()
        self._mark_dirty()

        should_flush = (
            len(self.pending_text) >= self.min_chars_per_flush
            or (time.time() - self.last_flush_time) >= self.flush_interval
        )
        if should_flush:
            self._render_streaming_markdown()
            self.pending_text = ""
            self.last_flush_time = time.time()

    def finalize_text(self, final_text: str | None = None):
        import logging

        logger = logging.getLogger(__name__)

        logger.debug(f"[FINALIZE] current full_response: {self.full_response!r}")
        logger.debug(f"[FINALIZE] final_text param: {final_text!r}")

        text_changed = False
        if final_text:
            if final_text != self.full_response:
                logger.warning(
                    f"[FINALIZE] TEXT MISMATCH! Replacing {len(self.full_response)} chars with {len(final_text)} chars"
                )
                logger.warning(f"[FINALIZE] Old: {self.full_response!r}")
                logger.warning(f"[FINALIZE] New: {final_text!r}")
                text_changed = True
            self.full_response = final_text

        logger.debug(f"[FINALIZE] after setting, full_response: {self.full_response!r}")

        # Always render if text changed or there's pending content
        if text_changed or self.pending_text:
            self._render_streaming_markdown()
            self.pending_text = ""

    def finalize_to_markdown(self):
        """Ensure final Markdown rendering is up to date."""
        if self._markdown_control is None:
            return

        # Update markdown with final content
        self._markdown_control.value = self.full_response

        if self.app.page:
            self.app.page.update()

    def finalize_message_bubble(self):
        """Finalize the message bubble by ensuring final Markdown render."""
        self._mark_dirty()
        # Ensure scroll to bottom after message is complete - use async version
        import asyncio

        asyncio.create_task(self._scroll_to_bottom_async())

    def show_error(self, message: str):
        if self.message_content_column is not None:
            error_widget = ft.Container(
                content=ft.Text(f"错误: {message}", color=ft.Colors.RED_400),
                padding=8,
                bgcolor=ft.Colors.RED_900,
                border_radius=4,
            )
            # Insert before content area
            self._insert_before_content(error_widget)
            self._mark_dirty(force=True)

    def flush(self, force: bool = False, scroll_to_bottom: bool = True):
        import asyncio

        if not self.app.page:
            return
        if force:
            self.app.page.update()
            self._dirty = False
            self.last_flush_time = time.time()
            if scroll_to_bottom:
                asyncio.create_task(self._scroll_to_bottom_async())
            return

        # Throttle page.update frequency for performance while keeping stream visible.
        if self._dirty and (time.time() - self.last_flush_time) >= self.flush_interval:
            self.app.page.update()
            self._dirty = False
            self.last_flush_time = time.time()
            if scroll_to_bottom:
                asyncio.create_task(self._scroll_to_bottom_async())

    def _scroll_to_bottom(self):
        """Scroll chat list to the bottom to show latest content."""
        try:
            chat_list = self.app.chat_list
            if chat_list and hasattr(chat_list, "scroll_to"):
                # Scroll to the last control in the list
                if chat_list.controls:
                    last_control = chat_list.controls[-1]
                    chat_list.scroll_to(
                        key=last_control.key
                        if hasattr(last_control, "key") and last_control.key
                        else None,
                        offset=-1,
                        duration=100,
                    )
                else:
                    chat_list.scroll_to(offset=-1, duration=100)
        except Exception:
            # Ignore scroll errors
            pass

    async def _scroll_to_bottom_async(self):
        """Async version of scroll to bottom - ensures scroll happens after render."""
        import asyncio

        try:
            # Small delay to ensure content is rendered
            await asyncio.sleep(0.05)
            self._scroll_to_bottom()
        except Exception:
            pass

    def _render_streaming_markdown(self):
        """Render streaming content as Markdown in real-time."""
        if self._markdown_control is None:
            return

        # Debug: log markdown content
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"[MARKDOWN] Setting value: {self.full_response[:200]!r}...")

        self._markdown_control.value = self.full_response
        # Update UI immediately for real-time streaming effect
        if self.app.page:
            self.app.page.update()
        # Auto-scroll to show latest content during streaming
        import asyncio

        asyncio.create_task(self._scroll_to_bottom_async())

    def _mark_dirty(self, force: bool = False):
        self._dirty = True
        if force:
            self.flush(force=True)
