from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import flet as ft

from gui.i18n import t
from gui.widgets.skill_call_widget import SkillCallWidget, SystemMessageWidget
from gui.widgets.skill_execution_widget import SkillExecutionWidget
from middleware.llm.utils import strip_control_tokens


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

        # Track active skill execution for real-time updates
        self.active_skill_execution: SkillExecutionWidget | None = None

        self._markdown_control: ft.Markdown | None = None

        # Step-scoped text rendering: each step gets its own Markdown area
        self._current_step: int | None = None
        self._step_markdown_controls: dict[int, ft.Markdown] = {}
        self._step_text_buffers: dict[int, str] = {}
        self._step_text_containers: dict[int, ft.Container] = {}
        self._step_bodies: dict[int, ft.Column] = {}
        self._step_arrows: dict[int, ft.Icon] = {}
        self._step_expanded: dict[int, bool] = {}
        self._last_finished_step: int | None = None

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

        if self._current_step is not None and self._current_step != step:
            self._current_step = None

        self._current_step = step
        self._step_expanded[step] = True

        def _on_link_tap(e):
            import webbrowser
            webbrowser.open(e.data)

        step_md = ft.Markdown(
            "",
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            on_tap_link=_on_link_tap,
        )
        step_text_container = ft.Container(
            content=step_md,
            bgcolor=ft.Colors.GREY_850
            if hasattr(ft.Colors, "GREY_850")
            else ft.Colors.GREY_800,
            border_radius=ft.BorderRadius.all(6),
            padding=8,
            margin=ft.margin.only(left=48),
            visible=False,
        )
        self._step_markdown_controls[step] = step_md
        self._step_text_buffers[step] = ""
        self._step_text_containers[step] = step_text_container

        step_label = ft.Text(
            t("chat.step_progress", step=step), size=12, color=ft.Colors.BLUE_300
        )

        step_header = ft.Container(
            content=ft.Row(
                [step_label],
                spacing=4,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=ft.Colors.GREY_800,
            padding=ft.Padding.symmetric(horizontal=12, vertical=8),
            border_radius=ft.BorderRadius.all(6),
            border=ft.Border.only(left=ft.BorderSide(3, ft.Colors.BLUE_400)),
            margin=ft.margin.only(left=20, top=4, bottom=4),
        )
        self.step_widgets[step] = step_header

        step_body = ft.Column(
            [],
            spacing=4,
            tight=True,
        )
        self._step_bodies[step] = step_body

        step_group = ft.Column(
            [step_header, step_body],
            spacing=2,
            tight=True,
        )
        # Insert before content area so steps appear before text
        self._insert_before_content(step_group)
        self._mark_dirty()

    def _insert_into_step_body(self, control):
        """Insert a control into the current step's body column."""
        if self._current_step is not None:
            body = self._step_bodies.get(self._current_step)
            if body is not None:
                body.controls.append(control)
                return
        self._insert_before_content(control)

    def _replace_arrow(self, step: int, icon_name: str):
        """Replace the arrow Icon instance in the header row to force UI refresh."""
        old_arrow = self._step_arrows.get(step)
        header = self.step_widgets.get(step)
        if old_arrow is None or header is None:
            return
        if not isinstance(header.content, ft.Row):
            return
        row = header.content
        new_arrow = ft.Icon(icon_name, size=16, color=ft.Colors.GREY_400)
        try:
            idx = row.controls.index(old_arrow)
            row.controls[idx] = new_arrow
        except ValueError:
            row.controls.append(new_arrow)
        self._step_arrows[step] = new_arrow

    def _toggle_step(self, step: int):
        """Toggle expand/collapse for a finished step group."""
        body = self._step_bodies.get(step)
        if body is None or step not in self._step_arrows:
            return

        expanded = self._step_expanded.get(step, True)
        self._step_expanded[step] = not expanded
        body.visible = not expanded
        self._replace_arrow(
            step,
            ft.Icons.EXPAND_MORE if not expanded else ft.Icons.CHEVRON_RIGHT,
        )
        self._mark_dirty(force=True)

    def _collapse_step(self, step: int):
        """Collapse a finished step's body."""
        body = self._step_bodies.get(step)
        if body is not None and self._step_expanded.get(step, True):
            self._step_expanded[step] = False
            body.visible = False
            self._replace_arrow(step, ft.Icons.CHEVRON_RIGHT)

    def on_step_finished(self, step: int, status: str):
        # Collapse the previous finished step now that a new one is done
        if self._last_finished_step is not None and self._last_finished_step != step:
            self._collapse_step(self._last_finished_step)

        header = self.step_widgets.get(step)
        if header and isinstance(header.content, ft.Row):
            row = header.content
            label = row.controls[0] if row.controls else None
            if isinstance(label, ft.Text):
                label.value = t("chat.step_finalize", step=step, status=status)
            header.border = ft.Border.only(
                left=ft.BorderSide(
                    3, ft.Colors.GREEN_400 if status == "finalize" else ft.Colors.BLUE_300
                )
            )

            if step not in self._step_arrows:
                icon_name = (
                    ft.Icons.CHEVRON_RIGHT if status == "finalize"
                    else ft.Icons.EXPAND_MORE
                )
                arrow = ft.Icon(icon_name, size=16, color=ft.Colors.GREY_400)
                self._step_arrows[step] = arrow
                row.controls.append(arrow)

                header.on_click = lambda e, s=step: self._toggle_step(s)
                header.ink = True

            if status == "finalize":
                self._step_expanded[step] = True
                self._collapse_step(step)
            else:
                self._step_expanded[step] = True

        self._last_finished_step = step

        if self._current_step == step:
            self._current_step = None
        self._mark_dirty()

    def on_tool_start(
        self, tool_call_id: str, tool_name: str, arguments: dict[str, Any] | None
    ):
        if self.message_content_column is None:
            return

        # Special handling for execute_skill - use real-time execution widget
        if tool_name == "execute_skill" and arguments:
            skill_name = arguments.get("skill_name", "unknown")
            widget = SkillExecutionWidget(skill_name)
            self.active_skill_execution = widget
            self.active_tool_widgets[tool_call_id] = widget

            # Set up callback on the app agent for real-time updates
            if self.app and hasattr(self.app, "_agent") and self.app._agent:

                async def skill_step_callback(
                    step_number, tool_name, status, signal, summary
                ):
                    self.on_skill_step(step_number, tool_name, status, signal, summary)

                self.app._agent.set_on_skill_step(skill_step_callback)

            self._insert_into_step_body(widget)
            self._mark_dirty()
            return

        # Regular tool call widget
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
        self._insert_into_step_body(widget)
        self._mark_dirty()

    def on_skill_start(self, skill_name: str):
        """Start tracking a skill execution with real-time updates."""
        if self.message_content_column is None:
            return

        # Create skill execution widget
        widget = SkillExecutionWidget(skill_name)
        self.active_skill_execution = widget

        self._insert_into_step_body(widget)
        self._mark_dirty()

    def on_skill_step(
        self,
        step_number: int,
        tool_name: str,
        status: str,
        signal: str,
        summary: str,
    ):
        """Update skill execution with a new step (real-time)."""
        if self.active_skill_execution is None:
            return

        self.active_skill_execution.add_step(
            step_number=step_number,
            tool_name=tool_name,
            status=status,
            signal=signal,
            summary=summary,
        )
        self._mark_dirty()

    def on_skill_finished(self, result: dict):
        """Mark skill execution as finished with final result."""
        if self.active_skill_execution is None:
            return

        self.active_skill_execution.set_final_result(result)
        self.active_skill_execution = None  # Clear reference
        self._mark_dirty()

    def on_tool_result(self, tool_call_id: str, tool_name: str, result: Any):
        old_widget = self.active_tool_widgets.get(tool_call_id)
        if not old_widget or self.message_content_column is None:
            return

        # Special handling for execute_skill - update the execution widget
        if tool_name == "execute_skill" and isinstance(
            old_widget, SkillExecutionWidget
        ):
            try:
                import json

                result_data = json.loads(result) if isinstance(result, str) else result
                self.on_skill_finished(result_data)

                # Clear the callback
                if self.app and hasattr(self.app, "_agent") and self.app._agent:
                    self.app._agent.set_on_skill_step(None)
            except Exception:
                pass
            return

        pretty_result, parsed_ok = self._format_tool_result_for_display(
            tool_name, result
        )
        status = "error" if self._is_error_tool_result(result, parsed_ok) else "success"
        new_widget = SkillCallWidget(tool_name, status=status, result=pretty_result)

        replaced = False
        for body in self._step_bodies.values():
            try:
                idx = body.controls.index(old_widget)
                body.controls[idx] = new_widget
                replaced = True
                break
            except ValueError:
                continue
        if not replaced:
            try:
                idx = self.message_content_column.controls.index(old_widget)
                self.message_content_column.controls[idx] = new_widget
            except ValueError:
                self.message_content_column.controls.append(new_widget)
        self.active_tool_widgets[tool_call_id] = new_widget
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
        elif self._looks_like_skill_execution_result(data):
            lines.extend(self._format_skill_execution_layers(data))
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
            lines.extend(self._format_dict_output_preview(output))

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

    def _looks_like_skill_execution_result(self, data: dict[str, Any]) -> bool:
        """Detect execute_skill envelope payload shape."""
        if not isinstance(data, dict):
            return False
        if "skill_name" in data and "ok" in data and "status" in data:
            return True
        output = data.get("output")
        if isinstance(output, dict) and (
            "execution_summary" in output or "final_response" in output
        ):
            return True
        return False

    def _format_skill_execution_layers(self, data: dict[str, Any]) -> list[str]:
        """Format skill executor internals as layered display content."""
        lines: list[str] = []
        skill_name = str(data.get("skill_name", "")).strip()
        output = data.get("output")

        if skill_name:
            lines.append(f"技能：{skill_name}")

        if isinstance(output, dict):
            final_response = str(output.get("final_response", "")).strip()
            if final_response:
                preview = (
                    final_response[:220] + "..."
                    if len(final_response) > 220
                    else final_response
                )
                lines.append(f"结果：{preview}")

            execution_summary = output.get("execution_summary")
            if isinstance(execution_summary, dict):
                turn_count = execution_summary.get("turn_count")
                tool_calls = execution_summary.get("tool_calls")
                primary_artifact = execution_summary.get("primary_artifact")

                if turn_count is not None or tool_calls is not None:
                    parts: list[str] = []
                    if turn_count is not None:
                        parts.append(f"executor 步数={turn_count}")
                    if tool_calls is not None:
                        parts.append(f"工具调用={tool_calls}")
                    lines.append("执行过程：" + "，".join(parts))

                if primary_artifact:
                    lines.append(f"主产物：{primary_artifact}")

                created_files = execution_summary.get("created_files") or []
                updated_files = execution_summary.get("updated_files") or []

                if isinstance(created_files, list) and created_files:
                    files_text = "、".join(str(x) for x in created_files[:3])
                    lines.append(
                        f"新增文件：{files_text}"
                        + (" ..." if len(created_files) > 3 else "")
                    )

                if isinstance(updated_files, list) and updated_files:
                    files_text = "、".join(str(x) for x in updated_files[:3])
                    lines.append(
                        f"更新文件：{files_text}"
                        + (" ..." if len(updated_files) > 3 else "")
                    )

                recent_observations = execution_summary.get("recent_observations") or []
                if isinstance(recent_observations, list) and recent_observations:
                    lines.append("内部步骤（最近）：")
                    for obs in recent_observations[:4]:
                        if not isinstance(obs, dict):
                            continue
                        tool_name = str(obs.get("tool") or "")
                        task_signal = str(obs.get("task_signal") or "")
                        exec_status = str(
                            obs.get("exec_status") or obs.get("status") or ""
                        )
                        summary_txt = str(obs.get("summary") or "").strip()
                        summary_preview = (
                            summary_txt[:90] + "..."
                            if len(summary_txt) > 90
                            else summary_txt
                        )

                        prefix_parts: list[str] = []
                        if tool_name:
                            prefix_parts.append(tool_name)
                        if exec_status:
                            prefix_parts.append(exec_status)
                        if task_signal:
                            prefix_parts.append(f"signal={task_signal}")

                        prefix = (
                            " | ".join(prefix_parts)
                            if prefix_parts
                            else "executor_step"
                        )
                        if summary_preview:
                            lines.append(f"- {prefix}: {summary_preview}")
                        else:
                            lines.append(f"- {prefix}")

            # Show richer output payload preview (not just keys)
            detail_lines = self._format_dict_output_preview(output)
            for detail in detail_lines:
                candidate = f"输出：{detail}"
                if candidate not in lines:
                    lines.append(candidate)
                if len(lines) >= 12:
                    break

        if not lines:
            lines.append("技能执行已完成")

        return lines

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

    def _format_dict_output_preview(self, output: dict[str, Any]) -> list[str]:
        """Show key-value preview instead of only dict keys."""
        lines: list[str] = []

        def _clip(value: str, max_len: int = 220) -> str:
            text = value.strip()
            return text[:max_len] + ("..." if len(text) > max_len else "")

        # Prefer meaningful fields first
        preferred_keys = [
            "final_response",
            "summary",
            "message",
            "content",
            "result",
            "error",
            "status",
        ]

        rendered_keys: set[str] = set()

        for key in preferred_keys:
            if key not in output:
                continue
            val = output.get(key)
            if val is None:
                continue
            rendered_keys.add(key)

            if isinstance(val, str):
                txt = val.strip()
                if txt:
                    lines.append(f"{key}: {_clip(txt)}")
            elif isinstance(val, (int, float, bool)):
                lines.append(f"{key}: {val}")
            elif isinstance(val, list):
                if val:
                    preview = ", ".join(_clip(str(x), 60) for x in val[:4])
                    lines.append(f"{key}: [{preview}{', ...' if len(val) > 4 else ''}]")
            elif isinstance(val, dict):
                nested_keys = list(val.keys())
                if nested_keys:
                    lines.append(
                        f"{key}: {{{'、'.join(str(k) for k in nested_keys[:6])}{' ...' if len(nested_keys) > 6 else ''}}}"
                    )

        # Then add a few remaining keys with values
        for key, val in output.items():
            if key in rendered_keys or len(lines) >= 6:
                continue

            if isinstance(val, str):
                txt = val.strip()
                if txt:
                    lines.append(f"{key}: {_clip(txt, 120)}")
            elif isinstance(val, (int, float, bool)):
                lines.append(f"{key}: {val}")
            elif isinstance(val, list):
                lines.append(f"{key}: list[{len(val)}]")
            elif isinstance(val, dict):
                lines.append(f"{key}: dict[{len(val)}]")

        if not lines:
            keys = list(output.keys())
            if keys:
                preview_keys = "、".join(str(k) for k in keys[:6])
                lines.append(
                    f"输出字段：{preview_keys}" + (" ..." if len(keys) > 6 else "")
                )

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
        if not delta:
            return

        delta = strip_control_tokens(delta)
        if not delta:
            return

        step = self._current_step
        if step is not None and step in self._step_markdown_controls:
            self._step_text_buffers[step] = self._step_text_buffers.get(step, "") + delta
            self.pending_text += delta
            text_ct = self._step_text_containers.get(step)
            if text_ct and not text_ct.visible:
                body = self._step_bodies.get(step)
                if body is not None and text_ct not in body.controls:
                    body.controls.append(text_ct)
                text_ct.visible = True
        else:
            if self._markdown_control is None:
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

        # Render remaining content before clearing _current_step,
        # so pending text still lands in the correct step buffer.
        if text_changed or self.pending_text:
            self._render_streaming_markdown()
            self.pending_text = ""

        self._current_step = None

        # Hide the global content area if no global text (all text in steps)
        if not self.full_response and self._content_area is not None:
            self._content_area.visible = False
            self._mark_dirty()

    def finalize_message_bubble(self):
        """Finalize the message bubble by ensuring final Markdown render."""
        if self._last_finished_step is not None:
            self._collapse_step(self._last_finished_step)
        self._last_finished_step = None
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
        import asyncio
        # Debug: log markdown content
        import logging

        logger = logging.getLogger(__name__)

        rendered = False

        for step, md_ctrl in self._step_markdown_controls.items():
            buf = self._step_text_buffers.get(step, "")
            if buf and md_ctrl.value != buf:
                md_ctrl.value = buf
                rendered = True

        if self._markdown_control is not None and self.full_response:
            if self._markdown_control.value != self.full_response:
                logger.debug(f"[MARKDOWN] Setting value: {self.full_response[:200]!r}...")
                self._markdown_control.value = self.full_response
                rendered = True

        if rendered and self.app.page:
            self.app.page.update()
            asyncio.create_task(self._scroll_to_bottom_async())

    def _mark_dirty(self, force: bool = False):
        self._dirty = True
        if force:
            self.flush(force=True)
