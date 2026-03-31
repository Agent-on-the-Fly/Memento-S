"""
Enhanced skill execution widget with real-time step tracking.
"""

import flet as ft
from typing import Any, Optional
from datetime import datetime


class SkillExecutionWidget(ft.Container):
    """Real-time skill execution tracker with step-by-step display."""

    def __init__(self, skill_name: str):
        super().__init__()
        self.skill_name = skill_name
        self.start_time = datetime.now()
        self.steps: list[dict] = []
        self.current_status = "running"
        self.final_result: Optional[dict] = None

        # Create UI components
        self._build_ui()

    def _build_ui(self):
        """Build the initial UI structure."""
        # Header with skill name and status
        self.header = ft.Row(
            [
                ft.Icon(
                    ft.icons.Icons.HOURGLASS_EMPTY,
                    color=ft.Colors.ORANGE,
                    size=18,
                ),
                ft.Text(
                    f"执行技能: {self.skill_name}",
                    size=13,
                    weight=ft.FontWeight.W_500,
                    color=ft.Colors.WHITE,
                ),
                ft.Container(
                    content=ft.Text(
                        "执行中...",
                        size=11,
                        color=ft.Colors.ORANGE,
                    ),
                    ref=self._status_text,
                ),
            ],
            spacing=8,
        )

        # Steps list (initially empty, will be populated)
        # Use ExpansionTile to allow collapse/expand but keep history visible
        self.steps_list = ft.Column(spacing=2)

        self.steps_expansion = ft.ExpansionTile(
            title=ft.Text(
                "执行步骤",
                size=12,
                color=ft.Colors.GREY_400,
                weight=ft.FontWeight.W_500,
            ),
            subtitle=ft.Text(
                "0 个步骤",
                size=10,
                color=ft.Colors.GREY_500,
            ),
            controls=[self.steps_list],
            initially_expanded=True,  # Default expanded to show all steps
            collapsed_text_color=ft.Colors.GREY_400,
            text_color=ft.Colors.GREY_300,
        )

        # Final result panel (hidden initially)
        self.result_panel = ft.Container(
            content=ft.Column([]),
            visible=False,
            bgcolor=ft.Colors.GREY_900,
            padding=8,
            border_radius=ft.BorderRadius.all(4),
            margin=ft.margin.only(top=8),
        )

        # Main content
        self.content = ft.Column(
            [
                self.header,
                ft.Divider(height=1, color=ft.Colors.GREY_700),
                self.steps_expansion,  # Steps with collapse/expand
                self.result_panel,
            ],
            spacing=4,
        )

        self.bgcolor = ft.Colors.GREY_800
        self.padding = ft.Padding.symmetric(horizontal=12, vertical=8)
        self.border_radius = ft.BorderRadius.all(6)
        self.border = ft.Border.only(left=ft.BorderSide(3, ft.Colors.ORANGE))
        self.margin = ft.margin.only(left=48, top=4, bottom=4)

    def add_step(
        self,
        step_number: int,
        tool_name: str,
        status: str,
        signal: str = "",
        summary: str = "",
        elapsed_ms: Optional[int] = None,
    ):
        """Add a new execution step in real-time."""
        step_data = {
            "number": step_number,
            "tool": tool_name,
            "status": status,
            "signal": signal,
            "summary": summary,
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.now(),
        }
        self.steps.append(step_data)

        # Create step UI
        step_widget = self._create_step_widget(step_data)
        self.steps_list.controls.append(step_widget)

        # Update subtitle with step count
        self.steps_expansion.subtitle.value = f"{len(self.steps)} 个步骤"

        # Auto-expand when new step added during execution
        if self.current_status == "running":
            self.steps_expansion.expanded = True

    def _create_step_widget(self, step: dict) -> ft.Container:
        """Create UI for a single step."""
        # Status color
        status_colors = {
            "success": ft.Colors.GREEN_400,
            "error": ft.Colors.RED_400,
            "running": ft.Colors.ORANGE_400,
        }
        status_color = status_colors.get(step["status"], ft.Colors.GREY_400)

        # Signal badge
        signal_badges = {
            "strong": ("强", ft.Colors.GREEN_700, ft.Colors.GREEN_200),
            "medium": ("中", ft.Colors.BLUE_700, ft.Colors.BLUE_200),
            "weak": ("弱", ft.Colors.GREY_700, ft.Colors.GREY_200),
            "none": ("无", ft.Colors.RED_700, ft.Colors.RED_200),
        }

        signal_widget = ft.Container()
        if step.get("signal"):
            badge = signal_badges.get(
                step["signal"], ("?", ft.Colors.GREY_700, ft.Colors.GREY_200)
            )
            signal_widget = ft.Container(
                content=ft.Text(
                    badge[0],
                    size=9,
                    color=badge[2],
                    weight=ft.FontWeight.BOLD,
                ),
                bgcolor=badge[1],
                padding=ft.Padding.symmetric(horizontal=4, vertical=1),
                border_radius=ft.BorderRadius.all(3),
            )

        # Summary preview
        summary_text = step.get("summary", "")[:80]
        if len(step.get("summary", "")) > 80:
            summary_text += "..."

        # Build step row
        return ft.Container(
            content=ft.Row(
                [
                    # Step number
                    ft.Container(
                        content=ft.Text(
                            str(step["number"]),
                            size=11,
                            weight=ft.FontWeight.BOLD,
                            color=status_color,
                        ),
                        width=24,
                        alignment=ft.alignment.center,
                    ),
                    # Tool name
                    ft.Text(
                        step["tool"],
                        size=11,
                        color=ft.Colors.WHITE,
                        width=100,
                    ),
                    # Status icon
                    ft.Icon(
                        ft.icons.Icons.CHECK_CIRCLE
                        if step["status"] == "success"
                        else ft.icons.Icons.ERROR
                        if step["status"] == "error"
                        else ft.icons.Icons.RADIO_BUTTON_UNCHECKED,
                        color=status_color,
                        size=14,
                    ),
                    # Signal badge
                    signal_widget,
                    # Summary
                    ft.Text(
                        summary_text,
                        size=10,
                        color=ft.Colors.GREY_400,
                        expand=True,
                    ),
                ],
                spacing=6,
            ),
            padding=ft.Padding.symmetric(vertical=2),
        )

    def set_final_result(self, result: dict):
        """Set the final execution result."""
        self.final_result = result
        # Support both "success" (internal) and "ok" (ToolDispatcher payload) fields
        self.current_status = (
            "success" if (result.get("success") or result.get("ok")) else "error"
        )

        # Update header status
        status_colors = {
            "success": (ft.Colors.GREEN, ft.icons.Icons.CHECK_CIRCLE, "已完成"),
            "error": (ft.Colors.RED, ft.icons.Icons.ERROR, "执行失败"),
        }
        color, icon, text = status_colors.get(
            self.current_status, (ft.Colors.GREY, ft.icons.Icons.HELP, "未知")
        )

        # Rebuild header
        self.header.controls[0] = ft.Icon(icon, color=color, size=18)
        self.header.controls[2].content.value = text
        self.header.controls[2].content.color = color

        # Update border color
        self.border = ft.Border.only(left=ft.BorderSide(3, color))

        # Update steps expansion title to show completion
        self.steps_expansion.title.value = f"执行步骤 (共 {len(self.steps)} 步)"

        # Keep steps expanded to show full history
        self.steps_expansion.expanded = True

        # Build result panel
        self._build_result_panel(result)
        self.result_panel.visible = True

    def _build_result_panel(self, result: dict):
        """Build the formatted final result panel."""
        # Support both direct output (ToolDispatcher payload) and nested output
        output = result.get("output", {})
        if not output and "execution_summary" in result:
            # Handle case where result is the output itself
            output = result

        result_controls = []

        # Final response section
        final_response = output.get("final_response", "")
        if final_response:
            result_controls.extend(
                [
                    ft.Text(
                        "执行结果:",
                        size=12,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.WHITE,
                    ),
                    ft.Container(
                        content=ft.Text(
                            final_response,
                            size=11,
                            color=ft.Colors.GREY_300,
                            selectable=True,
                        ),
                        bgcolor=ft.Colors.GREY_800,
                        padding=8,
                        border_radius=ft.BorderRadius.all(4),
                    ),
                ]
            )

        # Execution summary
        summary = output.get("execution_summary", {})
        if summary:
            summary_items = []

            # Key metrics
            turn_count = summary.get("turn_count")
            tool_calls = summary.get("tool_calls")
            primary = summary.get("primary_artifact")

            if turn_count is not None:
                summary_items.append(f"执行轮次: {turn_count}")
            if tool_calls is not None:
                summary_items.append(f"工具调用: {tool_calls}")
            if primary:
                summary_items.append(f"主产物: {primary}")

            # Created/updated files
            created = summary.get("created_files", [])
            updated = summary.get("updated_files", [])

            if created:
                files_text = ", ".join(created[:3])
                if len(created) > 3:
                    files_text += f" 等{len(created)}个文件"
                summary_items.append(f"新建: {files_text}")

            if updated:
                files_text = ", ".join(updated[:3])
                if len(updated) > 3:
                    files_text += f" 等{len(updated)}个文件"
                summary_items.append(f"更新: {files_text}")

            if summary_items:
                result_controls.extend(
                    [
                        ft.Divider(height=1, color=ft.Colors.GREY_700),
                        ft.Text("执行统计:", size=11, color=ft.Colors.GREY_400),
                        ft.Column(
                            [
                                ft.Text(f"• {item}", size=10, color=ft.Colors.GREY_300)
                                for item in summary_items
                            ],
                            spacing=2,
                        ),
                    ]
                )

        # Error details if failed
        if not result.get("success") and result.get("error"):
            result_controls.extend(
                [
                    ft.Divider(height=1, color=ft.Colors.GREY_700),
                    ft.Text("错误信息:", size=11, color=ft.Colors.RED_400),
                    ft.Text(
                        result["error"],
                        size=10,
                        color=ft.Colors.RED_300,
                    ),
                ]
            )

        self.result_panel.content = ft.Column(result_controls, spacing=6)

    @property
    def _status_text(self):
        """Reference to status text container."""
        return ft.Ref[ft.Container]()
