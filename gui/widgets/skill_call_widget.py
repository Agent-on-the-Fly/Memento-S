"""
Skill call display widget for showing skill execution status
"""

import flet as ft


class SkillCallWidget(ft.Container):
    """Display skill execution status in chat."""

    def __init__(self, skill_name: str, status: str = "running", result: str = None):
        super().__init__()
        self.skill_name = skill_name
        self.status = status
        self.result = result

        # Status colors
        status_colors = {
            "running": ft.Colors.ORANGE,
            "success": ft.Colors.GREEN,
            "error": ft.Colors.RED,
        }
        status_icons = {
            "running": ft.icons.Icons.HOURGLASS_EMPTY,
            "success": ft.icons.Icons.CHECK_CIRCLE,
            "error": ft.icons.Icons.ERROR,
        }

        # Build content
        content_controls = [
            ft.Row(
                [
                    ft.Icon(
                        status_icons.get(status, ft.icons.Icons.HANDYMAN),
                        color=status_colors.get(status, ft.Colors.GREY),
                        size=18,
                    ),
                    ft.Text(
                        f"Using skill: {skill_name}",
                        size=13,
                        weight=ft.FontWeight.W_500,
                        color=status_colors.get(status, ft.Colors.WHITE),
                    ),
                ],
                spacing=8,
            ),
        ]

        # Add result preview if available
        if result and status == "success":
            preview = result[:300] + "..." if len(str(result)) > 300 else str(result)
            content_controls.append(
                ft.Container(
                    content=ft.Text(
                        preview,
                        size=11,
                        color=ft.Colors.GREY_400,
                        font_family="Consolas, monospace",
                    ),
                    bgcolor=ft.Colors.GREY_900,
                    padding=8,
                    border_radius=ft.BorderRadius.all(4),
                    margin=ft.margin.only(top=8),
                )
            )
        elif status == "error":
            content_controls.append(
                ft.Text(
                    f"Error: {result}" if result else "Execution failed",
                    size=11,
                    color=ft.Colors.RED_400,
                )
            )

        self.content = ft.Column(content_controls, spacing=4)
        self.bgcolor = ft.Colors.GREY_800
        self.padding = ft.Padding.symmetric(horizontal=12, vertical=8)
        self.border_radius = ft.BorderRadius.all(6)
        self.border = ft.Border.only(
            left=ft.BorderSide(3, status_colors.get(status, ft.Colors.GREY))
        )
        self.margin = ft.margin.only(left=48, top=4, bottom=4)


class SystemMessageWidget(ft.Container):
    """System message widget for notifications."""

    def __init__(self, message: str, msg_type: str = "info"):
        super().__init__()

        colors = {
            "info": ft.Colors.BLUE_400,
            "success": ft.Colors.GREEN_400,
            "warning": ft.Colors.ORANGE_400,
            "error": ft.Colors.RED_400,
            "system": ft.Colors.GREY_400,
        }

        bg_colors = {
            "info": ft.Colors.BLUE_900,
            "success": ft.Colors.GREEN_900,
            "warning": ft.Colors.ORANGE_900,
            "error": ft.Colors.RED_900,
            "system": ft.Colors.GREY_900,
        }

        self.content = ft.Text(
            message,
            size=12,
            color=colors.get(msg_type, ft.Colors.GREY_400),
            weight=ft.FontWeight.W_500,
        )
        self.bgcolor = bg_colors.get(msg_type, ft.Colors.GREY_900)
        self.padding = ft.Padding.symmetric(horizontal=12, vertical=6)
        self.border_radius = ft.BorderRadius.all(4)
        self.alignment = ft.Alignment(0, 0)
        self.margin = ft.margin.symmetric(vertical=8)
