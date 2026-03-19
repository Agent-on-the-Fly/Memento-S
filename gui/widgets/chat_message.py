"""
Chat message widget with Markdown and code highlighting support
"""

import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import flet as ft


def _format_timestamp(dt: datetime) -> str:
    """Format timestamp to string."""
    if dt is None:
        return ""
    # Direct format without timezone conversion
    return dt.strftime("%H:%M")


class CodeBlock(ft.Container):
    """A syntax-highlighted code block widget."""

    def __init__(self, code: str, language: str = ""):
        super().__init__()
        self.code = code
        self.language = language.lower() if language else "text"

        self.content = ft.Column(
            [
                # Header with language and copy button
                ft.Row(
                    [
                        ft.Text(
                            self.language or "code",
                            size=11,
                            color=ft.Colors.GREY_400,
                            weight=ft.FontWeight.W_500,
                        ),
                        ft.IconButton(
                            icon=ft.icons.Icons.COPY,
                            icon_size=16,
                            icon_color=ft.Colors.GREY_400,
                            tooltip="Copy code",
                            on_click=self._copy_code,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                # Code content
                ft.Container(
                    content=ft.Text(
                        self.code,
                        size=13,
                        font_family="Consolas, Monaco, monospace",
                        selectable=True,
                        no_wrap=False,
                    ),
                    padding=ft.Padding.only(top=8),
                ),
            ],
            spacing=4,
        )

        self.bgcolor = ft.Colors.GREY_900
        self.padding = 16
        self.border_radius = ft.BorderRadius.all(8)
        self.border = ft.Border.all(1, ft.Colors.GREY_800)

    def _copy_code(self, e):
        """Copy code to clipboard."""
        if self.page:
            self.page.set_clipboard(self.code)
            self.page.show_snack_bar(
                ft.SnackBar(content=ft.Text("Code copied to clipboard!"))
            )


class ChatMessage(ft.Container):
    """A chat message widget with Markdown support."""

    MAX_WIDTH = 700

    def __init__(
        self,
        text: str,
        is_user: bool = False,
        timestamp: Optional[datetime] = None,
        show_avatar: bool = True,
        max_width: Optional[int] = None,
        steps: Optional[int] = None,
        duration_seconds: Optional[float] = None,
    ):
        super().__init__()
        self.text = text
        self.is_user = is_user
        self.max_width = max_width or self.MAX_WIDTH
        self.timestamp = timestamp or datetime.now()
        self.show_avatar = show_avatar
        self.steps = steps
        self.duration_seconds = duration_seconds

        # Parse content
        content_controls = self._parse_content(text)

        # Avatar
        avatar = (
            ft.CircleAvatar(
                content=ft.Icon(
                    ft.icons.Icons.PERSON if is_user else ft.icons.Icons.SMART_TOY,
                    color=ft.Colors.WHITE,
                ),
                bgcolor=ft.Colors.BLUE_700 if is_user else ft.Colors.GREEN_700,
                radius=20,
            )
            if show_avatar
            else None
        )

        # Calculate appropriate width based on content
        # Dynamic width: adapts to content length, max width based on window
        # Handle mixed Chinese and English characters: Chinese chars need ~16px, English ~8px
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_chars = len(text) - chinese_chars
        estimated_width = min(
            max(chinese_chars * 16 + other_chars * 8 + 40, 120),  # Min 120px
            self.max_width,  # Use instance max_width
        )

        # Message bubble with content
        bubble_content = ft.Column(
            content_controls,
            spacing=8,
            tight=True,
        )

        # Main bubble container - dynamic width based on content
        bubble = ft.Container(
            content=bubble_content,
            bgcolor=ft.Colors.BLUE_600 if is_user else ft.Colors.GREY_800,
            padding=16,
            border_radius=ft.BorderRadius.only(
                top_left=16 if is_user else 4,
                top_right=4 if is_user else 16,
                bottom_left=16,
                bottom_right=16,
            ),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=4,
                color=ft.Colors.BLACK12,
                offset=ft.Offset(0, 2),
            ),
            width=estimated_width,
        )

        # Metadata row with timestamp, steps, and duration (for assistant messages)
        metadata_controls = [
            ft.Text(
                _format_timestamp(self.timestamp),
                size=10,
                color=ft.Colors.GREY_500,
            )
        ]

        # Add steps and duration for assistant messages
        if not is_user and (
            self.steps is not None or self.duration_seconds is not None
        ):
            meta_parts = []
            if self.steps is not None:
                meta_parts.append(f"{self.steps}步")
            if self.duration_seconds is not None:
                if self.duration_seconds < 1:
                    meta_parts.append(f"{self.duration_seconds * 1000:.0f}ms")
                else:
                    meta_parts.append(f"{self.duration_seconds:.1f}s")

            if meta_parts:
                metadata_controls.append(
                    ft.Text(
                        f" ({', '.join(meta_parts)})",
                        size=10,
                        color=ft.Colors.GREY_600,
                    )
                )

        metadata_row = ft.Row(
            metadata_controls,
            spacing=2,
            alignment=ft.MainAxisAlignment.END
            if is_user
            else ft.MainAxisAlignment.START,
        )

        # Wrap avatar in Column with tight spacing to align with content top
        avatar_column = (
            ft.Column(
                [avatar if avatar else ft.Container()],
                spacing=0,
                tight=True,
            )
            if avatar
            else ft.Container()
        )

        # Content column with max width constraint - adaptive to content
        content_column = ft.Column(
            [bubble, metadata_row],
            spacing=4,
            horizontal_alignment=ft.CrossAxisAlignment.END
            if is_user
            else ft.CrossAxisAlignment.START,
        )

        # Layout with proper alignment - all aligned to top
        # Wrap in Container with max_width to constrain while allowing content to shrink
        if is_user:
            self.content = ft.Row(
                [
                    ft.Container(expand=True),
                    content_column,
                    ft.Container(
                        content=avatar_column,
                        padding=ft.Padding.only(left=12),
                    ),
                ],
                alignment=ft.MainAxisAlignment.END,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )
        else:
            self.content = ft.Row(
                [
                    ft.Container(
                        content=avatar_column,
                        padding=ft.Padding.only(right=12),
                    ),
                    content_column,
                    ft.Container(expand=True),
                ],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )

        self.padding = ft.Padding.symmetric(vertical=8)

    def _parse_content(self, text: str) -> list:
        """Parse markdown text into Flet controls."""
        # Check if text contains complex markdown (tables, headers, lists, quotes, links)
        # If so, use the full Markdown component
        complex_markdown_patterns = [
            r"^#{1,6}\s+",  # Headers: # ## ### etc.
            r"^\s*[-*+]\s+",  # Lists: - item, * item, + item
            r"^\s*\d+\.\s+",  # Numbered lists: 1. item
            r"^>\s*",  # Blockquotes: > quote
            r"\[.*?\]\(.*?\)",  # Links: [text](url)
            r"^\s*\|",  # Tables: | col1 | col2 |
            r"^\s*---\s*$",  # Horizontal rules: ---
            r"^\s*```\w*\s*$",  # Code blocks: ``` or ```python
        ]

        needs_full_markdown = any(
            re.search(pattern, text, re.MULTILINE)
            for pattern in complex_markdown_patterns
        )

        if needs_full_markdown:
            return [self._create_markdown(text)]

        # For simple text, process with code blocks and inline formatting
        controls = []
        parts = re.split(r"```(\w*)\n(.*?)```", text, flags=re.DOTALL)
        language = ""

        for i, part in enumerate(parts):
            if i % 3 == 0:  # Regular text
                if part.strip():
                    processed = self._process_inline_formatting(part.strip())
                    controls.extend(processed)
            elif i % 3 == 1:  # Language
                language = part
            else:  # Code content
                if part.strip():
                    controls.append(CodeBlock(part.strip(), language))

        return controls if controls else [ft.Text(text, selectable=True, no_wrap=False)]

    def _create_markdown(self, text: str) -> ft.Markdown:
        """Create a Markdown control for complex content."""

        def on_link_tap(e):
            """Open link in browser when clicked."""
            import webbrowser

            webbrowser.open(e.data)

        return ft.Markdown(
            text,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            on_tap_link=on_link_tap,
        )

    def _process_inline_formatting(self, text: str) -> list:
        """Process inline markdown formatting."""
        # Check if has inline code or bold/italic
        has_formatting = any(
            [
                "`" in text,
                "**" in text,
                "*" in text,
                "<b>" in text,
                "<i>" in text,
            ]
        )

        if not has_formatting:
            return [ft.Text(text, selectable=True, no_wrap=False)]

        # Split and process
        segments = self._split_text(text)
        controls = []

        for seg in segments:
            if seg["type"] == "text":
                controls.append(ft.Text(seg["content"], selectable=True, no_wrap=False))
            elif seg["type"] == "bold":
                controls.append(
                    ft.Text(
                        seg["content"],
                        weight=ft.FontWeight.BOLD,
                        selectable=True,
                        no_wrap=False,
                    )
                )
            elif seg["type"] == "italic":
                controls.append(
                    ft.Text(seg["content"], italic=True, selectable=True, no_wrap=False)
                )
            elif seg["type"] == "code":
                controls.append(
                    ft.Container(
                        content=ft.Text(
                            seg["content"],
                            font_family="Consolas, Monaco, monospace",
                            size=13,
                            color=ft.Colors.ORANGE_300,
                            selectable=True,
                        ),
                        bgcolor=ft.Colors.GREY_900,
                        padding=ft.Padding.symmetric(horizontal=4, vertical=2),
                        border_radius=ft.BorderRadius.all(4),
                    )
                )

        return controls

    def _split_text(self, text: str) -> list:
        """Split text by inline markdown patterns."""
        patterns = [
            (r"`(.*?)`", "code"),
            (r"\*\*(.*?)\*\*", "bold"),
            (r"\*(.*?)\*", "italic"),
            (r"<b>(.*?)</b>", "bold"),
            (r"<i>(.*?)</i>", "italic"),
        ]

        segments = [{"type": "text", "content": text}]

        for pattern, ptype in patterns:
            new_segments = []
            for seg in segments:
                if seg["type"] == "text":
                    parts = re.split(pattern, seg["content"])
                    for i, part in enumerate(parts):
                        if i % 2 == 0:
                            if part:
                                new_segments.append({"type": "text", "content": part})
                        else:
                            new_segments.append({"type": ptype, "content": part})
                else:
                    new_segments.append(seg)
            segments = new_segments

        return segments
