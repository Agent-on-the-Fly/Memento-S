"""
Session Sidebar - Displays Session list on the left

New Architecture:
- Left sidebar shows Sessions (conversations grouped by session)
- Clicking a Session loads all its Conversations in the main chat area
- This provides a cleaner, more organized view
"""

import asyncio
from datetime import datetime, timedelta
from typing import Callable, Optional

import flet as ft

from gui.i18n import t, tp
from middleware.storage import SessionService, SessionRead


class SessionListItem(ft.Container):
    """Session list item - displays a single Session"""

    def __init__(
        self,
        session: SessionRead,
        is_active: bool = False,
        on_click: Optional[Callable] = None,
        on_delete: Optional[Callable] = None,
        on_rename: Optional[Callable] = None,
    ):
        super().__init__()
        self.session = session

        # Format time (sorted/displayed by created_at, not updated_at)
        try:
            if session.created_at:
                dt = session.created_at
                if isinstance(dt, str):
                    dt = datetime.fromisoformat(dt)
                time_str = dt.strftime("%m-%d %H:%M")
            else:
                time_str = ""
        except Exception:
            time_str = ""

        # Build title
        title_text = ft.Text(
            session.title or t("sidebar.empty"),
            size=14,
            weight=ft.FontWeight.W_600 if is_active else ft.FontWeight.W_400,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
            color=ft.Colors.WHITE,
        )

        # Build subtitle (show message count and time)
        msg_count = session.conversation_count
        subtitle_text = (
            tp("sidebar.msg_count", msg_count)
            if msg_count > 0
            else t("sidebar.new_session")
        )
        subtitle_row = ft.Row(
            [
                ft.Text(
                    subtitle_text,
                    size=10,
                    color=ft.Colors.GREY_500,
                ),
                ft.Text("•", size=10, color=ft.Colors.GREY_600),
                ft.Text(
                    time_str,
                    size=10,
                    color=ft.Colors.GREY_500,
                ),
            ],
            spacing=4,
        )

        # Build icon
        leading_icon = ft.Icon(
            ft.icons.Icons.CHAT_BUBBLE,
            color=ft.Colors.BLUE_400 if is_active else ft.Colors.GREY_500,
            size=20,
        )

        # Build context menu
        menu_items = []
        if on_rename:
            menu_items.append(
                ft.PopupMenuItem(
                    content=ft.Text(t("sidebar.rename")),
                    icon=ft.icons.Icons.EDIT,
                    on_click=lambda e: on_rename(session.id),
                )
            )
        if on_delete:
            menu_items.append(
                ft.PopupMenuItem(
                    content=ft.Text(t("sidebar.delete")),
                    icon=ft.icons.Icons.DELETE,
                    on_click=lambda e: on_delete(session.id),
                )
            )

        # Build main row
        main_row = ft.Row(
            [
                leading_icon,
                ft.Column(
                    [
                        title_text,
                        subtitle_row,
                    ],
                    spacing=2,
                    expand=True,
                ),
                ft.PopupMenuButton(
                    icon=ft.icons.Icons.MORE_VERT,
                    items=menu_items if menu_items else None,
                    icon_color=ft.Colors.GREY_500,
                    tooltip=t("sidebar.tooltip_actions"),
                )
                if menu_items
                else ft.Container(width=40),
            ],
            spacing=12,
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # Container config
        self.content = main_row
        self.padding = ft.Padding(12, 10, 8, 10)
        self.bgcolor = ft.Colors.BLUE_900 if is_active else None
        self.border_radius = ft.BorderRadius.all(8)
        self.animate = ft.Animation(200, ft.AnimationCurve.EASE_OUT)
        self.on_click = lambda e: on_click(session.id) if on_click else None
        self.on_hover = self._on_hover
        self.data = session.id

    def _on_hover(self, e):
        """Hover effect"""
        if e.data == "true":
            if self.bgcolor != ft.Colors.BLUE_900:
                self.bgcolor = ft.Colors.GREY_800
        else:
            if self.bgcolor != ft.Colors.BLUE_900:
                self.bgcolor = None
        self.update()


class SessionSidebar(ft.Container):
    """Sidebar displaying Sessions list

    Architecture:
    - Shows Sessions (top-level containers)
    - Clicking a Session loads its Conversations in main chat area
    - "New Chat" creates a new Session
    """

    def __init__(
        self,
        on_new_chat: Optional[Callable] = None,
        on_select_session: Optional[Callable] = None,
        on_delete_session: Optional[Callable] = None,
        on_rename_session: Optional[Callable] = None,
        on_load_more: Optional[Callable] = None,
    ):
        super().__init__()
        self.on_new_chat = on_new_chat
        self.on_select_session = on_select_session
        self.on_delete_session = on_delete_session
        self.on_rename_session = on_rename_session
        self.on_load_more = on_load_more

        # Services
        self._session_service = SessionService()

        # State
        self.sessions: list[SessionRead] = []
        self.active_session_id: Optional[str] = None
        self.total_sessions: int = 0
        self.loaded_sessions: int = 0

        # UI Components
        self._session_list = ft.Column(
            spacing=4,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

        def _on_new_chat_click(e):
            if on_new_chat:
                on_new_chat()

        self._new_chat_button = ft.ElevatedButton(
            t("sidebar.new_chat"),
            icon=ft.icons.Icons.ADD,
            on_click=_on_new_chat_click,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE_700,
            ),
            width=200,
        )

        self._refresh_button = ft.IconButton(
            icon=ft.icons.Icons.REFRESH,
            tooltip=t("sidebar.refresh"),
            on_click=lambda e: asyncio.create_task(self.refresh()),
        )

        self._load_more_button = ft.TextButton(
            t("sidebar.load_more"),
            on_click=lambda e: asyncio.create_task(self._load_more()),
        )

        # Layout
        self.content = ft.Column(
            [
                # Header
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Text(
                                t("sidebar.title"), size=16, weight=ft.FontWeight.BOLD
                            ),
                            self._refresh_button,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    padding=ft.Padding(16, 12, 16, 8),
                ),
                # New chat button
                ft.Container(
                    content=self._new_chat_button,
                    padding=ft.Padding(16, 0, 16, 8),
                ),
                # Divider
                ft.Divider(height=1, color=ft.Colors.GREY_800),
                # Session list
                ft.Container(
                    content=self._session_list,
                    padding=ft.Padding(8, 8, 8, 8),
                    expand=True,
                ),
                # Load more
                ft.Container(
                    content=self._load_more_button,
                    alignment=ft.Alignment(0, 0),
                    padding=ft.Padding(0, 8, 0, 8),
                ),
            ],
            spacing=0,
            expand=True,
        )

        self.width = 280
        self.bgcolor = ft.Colors.GREY_900
        self.border = ft.Border(right=ft.BorderSide(1, ft.Colors.GREY_800))

        # 注册语言切换观察者
        from gui.i18n import add_observer

        add_observer(self._on_language_changed)

    def _on_language_changed(self, new_lang: str):
        """语言切换时的回调 - 刷新所有 UI 文本"""
        # 刷新按钮文本
        self._new_chat_button.text = t("sidebar.new_chat")
        self._refresh_button.tooltip = t("sidebar.refresh")
        self._load_more_button.text = t("sidebar.load_more")

        # 刷新标题
        header_row = self.content.controls[0].content
        header_row.controls[0].value = t("sidebar.title")

        # 刷新会话列表中的时间分组标题
        self._refresh_list_ui()

        self.update()

    async def refresh(self):
        """Refresh session list from storage"""
        try:
            sessions = await self._session_service.list_recent(limit=50)
            # Sort by created_at descending (newest created session first)
            self.sessions = sorted(
                sessions,
                key=lambda s: s.created_at if s.created_at else datetime.min,
                reverse=True,
            )
            self._refresh_list_ui()
        except Exception as e:
            print(f"[Sidebar] Error refreshing: {e}")

    def _refresh_list_ui(self):
        """Refresh the list UI"""
        self._session_list.controls.clear()

        if not self.sessions:
            self._session_list.controls.append(
                ft.Container(
                    content=ft.Text(
                        t("sidebar.empty"),
                        color=ft.Colors.GREY_500,
                        size=12,
                    ),
                    alignment=ft.Alignment(0, 0),
                    padding=20,
                )
            )
            self.update()
            return

        # Group by time period
        groups = self._group_by_time_period(self.sessions)

        for period_name, sessions in groups:
            # Period header
            self._session_list.controls.append(
                ft.Container(
                    content=ft.Text(
                        period_name,
                        size=11,
                        color=ft.Colors.GREY_500,
                        weight=ft.FontWeight.W_500,
                    ),
                    padding=ft.Padding(12, 8, 12, 4),
                )
            )

            # Session items
            for session in sessions:
                is_active = session.id == self.active_session_id
                item = SessionListItem(
                    session=session,
                    is_active=is_active,
                    on_click=self._on_item_click,
                    on_delete=self.on_delete_session,
                    on_rename=self.on_rename_session,
                )
                self._session_list.controls.append(item)

        self.update()

    def _group_by_time_period(
        self, sessions: list[SessionRead]
    ) -> list[tuple[str, list[SessionRead]]]:
        """Group sessions by time period"""
        now = datetime.now()
        today = now.date()
        yesterday = today - timedelta(days=1)
        this_week_start = today - timedelta(days=today.weekday())
        this_month_start = today.replace(day=1)

        groups = {
            "today": [],
            "yesterday": [],
            "this_week": [],
            "this_month": [],
            "earlier": [],
        }

        for session in sessions:
            try:
                created = session.created_at
                if isinstance(created, str):
                    created = datetime.fromisoformat(created)
                created_date = created.date()

                if created_date == today:
                    groups["today"].append(session)
                elif created_date == yesterday:
                    groups["yesterday"].append(session)
                elif created_date >= this_week_start:
                    groups["this_week"].append(session)
                elif created_date >= this_month_start:
                    groups["this_month"].append(session)
                else:
                    groups["earlier"].append(session)
            except Exception:
                groups["earlier"].append(session)

        # Return non-empty groups with sessions sorted by created_at (descending)
        result = []
        period_keys = ["today", "yesterday", "this_week", "this_month", "earlier"]
        for key in period_keys:
            if groups[key]:
                # Sort each group by created_at descending
                sorted_sessions = sorted(
                    groups[key],
                    key=lambda s: s.created_at if s.created_at else datetime.min,
                    reverse=True,
                )
                # Translate the period name for display
                display_name = t(f"sidebar.time.{key}")
                result.append((display_name, sorted_sessions))

        return result

    async def _load_more(self):
        """Load more sessions - delegate to app controller"""
        if self.on_load_more:
            await self.on_load_more()

    def _on_item_click(self, session_id: str):
        """Handle session item click"""
        self.set_active(session_id)
        if self.on_select_session:
            result = self.on_select_session(session_id)
            if result is not None:
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)

    def set_active(self, session_id: Optional[str]):
        """Set active session - only update highlight, don't reorder"""
        old_active_id = self.active_session_id
        self.active_session_id = session_id

        # Update UI without full refresh - only change highlight
        needs_update = False
        for control in self._session_list.controls:
            # Check if this is a SessionListItem by checking for session attribute
            if hasattr(control, "session") and hasattr(control, "data"):
                if control.data == session_id:
                    # Activate this item
                    if control.bgcolor != ft.Colors.BLUE_900:
                        control.bgcolor = ft.Colors.BLUE_900
                        needs_update = True
                elif control.data == old_active_id:
                    # Deactivate previous item
                    if control.bgcolor is not None:
                        control.bgcolor = None
                        needs_update = True

        if needs_update:
            self.update()

    def add_session(self, session: SessionRead):
        """Add a session to the list"""
        self.sessions.insert(0, session)
        self._refresh_list_ui()

    def update_session(self, session: SessionRead):
        """Update a session in the list"""
        for i, s in enumerate(self.sessions):
            if s.id == session.id:
                self.sessions[i] = session
                break
        self._refresh_list_ui()

    def update_session_stats(
        self, session_id: str, conversation_count: int, total_tokens: int
    ):
        """Update session stats without changing list order/timestamp.

        Important UX rule:
        - Clicking a session should only change highlight, not reorder the list.
        """
        for s in self.sessions:
            if s.id == session_id:
                # Keep original updated_at to avoid reordering on select/click.
                from middleware.storage.schemas import SessionRead

                updated_session = SessionRead(
                    id=s.id,
                    title=s.title,
                    description=s.description,
                    status=s.status,
                    meta_info=s.meta_info,
                    conversation_count=conversation_count,
                    total_tokens=total_tokens,
                    created_at=s.created_at,
                    updated_at=s.updated_at,
                )
                idx = self.sessions.index(s)
                self.sessions[idx] = updated_session
                break
        self._refresh_list_ui()

    def remove_session(self, session_id: str):
        """Remove a session from the list"""
        self.sessions = [s for s in self.sessions if s.id != session_id]
        if self.active_session_id == session_id:
            self.active_session_id = None
        self._refresh_list_ui()
