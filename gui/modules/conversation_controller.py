"""Conversation controller - manages Session and Conversation lifecycle for GUI.

Architecture:
- Session: Top-level container, persists throughout app lifetime
- Conversation: Stores complete message (user or assistant)
- Uses SessionManager as entry point, ConversationManager for conversation details
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import flet as ft

from middleware.config import g_config
from core.manager import SessionManager, ConversationManager
from gui.widgets.chat_message import ChatMessage
from gui.i18n import t
from utils.token_utils import count_tokens_messages


def _parse_timestamp(timestamp_str: str | None) -> datetime | None:
    """Parse ISO format timestamp string to datetime object."""
    if not timestamp_str:
        return None
    try:
        dt = datetime.fromisoformat(timestamp_str)
        # Remove timezone info if present, treat as local time
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return None
    try:
        dt = datetime.fromisoformat(timestamp_str)
        # If no timezone info, assume it's already in local time (UTC+8)
        # This handles legacy data stored without timezone
        if dt.tzinfo is None:
            from datetime import timezone, timedelta

            east_8 = timezone(timedelta(hours=8))
            dt = dt.replace(tzinfo=east_8)
        return dt
    except (ValueError, TypeError):
        return None


class ConversationController:
    """Controller for Session and Conversation lifecycle in GUI."""

    def __init__(self, app, logger):
        self.app = app
        self.logger = logger
        self._session_manager = SessionManager()
        self._conversation_manager = ConversationManager()

    async def _sync_session_stats(self, session_id: str | None):
        """Sync sidebar stats and GUI token display from accurate token calculation."""
        if not session_id:
            return

        session_data = await self._session_manager.get_session(session_id)
        if session_data:
            # Recalculate tokens accurately using tiktoken
            accurate_tokens = count_tokens_messages(self.app.messages)
            
            # Update sidebar stats only (must not reorder on select/click)
            self.app.sidebar.update_session_stats(
                session_id,
                session_data.get("conversation_count", 0),
                accurate_tokens,
            )
            # Update GUI token display
            self.app.total_tokens = accurate_tokens
            self.app._update_token_display()

    async def ensure_session(self) -> str:
        """Ensure current session exists, create if not."""
        if self.app.current_session_id:
            # Verify session still exists
            if await self._session_manager.exists(self.app.current_session_id):
                return self.app.current_session_id

        # Create new session
        return await self._create_new_session()

    async def _create_new_session(self) -> str:
        """Create a new session and add to sidebar."""
        try:
            model = ""
            if g_config and g_config.llm and g_config.llm.current:
                model = g_config.llm.current.model

            session_data = await self._session_manager.create_session(
                title=t("sidebar.new_session"),
                metadata={"model": model},
            )
            self.app.current_session_id = session_data["id"]

            # Add new session to sidebar
            from middleware.storage.schemas import SessionRead
            from datetime import datetime

            session_read = SessionRead(
                id=session_data["id"],
                title=session_data["title"],
                description=None,
                status="active",
                meta_info=session_data.get("metadata", {}),
                conversation_count=0,
                total_tokens=0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            self.app.sidebar.add_session(session_read)
            self.app.sidebar.set_active(session_data["id"])

            self.logger.info(
                f"[SESSION] Created and added to sidebar: {session_data['id']}"
            )
            return session_data["id"]

        except Exception as e:
            self.logger.error(f"[SESSION] Failed to create: {e}")
            raise

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for mixed language text."""
        from utils.token_utils import estimate_tokens

        return estimate_tokens(text)

    async def create_user_conversation(self, content: str) -> dict:
        """Create a user conversation."""
        session_id = await self.ensure_session()

        title = content[:50] + "..." if len(content) > 50 else content
        # Calculate approximate tokens for user message
        tokens = self._estimate_tokens(content)
        conversation = await self._conversation_manager.create_conversation(
            session_id=session_id,
            role="user",
            title=title,
            content=content,
            meta_info={"timestamp": datetime.now().isoformat()},
            tokens=tokens,
        )

        await self._sync_session_stats(session_id)

        self.logger.info(
            f"[CONV] Created user conversation: {conversation.id} ({tokens} tokens)"
        )
        return {
            "id": conversation.id,
            "role": "user",
            "content": content,
            "tokens": tokens,
        }

    async def create_assistant_conversation(
        self, content: str, reply_to: str = None, **kwargs
    ) -> dict:
        """Create an assistant conversation."""
        session_id = self.app.current_session_id
        if not session_id:
            raise ValueError("No active session")

        title = content[:50] + "..." if len(content) > 50 else content
        meta_info = {
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        if reply_to:
            meta_info["reply_to"] = reply_to

        conversation = await self._conversation_manager.create_conversation(
            session_id=session_id,
            role="assistant",
            title=title,
            content=content,
            meta_info=meta_info,
            tokens=kwargs.get("tokens", self._estimate_tokens(content)),
        )

        await self._sync_session_stats(session_id)

        self.logger.info(f"[CONV] Created assistant conversation: {conversation.id}")
        return {
            "id": conversation.id,
            "role": "assistant",
            "content": content,
        }

    async def on_new_chat(self):
        """Start new chat - clear UI and detach from current session.

        Note:
        - We intentionally do NOT create session immediately.
        - A new session is created lazily on first user message via ensure_session().
        """
        self.app.chat_list.controls.clear()
        self.app.messages.clear()
        self.app.total_tokens = 0
        self.app.current_session_id = None
        self.app._update_token_display()
        self.app.sidebar.set_active(None)
        self.app._update_toolbar_title(t("sidebar.new_session"))
        self.app._set_status(t("status.new_session"))
        self.app.page.update()

        self.logger.info("[NEW_CHAT] Cleared UI and reset active session")

    async def on_select_session(self, session_id: str):
        """Load a session and display all its conversations in chat area."""
        # Prevent duplicate clicks or selecting the current session
        if getattr(self, "_loading_session_id", None) == session_id:
            self.logger.info(
                f"[SELECT] Already loading session: {session_id}, skipping"
            )
            return

        # If already the current session, do nothing
        if self.app.current_session_id == session_id:
            self.logger.info(
                f"[SELECT] Session {session_id} already selected, skipping refresh"
            )
            return

        self._loading_session_id = session_id

        try:
            self.logger.info(f"[SELECT] Loading session: {session_id}")

            # Load session
            session = await self._session_manager.get_session(session_id)
            if not session:
                self.app._show_error(t("error.session_not_exist", id=session_id))
                return

            # Update session ID
            self.app.current_session_id = session_id

            # Get all conversations in session
            conversations = await self._conversation_manager.list_session_conversations(
                session_id, limit=100
            )

            # Build messages list for display
            self.app.messages = [
                {
                    "role": conv.role,
                    "content": conv.content,
                    "conversation_id": conv.id,
                    "session_id": conv.session_id,
                    "sequence": conv.sequence,
                    "title": conv.title,
                    "content_detail": conv.content_detail,
                    "tool_calls": conv.tool_calls,
                    "tool_call_id": conv.tool_call_id,
                    "meta_info": conv.meta_info,
                    "tokens": conv.tokens,
                    "timestamp": conv.created_at.isoformat()
                    if conv.created_at
                    else None,
                    "updated_at": conv.updated_at.isoformat()
                    if conv.updated_at
                    else None,
                }
                for conv in conversations
            ]

            # Render to chat list with dynamic width
            self.app.chat_list.controls.clear()

            # Calculate dynamic max width based on window size
            page_width = self.app.page.width or 1200
            sidebar_width = 280
            margins = 120
            available_width = page_width - sidebar_width - margins
            max_width = max(500, min(int(available_width * 0.95), 900))

            for msg in self.app.messages:
                is_user = msg.get("role") == "user"
                content = msg.get("content", "")
                if isinstance(content, str):
                    msg_widget = ChatMessage(
                        content,
                        is_user=is_user,
                        max_width=max_width,
                        steps=msg.get("steps"),
                        duration_seconds=msg.get("duration_seconds"),
                        timestamp=_parse_timestamp(msg.get("timestamp")),
                    )
                    self.app.chat_list.controls.append(msg_widget)

            # Update UI
            self.app.sidebar.set_active(session_id)
            self.app._update_toolbar_title(
                session.get("title", t("sidebar.new_session"))
            )

            # Sync token count from session
            await self._sync_session_stats(session_id)

            self.app._set_status(
                t(
                    "status.session_loaded",
                    title=session.get("title", "Untitled"),
                    count=len(conversations),
                )
            )
            self.app.page.update()

            self.logger.info(
                f"[SELECT] Loaded session with {len(self.app.messages)} messages"
            )

        except Exception as e:
            self.logger.error(f"[SELECT] Error: {e}")
            self.app._show_error(t("status.load_sessions_failed", error=str(e)))
        finally:
            # Clear loading flag
            self._loading_session_id = None

    async def on_delete_conversation(self, conversation_id: str):
        """Delete conversation with confirmation."""
        self.logger.info(f"[DELETE] Dialog for: {conversation_id}")

        async def _perform_delete():
            try:
                success = await self._conversation_manager.delete_conversation(
                    conversation_id
                )

                if success:
                    self.logger.info(f"[DELETE] Deleted: {conversation_id}")

                    # Keep in-memory chat consistent for active session
                    self.app.messages = [
                        m
                        for m in self.app.messages
                        if m.get("conversation_id") != conversation_id
                    ]
                    self.app.chat_list.controls.clear()

                    # Calculate dynamic max width
                    page_width = self.app.page.width or 1200
                    sidebar_width = 280
                    margins = 120
                    available_width = page_width - sidebar_width - margins
                    max_width = max(500, min(int(available_width * 0.95), 900))

                    for msg in self.app.messages:
                        is_user = msg.get("role") == "user"
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            self.app.chat_list.controls.append(
                                ChatMessage(
                                    content,
                                    is_user=is_user,
                                    max_width=max_width,
                                    steps=msg.get("steps"),
                                    duration_seconds=msg.get("duration_seconds"),
                                    timestamp=_parse_timestamp(msg.get("timestamp")),
                                )
                            )

                    await self._sync_session_stats(self.app.current_session_id)
                    self.app.page.update()
                    self.app._show_snackbar(t("status.conversation_deleted"))
                else:
                    self.app._show_snackbar(t("status.conversation_delete_failed"))

            except Exception as e:
                self.logger.error(f"[DELETE] Error: {e}")
                self.app._show_snackbar(f"删除失败: {str(e)}")

        def on_cancel(e):
            dialog.open = False
            self.app.page.update()

        def on_confirm(e):
            dialog.open = False
            self.app.page.update()
            asyncio.create_task(_perform_delete())

        dialog = ft.AlertDialog(
            title=ft.Text(t("dialog.delete_conversation_title")),
            content=ft.Text(t("dialog.delete_conversation_confirm")),
            actions=[
                ft.TextButton(t("dialog.cancel"), on_click=on_cancel),
                ft.ElevatedButton(
                    t("dialog.delete"),
                    on_click=on_confirm,
                    bgcolor=ft.Colors.RED_700,
                    color=ft.Colors.WHITE,
                ),
            ],
        )

        self.app.page.overlay.append(dialog)
        dialog.open = True
        self.app.page.update()

    async def on_rename_conversation(self, conversation_id: str):
        """Rename conversation."""
        self.logger.info(f"[RENAME] Dialog for: {conversation_id}")

        try:
            conversation = await self._conversation_manager.get_conversation(
                conversation_id
            )
            if not conversation:
                self.app._show_snackbar(
                    t("dialog.not_found", type=t("sidebar.conversations"))
                )
                return

            new_title_field = ft.TextField(
                value=conversation.title,
                label=t("dialog.new_title_label"),
                autofocus=True,
            )

            async def _perform_rename(new_title: str):
                try:
                    from middleware.storage.schemas import ConversationUpdate

                    updated = await self._conversation_manager.update_conversation(
                        conversation_id,
                        title=new_title,
                    )

                    if updated:
                        self.logger.info(f"[RENAME] Updated to '{new_title}'")
                        # Conversation titles are not displayed in current sidebar UI;
                        # keep operation successful and refresh status only.
                        self.app._set_status(t("status.title_updated"))
                        self.app.page.update()
                        self.app._show_snackbar(t("status.conversation_renamed"))
                    else:
                        self.app._show_snackbar(t("status.conversation_rename_failed"))

                except Exception as e:
                    self.logger.error(f"[RENAME] Error: {e}")
                    self.app._show_snackbar(f"重命名失败: {str(e)}")

            def on_cancel(e):
                dialog.open = False
                self.app.page.update()

            def on_confirm(e):
                new_title = new_title_field.value.strip()
                if not new_title:
                    self.app._show_snackbar(t("dialog.title_empty"))
                    return

                dialog.open = False
                self.app.page.update()
                asyncio.create_task(_perform_rename(new_title))

            dialog = ft.AlertDialog(
                title=ft.Text(t("dialog.rename_conversation_title")),
                content=new_title_field,
                actions=[
                    ft.TextButton(t("dialog.cancel"), on_click=on_cancel),
                    ft.ElevatedButton(t("dialog.save"), on_click=on_confirm),
                ],
            )

            self.app.page.overlay.append(dialog)
            dialog.open = True
            self.app.page.update()

        except Exception as e:
            self.logger.error(f"[RENAME] Error: {e}")
            self.app._show_snackbar(
                t("dialog.load_failed", type=t("sidebar.conversations"), error=str(e))
            )

    async def load_most_recent_session(self, page_size: int = 20):
        """Load sessions list on app start (paginated)."""
        try:
            # Load sessions for sidebar
            all_sessions = await self._session_manager.list_sessions(limit=100)

            if all_sessions:
                # Store pagination state
                self.app.sidebar.total_sessions = len(all_sessions)
                self.app.sidebar.loaded_sessions = min(page_size, len(all_sessions))

                # Only load first page to sidebar
                sessions = all_sessions[:page_size]

                for s in sessions:
                    # Convert dict to SessionRead-like object
                    from middleware.storage.schemas import SessionRead
                    from datetime import datetime

                    session_read = SessionRead(
                        id=s["id"],
                        title=s["title"],
                        description=s.get("description"),
                        status=s.get("status", "active"),
                        meta_info=s.get("metadata", {}),
                        conversation_count=s.get("conversation_count", 0),
                        total_tokens=s.get("total_tokens", 0),
                        created_at=datetime.fromisoformat(s["created_at"])
                        if s.get("created_at")
                        else datetime.now(),
                        updated_at=datetime.fromisoformat(s["updated_at"])
                        if s.get("updated_at")
                        else datetime.now(),
                    )
                    self.app.sidebar.add_session(session_read)

                # Auto-select the most recent session and load its conversations
                most_recent = sessions[0]
                self.app.current_session_id = most_recent["id"]
                self.logger.info(
                    f"[INIT] Auto-selected recent session: {most_recent['id']}"
                )

                # Load conversations for the selected session
                conversations = (
                    await self._conversation_manager.list_session_conversations(
                        most_recent["id"], limit=100
                    )
                )

                # Load messages into chat area
                self.app.messages = [
                    {
                        "role": conv.role,
                        "content": conv.content,
                        "conversation_id": conv.id,
                        "session_id": conv.session_id,
                        "sequence": conv.sequence,
                        "title": conv.title,
                        "content_detail": conv.content_detail,
                        "tool_calls": conv.tool_calls,
                        "tool_call_id": conv.tool_call_id,
                        "meta_info": conv.meta_info,
                        "tokens": conv.tokens,
                        "timestamp": conv.created_at.isoformat()
                        if conv.created_at
                        else None,
                        "updated_at": conv.updated_at.isoformat()
                        if conv.updated_at
                        else None,
                    }
                    for conv in conversations
                ]

                # Render messages to chat list with dynamic width
                # Calculate dynamic max width
                page_width = self.app.page.width or 1200
                sidebar_width = 280
                margins = 120
                available_width = page_width - sidebar_width - margins
                max_width = max(500, min(int(available_width * 0.95), 900))

                for msg in self.app.messages:
                    is_user = msg.get("role") == "user"
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        msg_widget = ChatMessage(
                            content,
                            is_user=is_user,
                            max_width=max_width,
                            steps=msg.get("steps"),
                            duration_seconds=msg.get("duration_seconds"),
                            timestamp=_parse_timestamp(msg.get("timestamp")),
                        )
                        self.app.chat_list.controls.append(msg_widget)

                # Update UI
                self.app.sidebar.set_active(most_recent["id"])
                self.app._update_toolbar_title(
                    most_recent.get("title", t("sidebar.new_session"))
                )

                # Sync token count
                await self._sync_session_stats(most_recent["id"])

                # Show pagination status
                total = len(all_sessions)
                loaded = len(sessions)
                if total > loaded:
                    self.app._set_status(
                        t("status.sessions_partial", loaded=loaded, total=total)
                    )
                else:
                    self.app._set_status(t("status.sessions_loaded", loaded=loaded))

                self.app.page.update()

                self.logger.info(
                    f"[INIT] Loaded {loaded} sessions and {len(conversations)} messages"
                )
                return True
            else:
                self.logger.info("[INIT] No existing sessions found")
                return False

        except Exception as e:
            self.logger.error(f"[INIT] Error loading sessions: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return False

    async def load_more_conversations(self, page_size: int = 20):
        """Load more conversations (pagination)."""
        if not self.app.current_session_id:
            return

        try:
            session_id = self.app.current_session_id
            current_loaded = self.app.sidebar.loaded_conversations

            # Get all conversations
            all_conversations = (
                await self._conversation_manager.list_session_conversations(
                    session_id, limit=1000
                )
            )

            # Calculate next page
            next_batch = all_conversations[current_loaded : current_loaded + page_size]

            if not next_batch:
                self.app._show_snackbar(t("status.no_more_conversations"))
                return

            # Add to sidebar
            for conv in next_batch:
                self.app.sidebar.add_conversation(conv)

            # Update pagination state
            self.app.sidebar.loaded_conversations = current_loaded + len(next_batch)

            # Update status
            total = len(all_conversations)
            loaded = self.app.sidebar.loaded_conversations
            remaining = total - loaded

            if remaining > 0:
                self.app._set_status(
                    t(
                        "status.conversations_partial",
                        loaded=loaded,
                        total=total,
                        remaining=remaining,
                    )
                )
            else:
                self.app._set_status(t("status.all_loaded", total=total))

            self.app.page.update()
            self.logger.info(
                f"[PAGINATION] Loaded {len(next_batch)} more conversations ({loaded}/{total})"
            )

        except Exception as e:
            self.logger.error(f"[PAGINATION] Error loading more conversations: {e}")
            self.app._show_snackbar(t("status.load_conversations_failed", error=str(e)))

    async def generate_conversation_title(self, conversation_id: str = None):
        """Generate title for a conversation using LLM."""
        # This would use the LLM to generate a title based on content
        # For now, just use first 30 chars of content
        pass

    async def list_conversations_for_sidebar(self) -> list:
        """Get all conversations for current session to display in sidebar."""
        if not self.app.current_session_id:
            return []

        try:
            conversations = await self._conversation_manager.list_session_conversations(
                self.app.current_session_id, limit=100
            )
            return conversations
        except Exception as e:
            self.logger.error(f"[LIST] Error: {e}")
            return []

    async def on_delete_session(self, session_id: str):
        """Delete session with confirmation."""
        self.logger.info(f"[DELETE SESSION] Dialog for: {session_id}")

        async def _perform_delete():
            try:
                success = await self._session_manager.delete_session(session_id)

                if success:
                    self.logger.info(f"[DELETE SESSION] Deleted: {session_id}")
                    self.app.sidebar.remove_session(session_id)
                    self.app.page.update()
                    self.app._show_snackbar(t("status.session_deleted"))
                    # Clear chat if deleted session was active
                    if self.app.current_session_id == session_id:
                        self.app.chat_list.controls.clear()
                        self.app.messages.clear()
                        self.app.current_session_id = None
                        self.app._update_toolbar_title(t("sidebar.new_session"))
                else:
                    self.app._show_snackbar(t("status.session_delete_failed"))

            except Exception as e:
                self.logger.error(f"[DELETE SESSION] Error: {e}")
                self.app._show_snackbar(f"删除失败: {str(e)}")

        def on_cancel(e):
            dialog.open = False
            self.app.page.update()

        def on_confirm(e):
            dialog.open = False
            self.app.page.update()
            asyncio.create_task(_perform_delete())

        dialog = ft.AlertDialog(
            title=ft.Text(t("dialog.delete_session_title")),
            content=ft.Text(t("dialog.delete_session_confirm")),
            actions=[
                ft.TextButton(t("dialog.cancel"), on_click=on_cancel),
                ft.ElevatedButton(
                    t("dialog.delete"),
                    on_click=on_confirm,
                    bgcolor=ft.Colors.RED_700,
                    color=ft.Colors.WHITE,
                ),
            ],
        )

        self.app.page.overlay.append(dialog)
        dialog.open = True
        self.app.page.update()

    async def on_rename_session(self, session_id: str):
        """Rename session."""
        self.logger.info(f"[RENAME SESSION] Dialog for: {session_id}")

        try:
            session = await self._session_manager.get_session(session_id)
            if not session:
                self.app._show_snackbar(t("dialog.not_found", type=t("sidebar.title")))
                return

            new_title_field = ft.TextField(
                value=session.get("title", ""),
                label=t("dialog.new_title_label"),
                autofocus=True,
            )

            async def _perform_rename(new_title: str):
                try:
                    from middleware.storage.schemas import SessionUpdate

                    updated = await self._session_manager.update_session(
                        session_id,
                        title=new_title,
                    )

                    if updated:
                        self.logger.info(f"[RENAME SESSION] Updated to '{new_title}'")
                        # Reload to get updated data
                        s = await self._session_manager.get_session(session_id)
                        # Create a SessionRead-like object for sidebar
                        from middleware.storage.schemas import SessionRead
                        from datetime import datetime

                        session_read = SessionRead(
                            id=s["id"],
                            title=s["title"],
                            description=s.get("description"),
                            status=s.get("status", "active"),
                            meta_info=s.get("metadata", {}),
                            conversation_count=s.get("conversation_count", 0),
                            total_tokens=s.get("total_tokens", 0),
                            created_at=datetime.fromisoformat(s["created_at"])
                            if s.get("created_at")
                            else datetime.now(),
                            updated_at=datetime.fromisoformat(s["updated_at"])
                            if s.get("updated_at")
                            else datetime.now(),
                        )
                        self.app.sidebar.update_session(session_read)
                        self.app.page.update()
                        self.app._show_snackbar(t("status.session_renamed"))
                    else:
                        self.app._show_snackbar(t("status.conversation_rename_failed"))

                except Exception as e:
                    self.logger.error(f"[RENAME SESSION] Error: {e}")
                    self.app._show_snackbar(f"重命名失败: {str(e)}")

            def on_cancel(e):
                dialog.open = False
                self.app.page.update()

            def on_confirm(e):
                new_title = new_title_field.value.strip()
                if not new_title:
                    self.app._show_snackbar(t("dialog.title_empty"))
                    return

                dialog.open = False
                self.app.page.update()
                asyncio.create_task(_perform_rename(new_title))

            dialog = ft.AlertDialog(
                title=ft.Text(t("dialog.rename_session_title")),
                content=new_title_field,
                actions=[
                    ft.TextButton(t("dialog.cancel"), on_click=on_cancel),
                    ft.ElevatedButton(t("dialog.save"), on_click=on_confirm),
                ],
            )

            self.app.page.overlay.append(dialog)
            dialog.open = True
            self.app.page.update()

        except Exception as e:
            self.logger.error(f"[RENAME SESSION] Error: {e}")
            self.app._show_snackbar(
                t("dialog.load_failed", type=t("sidebar.title"), error=str(e))
            )

    async def load_more_sessions(self, page_size: int = 20):
        """Load more sessions (pagination)."""
        try:
            current_loaded = self.app.sidebar.loaded_sessions

            # Get all sessions
            all_sessions = await self._session_manager.list_sessions(limit=100)

            # Calculate next page
            next_batch = all_sessions[current_loaded : current_loaded + page_size]

            if not next_batch:
                self.app._show_snackbar(t("status.no_more_sessions"))
                return

            # Add to sidebar
            for s in next_batch:
                # Convert dict to SessionRead-like object
                from middleware.storage.schemas import SessionRead
                from datetime import datetime

                session_read = SessionRead(
                    id=s["id"],
                    title=s["title"],
                    description=s.get("description"),
                    status=s.get("status", "active"),
                    meta_info=s.get("metadata", {}),
                    conversation_count=s.get("conversation_count", 0),
                    total_tokens=s.get("total_tokens", 0),
                    created_at=datetime.fromisoformat(s["created_at"])
                    if s.get("created_at")
                    else datetime.now(),
                    updated_at=datetime.fromisoformat(s["updated_at"])
                    if s.get("updated_at")
                    else datetime.now(),
                )
                self.app.sidebar.add_session(session_read)

            # Update pagination state
            self.app.sidebar.loaded_sessions = current_loaded + len(next_batch)

            # Update status
            total = len(all_sessions)
            loaded = self.app.sidebar.loaded_sessions
            remaining = total - loaded

            if remaining > 0:
                self.app._set_status(
                    t("status.sessions_partial", loaded=loaded, total=total)
                )
            else:
                self.app._set_status(t("status.all_loaded", total=total))

            self.app.page.update()
            self.logger.info(
                f"[PAGINATION] Loaded {len(next_batch)} more sessions ({loaded}/{total})"
            )

        except Exception as e:
            self.logger.error(f"[PAGINATION] Error loading more sessions: {e}")
            self.app._show_snackbar(t("status.load_sessions_failed", error=str(e)))
