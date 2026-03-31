"""
Storage module for Weixin SDK.

Provides persistence for sync buffers (cursors) to resume message polling across restarts.
"""

from .sync_buffer import SyncBufferManager

__all__ = ["SyncBufferManager"]
