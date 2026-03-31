"""Feishu WebSocket bridge command for Memento-S.

注意: 此文件已重构，实际逻辑在 im/feishu/ 模块中。
本文件仅保留 re-export 以兼容旧的导入路径。
"""

from __future__ import annotations

from im.feishu.cli import feishu_bridge_command

__all__ = ["feishu_bridge_command"]
