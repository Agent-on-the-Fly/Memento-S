"""企业微信桥接 CLI 命令

提供命令行入口启动企业微信桥接。
"""

from __future__ import annotations

import asyncio

from rich.console import Console

from im.wecom.bridge import WecomBridge

console = Console()


def wecom_bridge_command() -> None:
    """启动企业微信 WebSocket 长链接，接收消息并由 Agent 处理。

    用法:
        memento wecom
        python cli/main.py wecom
    """
    console.print("[bold cyan]Memento-S × 企业微信 Bridge[/bold cyan]")

    bridge = WecomBridge()

    try:
        asyncio.run(_run_bridge(bridge))
    except KeyboardInterrupt:
        console.print("\n[dim]正在退出...[/dim]")


async def _run_bridge(bridge: WecomBridge) -> None:
    """运行企业微信桥接"""
    await bridge.start()
