"""钉钉桥接 CLI 命令

提供命令行入口启动钉钉桥接。
"""

from __future__ import annotations

import asyncio

from rich.console import Console

from im.dingtalk.bridge import DingtalkBridge

console = Console()


def dingtalk_bridge_command() -> None:
    """启动钉钉 Stream 长链接，接收消息并由 Agent 处理。

    用法:
        memento dingtalk
        python cli/main.py dingtalk
    """
    console.print("[bold cyan]Memento-S × 钉钉 Bridge[/bold cyan]")

    bridge = DingtalkBridge()

    try:
        asyncio.run(_run_bridge(bridge))
    except KeyboardInterrupt:
        console.print("\n[dim]正在退出...[/dim]")


async def _run_bridge(bridge: DingtalkBridge) -> None:
    """运行钉钉桥接"""
    await bridge.start()
