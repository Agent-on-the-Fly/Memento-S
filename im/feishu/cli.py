"""飞书桥接 CLI 命令

提供命令行入口启动飞书桥接。
"""

from __future__ import annotations

import asyncio

from rich.console import Console

from im.feishu.bridge import FeishuBridge, get_feishu_bridge

console = Console()


def feishu_bridge_command() -> None:
    """启动飞书 WebSocket 长链接，接收消息并由 Agent 处理。

    用法:
        memento feishu
        python cli/main.py feishu
    """
    # 检查是否已由 bootstrap 启动
    try:
        import bootstrap as _bs

        if _bs._feishu_bridge_started:
            console.print(
                "[dim]飞书长链接已由启动项自动建立，保持进程运行... (Ctrl+C 退出)[/dim]"
            )
            try:
                asyncio.run(asyncio.Event().wait())
            except KeyboardInterrupt:
                console.print("\n[dim]正在退出...[/dim]")
            return
    except Exception:
        pass

    console.print("[bold cyan]Memento-S × 飞书 Bridge[/bold cyan]")

    bridge = FeishuBridge()

    try:
        asyncio.run(_run_bridge(bridge))
    except KeyboardInterrupt:
        console.print("\n[dim]正在退出...[/dim]")


async def _run_bridge(bridge: FeishuBridge) -> None:
    """运行飞书桥接"""
    await bridge.start()
