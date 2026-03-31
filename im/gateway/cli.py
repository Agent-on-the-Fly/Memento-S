"""Gateway 模式桥接 CLI 命令"""

from __future__ import annotations

import asyncio

from rich.console import Console

from im.gateway.agent_worker import (
    GatewayAgentWorker,
    get_gateway_worker,
    start_gateway_worker,
    stop_gateway_worker,
)

console = Console()


def gateway_worker_command(
    gateway_url: str = "ws://127.0.0.1:8765",
    agent_id: str = "agent_main",
) -> None:
    """启动 Gateway Agent Worker，连接到 Gateway 并处理消息。

    用法:
        memento gateway-worker
        python cli/main.py gateway-worker
    """
    console.print("[bold cyan]Memento-S × Gateway Agent Worker[/bold cyan]")
    console.print(f"[dim]Connecting to {gateway_url}...[/dim]")

    try:
        asyncio.run(_run_worker(gateway_url, agent_id))
    except KeyboardInterrupt:
        console.print("\n[dim]正在退出...[/dim]")


async def _run_worker(gateway_url: str, agent_id: str) -> None:
    """运行 Gateway Worker"""
    worker = GatewayAgentWorker(gateway_url=gateway_url, agent_id=agent_id)
    await worker.start()

    # 保持运行
    stop_event = asyncio.Event()
    await stop_event.wait()
