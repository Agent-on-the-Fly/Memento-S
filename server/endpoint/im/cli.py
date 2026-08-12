# SPDX-License-Identifier: Apache-2.0
"""
server/endpoint/im/cli.py
IM 渠道 CLI 命令（迁移自 cli/commands/ 和 im/*/cli.py）。

统一通过 EndpointService 管理 IM 渠道。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


# ---------------------------------------------------------------------------
# 飞书命令
# ---------------------------------------------------------------------------


def feishu_bridge_command() -> None:
    """启动飞书渠道（通过 EndpointService）

    用法:
        memento feishu
    """
    console.print("[bold cyan]Memento-S × 飞书渠道[/bold cyan]")

    try:
        from server.endpoint.im import EndpointService
        from middleware.im.gateway import ChannelType, ConnectionMode

        service = EndpointService.get_instance()

        # 检查凭证
        feishu_cfg = {}
        try:
            cfg_path = Path.home() / "memento_s" / "config.json"
            with open(cfg_path, "r", encoding="utf-8") as f:
                feishu_cfg = json.load(f).get("im", {}).get("feishu", {})
        except Exception:
            pass

        app_id = feishu_cfg.get("app_id", "")
        app_secret = feishu_cfg.get("app_secret", "")
        if not (app_id and app_secret):
            console.print("[red]错误：未配置飞书凭证。[/red]")
            console.print("[dim]请在设置中配置 app_id 和 app_secret[/dim]")
            return

        try:
            asyncio.run(
                service.start_channel(
                    account_id="feishu_main",
                    channel_type=ChannelType.FEISHU,
                    credentials={
                        "app_id": app_id,
                        "app_secret": app_secret,
                        "encrypt_key": feishu_cfg.get("encrypt_key", ""),
                        "verification_token": feishu_cfg.get("verification_token", ""),
                    },
                    mode=ConnectionMode.WEBSOCKET,
                )
            )
            console.print("[green]✓ 飞书渠道已启动，等待消息中...[/green]")
            console.print("[dim]按 Ctrl+C 退出[/dim]")

            async def wait():
                await asyncio.Event().wait()

            asyncio.run(wait())
        except KeyboardInterrupt:
            console.print("\n[dim]正在退出...[/dim]")
        except Exception as e:
            console.print(f"[red]启动失败: {e}[/red]")
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")


# ---------------------------------------------------------------------------
# 钉钉命令
# ---------------------------------------------------------------------------


def dingtalk_bridge_command() -> None:
    """启动钉钉渠道（通过 EndpointService）

    用法:
        memento dingtalk
    """
    console.print("[bold cyan]Memento-S × 钉钉渠道[/bold cyan]")

    try:
        from server.endpoint.im import EndpointService
        from middleware.im.gateway import ChannelType, ConnectionMode

        service = EndpointService.get_instance()

        try:
            asyncio.run(
                service.start_channel(
                    account_id="dingtalk_main",
                    channel_type=ChannelType.DINGTALK,
                    credentials={},
                    mode=ConnectionMode.WEBSOCKET,
                )
            )
            console.print("[green]✓ 钉钉渠道已启动，等待消息中...[/green]")
            console.print("[dim]按 Ctrl+C 退出[/dim]")
            asyncio.run(asyncio.Event().wait())
        except KeyboardInterrupt:
            console.print("\n[dim]正在退出...[/dim]")
        except Exception as e:
            console.print(f"[red]启动失败: {e}[/red]")
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")


# ---------------------------------------------------------------------------
# 微信命令
# ---------------------------------------------------------------------------


def wechat_bridge_command() -> None:
    """启动微信渠道（通过 EndpointService）

    用法:
        memento wechat
    """
    console.print("[bold cyan]Memento-S × 微信渠道[/bold cyan]")

    try:
        from server.endpoint.im import EndpointService
        from middleware.im.gateway import ChannelType, ConnectionMode

        service = EndpointService.get_instance()

        # 检查凭证
        wechat_cfg = {}
        try:
            cfg_path = Path.home() / "memento_s" / "config.json"
            with open(cfg_path, "r", encoding="utf-8") as f:
                wechat_cfg = json.load(f).get("im", {}).get("wechat", {})
        except Exception:
            pass

        token = wechat_cfg.get("token", "")
        if not token:
            console.print("[red]错误：未配置微信凭证。[/red]")
            console.print("[dim]请先运行 [cyan]memento wechat login[/cyan] 登录[/dim]")
            return

        try:
            asyncio.run(
                service.start_channel(
                    account_id="wechat_main",
                    channel_type=ChannelType.WECHAT,
                    credentials={
                        "token": token,
                        "base_url": wechat_cfg.get("base_url", "https://ilinkai.weixin.qq.com"),
                    },
                    mode=ConnectionMode.POLLING,
                )
            )
            console.print("[green]✓ 微信渠道已启动，等待消息中...[/green]")
            console.print("[dim]按 Ctrl+C 退出[/dim]")

            async def wait():
                await asyncio.Event().wait()

            asyncio.run(wait())
        except KeyboardInterrupt:
            console.print("\n[dim]正在退出...[/dim]")
        except Exception as e:
            console.print(f"[red]启动失败: {e}[/red]")
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")


# ---------------------------------------------------------------------------
# 企业微信命令
# ---------------------------------------------------------------------------


def wecom_bridge_command() -> None:
    """启动企业微信渠道（通过 EndpointService）

    用法:
        memento wecom
    """
    console.print("[bold cyan]Memento-S × 企业微信渠道[/bold cyan]")

    try:
        from server.endpoint.im import EndpointService
        from middleware.im.gateway import ChannelType, ConnectionMode

        service = EndpointService.get_instance()

        try:
            asyncio.run(
                service.start_channel(
                    account_id="wecom_main",
                    channel_type=ChannelType.WECOM,
                    credentials={},
                    mode=ConnectionMode.WEBSOCKET,
                )
            )
            console.print("[green]✓ 企业微信渠道已启动，等待消息中...[/green]")
            console.print("[dim]按 Ctrl+C 退出[/dim]")
            asyncio.run(asyncio.Event().wait())
        except KeyboardInterrupt:
            console.print("\n[dim]正在退出...[/dim]")
        except Exception as e:
            console.print(f"[red]启动失败: {e}[/red]")
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")


# ---------------------------------------------------------------------------
# Gateway Worker 命令
# ---------------------------------------------------------------------------


def gateway_worker_command(
    gateway_url: str = "ws://127.0.0.1:8765",
    agent_id: str = "agent_main",
) -> None:
    """启动 Gateway Agent Worker，连接到 Gateway 并处理消息。

    用法:
        memento gateway-worker
    """
    console.print("[bold cyan]Memento-S × Gateway Agent Worker[/bold cyan]")
    console.print(f"[dim]Connecting to {gateway_url}...[/dim]")

    try:
        asyncio.run(_run_worker(gateway_url, agent_id))
    except KeyboardInterrupt:
        console.print("\n[dim]正在退出...[/dim]")


async def _run_worker(gateway_url: str, agent_id: str) -> None:
    from server.endpoint.im import EndpointService

    service = EndpointService.get_instance()
    service.start_in_background()

    stop_event = asyncio.Event()
    await stop_event.wait()


# ---------------------------------------------------------------------------
# IM 状态检查命令
# ---------------------------------------------------------------------------


def im_status_command() -> None:
    """检查 IM 渠道状态

    用法:
        memento im-status
    """
    console.print()
    console.print(
        Panel.fit("[bold cyan]IM 渠道状态检查[/bold cyan]", border_style="cyan")
    )
    console.print()

    # 1. 检查配置
    config = _check_configuration()
    config_table = Table(title="[1] Configuration", box=box.ROUNDED)
    config_table.add_column("Item", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_row("Config file", config.get("config_file", ""))
    if "error" in config:
        config_table.add_row("Error", f"[red]{config['error']}[/red]")
    else:
        config_table.add_row("Gateway enabled", str(config.get("gateway_enabled", False)))
        for platform, info in config.get("im_platforms", {}).items():
            status = "[green]enabled[/green]" if info.get("enabled") else "[red]disabled[/red]"
            configured = "configured" if info.get("configured") else "not configured"
            config_table.add_row(f"Platform: {platform}", f"{status} ({configured})")
    console.print(config_table)
    console.print()

    # 2. 检查 EndpointService
    service_table = Table(title="[2] EndpointService", box=box.ROUNDED)
    service_table.add_column("Item", style="cyan")
    service_table.add_column("Value", style="white")

    try:
        from server.endpoint.im import EndpointService

        service = EndpointService.get_instance()
        service_table.add_row("Status", "[green]running[/green]" if service.is_running else "[red]stopped[/red]")
        channels = service.list_channels()
        if channels:
            channel_names = [c.get("channel_type", "?") for c in channels]
            service_table.add_row("Active channels", ", ".join(channel_names))
            service_table.add_row("Channel count", str(len(channels)))
        else:
            service_table.add_row("Active channels", "None")
    except Exception as e:
        service_table.add_row("Status", f"[red]error: {e}[/red]")

    console.print(service_table)
    console.print()

    # 3. 总结
    try:
        from server.endpoint.im import EndpointService

        service = EndpointService.get_instance()
        if service.is_running:
            channels = service.list_channels()
            if channels:
                channel_names = [c.get("channel_type", "?") for c in channels]
                console.print(
                    Panel.fit(
                        "[bold]Summary[/bold]",
                        border_style="green",
                    )
                )
                console.print(f"[green]IM 服务运行中，活跃渠道: {', '.join(channel_names)}[/green]")
            else:
                console.print("[yellow]IM 服务运行中，暂无活跃渠道[/yellow]")
        else:
            console.print(
                Panel.fit(
                    "[bold]Summary[/bold]",
                    border_style="red",
                )
            )
            console.print("[red]IM 服务未运行[/red]")
            console.print()
            console.print("[bold]启动方式:[/bold]")
            console.print("  飞书:    [cyan]memento feishu[/cyan]")
            console.print("  钉钉:    [cyan]memento dingtalk[/cyan]")
            console.print("  企业微信: [cyan]memento wecom[/cyan]")
            console.print("  微信:    [cyan]memento wechat[/cyan]")
    except Exception as e:
        console.print(f"[red]检查失败: {e}[/red]")

    console.print()


def _check_configuration() -> dict:
    """检查配置文件"""
    result = {"config_file": "", "gateway_enabled": False, "im_platforms": {}}
    config_path = Path.home() / "memento_s" / "config.json"
    result["config_file"] = str(config_path)
    if not config_path.exists():
        result["error"] = "Config file not found"
        return result
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        gateway_cfg = config.get("gateway", {})
        result["gateway_enabled"] = gateway_cfg.get("enabled", False)
        im_cfg = config.get("im", {})
        for platform in ["feishu", "dingtalk", "wecom", "wechat"]:
            platform_cfg = im_cfg.get(platform, {})
            result["im_platforms"][platform] = {
                "enabled": platform_cfg.get("enabled", False),
                "configured": bool(
                    platform_cfg.get("app_id")
                    or platform_cfg.get("app_key")
                    or platform_cfg.get("corp_id")
                    or platform_cfg.get("token")  # WeChat uses token
                ),
            }
    except Exception as e:
        result["error"] = str(e)
    return result
