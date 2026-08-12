# SPDX-License-Identifier: Apache-2.0
"""
IM 渠道状态检查命令（通过 EndpointService）

用法:
    memento im-status
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def im_status_command() -> None:
    """检查 IM 渠道状态"""
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
                    Panel.fit("[bold]Summary[/bold]", border_style="green")
                )
                console.print(f"[green]IM 服务运行中，活跃渠道: {', '.join(channel_names)}[/green]")
            else:
                console.print("[yellow]IM 服务运行中，暂无活跃渠道[/yellow]")
        else:
            console.print(
                Panel.fit("[bold]Summary[/bold]", border_style="red")
            )
            console.print("[red]IM 服务未运行[/red]")
            console.print()
            console.print("[bold]启动方式:[/bold]")
            console.print("  飞书:     [cyan]memento feishu[/cyan]")
            console.print("  钉钉:     [cyan]memento dingtalk[/cyan]")
            console.print("  企业微信: [cyan]memento wecom[/cyan]")
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
        for platform in ["feishu", "dingtalk", "wecom"]:
            platform_cfg = im_cfg.get(platform, {})
            result["im_platforms"][platform] = {
                "enabled": platform_cfg.get("enabled", False),
                "configured": bool(
                    platform_cfg.get("app_id")
                    or platform_cfg.get("app_key")
                    or platform_cfg.get("corp_id")
                ),
            }
    except Exception as e:
        result["error"] = str(e)
    return result


if __name__ == "__main__":
    im_status_command()
