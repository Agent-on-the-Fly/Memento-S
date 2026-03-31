"""
IM Platform Status Checker
检查 IM 平台（Gateway 和 Bridge 模式）的启动状态

用法:
    memento im-status
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def check_bridge_mode() -> dict[str, Any]:
    """检查 Bridge 模式状态（使用顶层 im/ 模块）"""
    result = {"mode": "bridge", "status": "not_started", "details": {}}

    try:
        # 使用顶层 im 模块检查状态
        from im.feishu import get_feishu_bridge
        from im.dingtalk import get_dingtalk_bridge
        from im.wecom import get_wecom_bridge
        import bootstrap as _bs

        # 飞书
        feishu_bridge = get_feishu_bridge()
        result["details"]["feishu_started"] = _bs._feishu_bridge_started
        result["details"]["feishu_running"] = (
            feishu_bridge.is_running if feishu_bridge else False
        )
        result["details"]["feishu_session_count"] = (
            len(feishu_bridge._sender_sessions) if feishu_bridge else 0
        )

        # 钉钉
        dingtalk_bridge = get_dingtalk_bridge()
        result["details"]["dingtalk_running"] = (
            dingtalk_bridge.is_running if dingtalk_bridge else False
        )

        # 企业微信
        wecom_bridge = get_wecom_bridge()
        result["details"]["wecom_running"] = (
            wecom_bridge.is_running if wecom_bridge else False
        )

        # 综合状态
        if (
            result["details"]["feishu_running"]
            or result["details"]["dingtalk_running"]
            or result["details"]["wecom_running"]
            or result["details"]["feishu_started"]
        ):
            result["status"] = "running"
        else:
            result["status"] = "stopped"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def check_gateway_mode() -> dict[str, Any]:
    """检查 Gateway 模式状态"""
    result = {"mode": "gateway", "status": "not_started", "details": {}}

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from middleware.im.gateway_starter import get_gateway_manager

        manager = get_gateway_manager()

        if manager is None:
            result["status"] = "not_initialized"
            result["details"]["message"] = "GatewayManager not created yet"
        else:
            result["details"]["is_running"] = manager.is_running
            result["details"]["gateway_present"] = manager.gateway is not None

            if manager.is_running:
                result["status"] = "running"

                # 检查已启动的渠道
                if manager.gateway:
                    adapters = getattr(manager.gateway, "_adapters", {})
                    result["details"]["active_channels"] = list(adapters.keys())
                    result["details"]["channel_count"] = len(adapters)
            else:
                result["status"] = "stopped"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def check_configuration() -> dict[str, Any]:
    """检查配置文件中的 IM 配置"""
    result = {"config_file": "", "gateway_enabled": False, "im_platforms": {}}

    config_path = Path.home() / "memento_s" / "config.json"
    result["config_file"] = str(config_path)

    if not config_path.exists():
        result["error"] = "Config file not found"
        return result

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 检查 Gateway 配置
        gateway_cfg = config.get("gateway", {})
        result["gateway_enabled"] = gateway_cfg.get("enabled", False)
        result["gateway_config"] = {
            "mode": gateway_cfg.get("mode", "bridge"),
            "websocket": f"{gateway_cfg.get('websocket_host', '127.0.0.1')}:{gateway_cfg.get('websocket_port', 8765)}",
            "webhook": f"{gateway_cfg.get('webhook_host', '127.0.0.1')}:{gateway_cfg.get('webhook_port', 18080)}",
        }

        # 检查 IM 平台配置
        im_cfg = config.get("im", {})
        platforms = ["feishu", "dingtalk", "wecom"]

        for platform in platforms:
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


def print_status():
    """打印状态报告（使用 Rich 美化）"""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]IM Platform Status Check[/bold cyan]", border_style="cyan"
        )
    )
    console.print()

    # 1. 检查配置
    config = check_configuration()

    config_table = Table(title="[1] Configuration", box=box.ROUNDED)
    config_table.add_column("Item", style="cyan")
    config_table.add_column("Value", style="white")

    config_table.add_row("Config file", config["config_file"])

    if "error" in config:
        config_table.add_row("Error", f"[red]{config['error']}[/red]")
    else:
        config_table.add_row("Gateway enabled", str(config["gateway_enabled"]))
        if config["gateway_enabled"]:
            config_table.add_row("Mode", config["gateway_config"]["mode"])
            config_table.add_row("WebSocket", config["gateway_config"]["websocket"])
            config_table.add_row("Webhook", config["gateway_config"]["webhook"])

        for platform, info in config["im_platforms"].items():
            status = (
                "[green]enabled[/green]" if info["enabled"] else "[red]disabled[/red]"
            )
            configured = "configured" if info["configured"] else "not configured"
            config_table.add_row(f"Platform: {platform}", f"{status} ({configured})")

    console.print(config_table)
    console.print()

    # 2. 检查 Bridge 模式
    bridge = check_bridge_mode()

    bridge_table = Table(title="[2] Bridge Mode", box=box.ROUNDED)
    bridge_table.add_column("Platform", style="cyan")
    bridge_table.add_column("Status", style="white")
    bridge_table.add_column("Sessions", style="white")

    # 飞书
    if bridge["details"].get("feishu_running") or bridge["details"].get("feishu_started"):
        bridge_table.add_row("Feishu", "[green]running[/green]", str(bridge["details"].get("feishu_session_count", 0)))
    else:
        bridge_table.add_row("Feishu", "[red]stopped[/red]", "-")

    # 钉钉
    if bridge["details"].get("dingtalk_running"):
        bridge_table.add_row("DingTalk", "[green]running[/green]", "-")
    else:
        bridge_table.add_row("DingTalk", "[red]stopped[/red]", "-")

    # 企业微信
    if bridge["details"].get("wecom_running"):
        bridge_table.add_row("WeCom", "[green]running[/green]", "-")
    else:
        bridge_table.add_row("WeCom", "[red]stopped[/red]", "-")

    if bridge["status"] == "error":
        bridge_table.add_row("Error", f"[red]{bridge.get('error', 'Unknown')}[/red]", "-")

    console.print(bridge_table)
    console.print()

    # 3. 检查 Gateway 模式
    gateway = check_gateway_mode()

    gateway_table = Table(title="[3] Gateway Mode", box=box.ROUNDED)
    gateway_table.add_column("Item", style="cyan")
    gateway_table.add_column("Value", style="white")

    if gateway["status"] == "running":
        gateway_table.add_row("Status", "[green]running[/green]")
        gateway_table.add_row(
            "Instance",
            "present" if gateway["details"].get("gateway_present") else "missing",
        )
        channels = gateway["details"].get("active_channels", [])
        gateway_table.add_row(
            "Active channels", ", ".join(channels) if channels else "None"
        )
        gateway_table.add_row(
            "Channel count", str(gateway["details"].get("channel_count", 0))
        )
    elif gateway["status"] == "not_initialized":
        gateway_table.add_row("Status", "[yellow]not initialized[/yellow]")
        gateway_table.add_row("To enable", "Set gateway.enabled=true in config")
    else:
        gateway_table.add_row(
            "Status", f"[red]error: {gateway.get('error', 'Unknown')}[/red]"
        )

    console.print(gateway_table)
    console.print()

    # 4. 总结
    console.print(
        Panel.fit(
            "[bold]Summary[/bold]",
            border_style="green"
            if bridge["status"] == "running" or gateway["status"] == "running"
            else "red",
        )
    )

    if bridge["status"] == "running":
        running_platforms = []
        if bridge["details"].get("feishu_running") or bridge["details"].get("feishu_started"):
            running_platforms.append("Feishu")
        if bridge["details"].get("dingtalk_running"):
            running_platforms.append("DingTalk")
        if bridge["details"].get("wecom_running"):
            running_platforms.append("WeCom")
        console.print(f"[green]OK[/green] Bridge mode is RUNNING ({', '.join(running_platforms)})")

    if gateway["status"] == "running":
        console.print("[green]OK[/green] Gateway mode is RUNNING")
        channels = gateway["details"].get("active_channels", [])
        if channels:
            console.print(f"  Active IM platforms: {', '.join(channels)}")

    if bridge["status"] != "running" and gateway["status"] != "running":
        console.print("[red]STOPPED[/red] No IM platform is currently running")
        console.print()
        console.print("[bold]To start:[/bold]")
        console.print("  Feishu:    [cyan]memento feishu[/cyan]")
        console.print("  DingTalk:  [cyan]memento dingtalk[/cyan]")
        console.print("  WeCom:     [cyan]memento wecom[/cyan]")
        console.print(
            "  Gateway mode: [cyan]Set gateway.enabled=true and run memento agent[/cyan]"
        )

    console.print()


def im_status_command() -> None:
    """CLI command: Check IM platform status."""
    print_status()


if __name__ == "__main__":
    im_status_command()
