"""WeChat CLI commands for Memento-S.

微信管理命令：
  - login: 扫码登录获取 token
  - status: 查看微信连接状态
  - logout: 退出登录

用法：
    memento wechat --help
    memento wechat login
    memento wechat status
    memento wechat logout
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

# Project root bootstrap
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from middleware.config import g_config
from utils.logger import get_logger

console = Console()
logger = get_logger(__name__)

wechat_app = typer.Typer(name="wechat", help="WeChat management commands")


def _get_wechat_config_path() -> Path:
    """获取主配置文件路径。"""
    # Config is in parent of workspace_dir
    workspace = Path(g_config.paths.workspace_dir)
    return workspace.parent / "config.json"


def _load_config() -> dict:
    """加载当前主配置。"""
    config_path = _get_wechat_config_path()
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"无法加载配置: {e}")
    return {}


def _save_config(config: dict) -> None:
    """保存配置到主配置文件。"""
    config_path = _get_wechat_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@wechat_app.command()
def login(
    account_id: str = typer.Option("personal", "--account", "-a", help="Account ID"),
    base_url: str = typer.Option(
        "https://ilinkai.weixin.qq.com", "--base-url", "-b", help="API Base URL"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-login even if already logged in"
    ),
) -> None:
    """微信扫码登录，获取 token 并保存到配置。"""
    console.print(
        Panel.fit(
            "[bold cyan]WeChat Login[/bold cyan]\n"
            "Scan QR code to authenticate with WeChat",
            title="Memento-S",
            border_style="cyan",
        )
    )

    # Check if SDK is available
    # First try local 3rd party version
    _3RD_DIR = _PROJECT_ROOT / "3rd"
    if str(_3RD_DIR) not in sys.path:
        sys.path.insert(0, str(_3RD_DIR))

    try:
        from weixin_sdk.auth.qr_login import QRLoginManager
        from weixin_sdk import DEFAULT_BASE_URL

        logger.info("Using weixin_sdk from local 3rd/ directory")
    except ImportError:
        console.print("[red]Error: weixin_sdk not found.[/red]")
        console.print("[dim]Please ensure 3rd/weixin_sdk exists or install with:[/dim]")
        console.print(
            "  uv pip install -e C:\\Users\\75484\\src\\openclaw-weixin-python"
        )
        raise typer.Exit(1)

    # Check if already logged in
    config = _load_config()
    wechat_cfg = config.get("im", {}).get("wechat", {})
    existing_token = wechat_cfg.get("token") if isinstance(wechat_cfg, dict) else None

    if existing_token and not force:
        console.print("[yellow]Already logged in. Use --force to re-login.[/yellow]")
        if not Confirm.ask("Do you want to re-login?"):
            console.print("[dim]Login cancelled.[/dim]")
            return

    async def _do_login():
        """执行登录流程。"""
        mgr = QRLoginManager(base_url)

        try:
            console.print("\n[dim]Starting login process...[/dim]")
            result = await mgr.start_login(
                timeout_seconds=300,  # 5 minutes
                max_qr_refreshes=3,
                poll_interval=1.0,
            )

            if not result.bot_token:
                console.print("[red]Login failed: No token received[/red]")
                raise typer.Exit(1)

            # Save to config (扁平结构，与其他平台一致)
            config = _load_config()

            # Ensure structure exists
            if "im" not in config:
                config["im"] = {}

            config["im"]["wechat"] = {
                "enabled": True,
                "base_url": base_url,
                "token": result.bot_token,
            }

            _save_config(config)

            # Success output
            console.print("\n" + "=" * 60)
            console.print("[bold green]✓ Login successful![/bold green]")
            console.print("=" * 60)
            console.print(f"[dim]Account ID:[/dim]   {account_id}")
            console.print(
                f"[dim]Bot ID:[/dim]       {result.ilink_bot_id if hasattr(result, 'ilink_bot_id') else 'N/A'}"
            )
            console.print(f"[dim]Token saved to:[/dim] {_get_wechat_config_path()}")
            console.print("\n[dim]You can now start the gateway with:[/dim]")
            console.print("  [cyan]memento gateway-worker[/cyan]")

        except asyncio.TimeoutError:
            console.print("\n[red]✗ Login timeout (5 minutes)[/red]")
            console.print("[dim]Please try again.[/dim]")
            raise typer.Exit(1)

        except Exception as e:
            error_msg = str(e)
            console.print(f"\n[red]✗ Login failed: {error_msg}[/red]")

            if "404" in error_msg:
                console.print(
                    "\n[yellow]Note:[/yellow] API returned 404. Possible causes:"
                )
                console.print("  - Network connectivity issues")
                console.print("  - API endpoint not accessible from your location")
                console.print(
                    "  - Alternative: Copy token from openclaw-weixin project"
                )

            raise typer.Exit(1)

    # Run async login
    try:
        asyncio.run(_do_login())
    except KeyboardInterrupt:
        console.print("\n\n[dim]Login cancelled by user.[/dim]")
        raise typer.Exit(0)


async def _verify_token(base_url: str, token: str) -> tuple[bool, str]:
    """验证 token 是否有效。

    简化的验证方法：尝试初始化客户端并获取配置。
    如果成功，说明 token 有效。

    Returns:
        (is_valid, message): 是否有效及状态消息
    """
    # Add 3rd party SDK to path
    _3RD_DIR = _PROJECT_ROOT / "3rd"
    if str(_3RD_DIR) not in sys.path:
        sys.path.insert(0, str(_3RD_DIR))

    try:
        from weixin_sdk import WeixinClient
        from weixin_sdk.exceptions import WeixinSessionExpiredError, WeixinAPIError

        # 创建客户端并尝试获取配置
        client = WeixinClient(base_url=base_url, token=token)

        try:
            # 尝试获取配置来验证 token
            # 这会实际调用 API 验证 token 是否有效
            await client.get_config()
            return True, "Token 有效"
        except WeixinSessionExpiredError:
            return False, "Token 已过期"
        except WeixinAPIError as e:
            if e.code == -14 or (e.response and e.response.get("errcode") == -14):
                return False, "Token 已过期"
            elif e.code == 401 or e.code == 403:
                return False, "Token 无效"
            else:
                return False, f"API 错误 (code: {e.code})"
        except Exception as e:
            # 其他错误，可能是网络问题
            error_str = str(e).lower()
            if "unauthorized" in error_str or "auth" in error_str:
                return False, "Token 无效或已过期"
            elif "timeout" in error_str or "connect" in error_str:
                return True, "Token 已配置 (网络问题无法验证)"
            else:
                # 无法确定，假设有效（让 Gateway 启动后实际测试）
                return True, "Token 已配置"

    except ImportError:
        # SDK 不可用，跳过验证
        return True, "无法验证 (SDK 未安装)"


@wechat_app.command()
def status(
    verify: bool = typer.Option(
        True, "--verify/--no-verify", help="验证 token 有效性（默认启用）"
    ),
) -> None:
    """查看微信连接状态，可选验证 token 有效性。"""
    config = _load_config()

    wechat_config = config.get("im", {}).get("wechat", {})

    if not wechat_config:
        console.print(
            Panel(
                "[yellow]WeChat not configured[/yellow]\n\n"
                "Run [cyan]memento wechat login[/cyan] to set up.",
                title="Status",
                border_style="yellow",
            )
        )
        return

    # 扁平结构：直接读取字段
    enabled = (
        wechat_config.get("enabled", False)
        if isinstance(wechat_config, dict)
        else False
    )
    token = wechat_config.get("token", "") if isinstance(wechat_config, dict) else ""
    base_url = (
        wechat_config.get("base_url", "https://ilinkai.weixin.qq.com")
        if isinstance(wechat_config, dict)
        else "https://ilinkai.weixin.qq.com"
    )

    console.print(
        Panel.fit(
            f"[bold {'green' if enabled else 'red'}]"
            f"{'● Enabled' if enabled else '○ Disabled'}[/bold {'green' if enabled else 'red'}]",
            title="WeChat Status",
            border_style="cyan",
        )
    )

    # 显示配置状态
    has_token = bool(token)

    if verify and has_token:
        console.print("[dim]Verifying token...[/dim]\n")
        try:
            is_valid, message = asyncio.run(_verify_token(base_url, token))
            if is_valid:
                status_icon = "[green]✓[/green]"
                status_text = f"[green]{message}[/green]"
            else:
                status_icon = "[red]✗[/red]"
                status_text = f"[red]{message}[/red]"
            console.print(f"  {status_icon} Token: {status_text}")
        except Exception as e:
            console.print(
                f"  [yellow]?[/yellow] Token: [yellow]验证失败 - {str(e)[:30]}[/yellow]"
            )
    elif has_token:
        # 不验证，仅显示 token 存在
        console.print(f"  [green]✓[/green] Token: [dim]configured (not verified)[/dim]")
    else:
        console.print(f"  [red]✗[/red] Token: [red]not configured[/red]")

    # 提示信息
    if verify:
        console.print("\n[dim]Tip: Use --no-verify to skip token validation[/dim]")

    # 如果验证失败，提示重新登录
    if verify and has_token:
        try:
            is_valid, _ = asyncio.run(_verify_token(base_url, token))
            if not is_valid:
                console.print("\n[yellow]⚠ Token 验证失败，可能需要重新登录:[/yellow]")
                console.print("  [cyan]memento wechat login --force[/cyan]")
        except:
            pass


@wechat_app.command()
def logout(
    force: bool = typer.Option(
        False, "--force", "-f", help="Force logout without confirmation"
    ),
) -> None:
    """退出微信登录，清除保存的 token。"""
    config = _load_config()

    wechat_config = config.get("im", {}).get("wechat", {})

    if not isinstance(wechat_config, dict):
        console.print("[yellow]No WeChat configuration found.[/yellow]")
        return

    token = wechat_config.get("token", "")

    if not token:
        console.print("[yellow]Not logged in.[/yellow]")
        return

    if force or Confirm.ask("Logout WeChat?"):
        # 清除 token（扁平结构）
        config["im"]["wechat"]["token"] = None
        config["im"]["wechat"]["enabled"] = False
        _save_config(config)
        console.print("[green]✓ Logged out successfully.[/green]")
    else:
        console.print("[dim]Cancelled.[/dim]")


# Legacy command for backward compatibility
def wechat_bridge_command() -> None:
    """Start WeChat bridge (deprecated, use gateway-worker instead)."""
    console.print(
        Panel(
            "[yellow]WeChat bridge command is deprecated.[/yellow]\n\n"
            "Use [cyan]memento gateway-worker[/cyan] to start the gateway\n"
            "with WeChat support.",
            title="Deprecated",
            border_style="yellow",
        )
    )
    console.print("\n[dim]Make sure you have:[/dim]")
    console.print("  1. Run [cyan]memento wechat login[/cyan] to authenticate")
    console.print("  2. Run [cyan]memento gateway-worker[/cyan] to start")
