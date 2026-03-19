"""Doctor command for displaying configuration and environment info."""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Handle project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from middleware.config import g_config

console = Console()

# Configuration section display order and labels
SECTION_LABELS = {
    "app": "📱 Application",
    "llm": "🤖 LLM Configuration",
    "providers": "🔌 Providers",
    "search": "🔍 Search",
    "skills": "🛠️  Skills",
    "paths": "📁 Paths",
    "logging": "📝 Logging",
    "agent": "🎯 Agent",
}

IMPORTANT_KEYS = {
    "llm.active_profile": "Active Profile",
    "llm.current.model": "Current Model",
    "llm.current.provider": "Provider",
    "llm.current.temperature": "Temperature",
    "llm.current.context_window": "Context Window",
    "llm.current.max_tokens": "Max Output Tokens",
    "llm.current.timeout": "Timeout",
    "app.name": "App Name",
    "app.version": "Version",
}


def _get_nested_value(data: dict, key_path: str):
    """Get nested value from dict using dot notation."""
    parts = key_path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _create_section_table(section_name: str, section_data: dict, parent_key: str = ""):
    """Create a styled table for a configuration section."""
    table = Table(
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Key", style="cyan", min_width=25)
    table.add_column("Value", style="green", min_width=30)

    items = []
    for key, value in section_data.items():
        full_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, dict):
            if section_name == "providers" and key:
                items.append((f"[bold]{key}[/bold]", ""))
                for sub_key, sub_val in value.items():
                    items.append((f"  {sub_key}", str(sub_val)))
            elif section_name == "llm" and key == "profiles":
                items.append((f"[bold]{key}[/bold]", f"{len(value)} profile(s)"))
                for profile_name in value.keys():
                    items.append((f"  • {profile_name}", ""))
            elif section_name == "llm" and key == "current":
                for sub_key, sub_val in value.items():
                    display_key = f"[bold yellow]{sub_key}[/bold yellow]"
                    items.append((display_key, str(sub_val)))
            else:
                items.append((f"[bold]{key}[/bold]", ""))
                for sub_key, sub_val in value.items():
                    items.append((f"  {sub_key}", str(sub_val)))
        else:
            if full_key in IMPORTANT_KEYS:
                key_display = f"[bold bright_cyan]{key}[/bold bright_cyan]"
            else:
                key_display = key
            items.append((key_display, str(value)))

    for key, value in items:
        table.add_row(key, value)

    return table


def _create_summary_panel(data: dict):
    """Create a summary panel with key information."""
    active_profile = _get_nested_value(data, "llm.active_profile") or "N/A"
    current_model = _get_nested_value(data, "llm.current.model") or "N/A"
    provider = _get_nested_value(data, "llm.current.provider") or "N/A"
    app_name = _get_nested_value(data, "app.name") or "N/A"
    version = _get_nested_value(data, "app.version") or "N/A"

    summary_text = f"""\
[bold cyan]Application:[/bold cyan] {app_name} v{version}

[bold cyan]Current LLM:[/bold cyan]
  • Profile: [green]{active_profile}[/green]
  • Model: [green]{current_model}[/green]
  • Provider: [green]{provider}[/green]
"""

    return Panel(
        summary_text,
        title="[bold blue]Configuration Summary[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    )


def doctor_command() -> None:
    """Print configuration and environment info with formatted display."""
    console.print()
    console.print(
        Panel.fit(
            "[bold white]Memento-S Doctor[/bold white]",
            border_style="bright_blue",
            padding=(1, 4),
        )
    )
    console.print()

    cfg = g_config

    # Print environment check
    ok, no = "[green]✓[/green]", "[red]✗[/red]"
    workspace = cfg.paths.workspace_dir
    cache_dir = cfg.paths.cache_dir
    tmp_dir = cfg.paths.tmp_dir
    console.print("[bold]Environment Check[/bold]")
    console.print(
        f"  Workspace:     {workspace} {ok if workspace and workspace.exists() else no}"
    )
    console.print(
        f"  Cache:         {cache_dir} {ok if cache_dir and cache_dir.exists() else no}"
    )
    console.print(
        f"  Tmp:           {tmp_dir} {ok if tmp_dir and tmp_dir.exists() else no}"
    )
    console.print()

    # Print formatted configuration
    data = cfg.to_json_dict()

    # Print summary panel
    console.print(_create_summary_panel(data))
    console.print()

    # Print sections in order
    for section_key, section_label in SECTION_LABELS.items():
        if section_key in data:
            section_data = data[section_key]
            if not section_data:
                continue

            table = _create_section_table(section_key, section_data, section_key)
            panel = Panel(
                table,
                title=f"[bold]{section_label}[/bold]",
                title_align="left",
                border_style="dim cyan",
                padding=(1, 1),
            )
            console.print(panel)
            console.print()

    # Print any other sections not in the predefined list
    other_sections = {
        k: v
        for k, v in data.items()
        if k not in SECTION_LABELS and k not in ("$schema", "version")
    }
    if other_sections:
        for section_key, section_data in other_sections.items():
            if not section_data:
                continue

            table = _create_section_table(section_key, section_data, section_key)
            panel = Panel(
                table,
                title=f"[bold]{section_key.title()}[/bold]",
                title_align="left",
                border_style="dim",
                padding=(1, 1),
            )
            console.print(panel)
            console.print()
