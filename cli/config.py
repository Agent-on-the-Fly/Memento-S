
import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from dotenv import find_dotenv, set_key, unset_key
from pathlib import Path
from typing import Optional

from core.config import g_settings

console = Console()
config_app = typer.Typer(help="Manage configuration and .env file.", no_args_is_help=False)

PROVIDERS = ["openai", "anthropic", "google", "openrouter", "ollama"]

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "google": "gemini-1.5-pro",
    "openrouter": "anthropic/claude-3.5-sonnet",
    "ollama": "llama3",
}

PROVIDER_ENV_MAP = {
    "openai": {"key": "OPENAI_API_KEY", "base": "OPENAI_API_BASE"},
    "anthropic": {"key": "ANTHROPIC_API_KEY", "base": None},
    "google": {"key": "GOOGLE_API_KEY", "base": None},
    "openrouter": {"key": "OPENROUTER_API_KEY", "base": "OPENROUTER_BASE_URL"},
    "ollama": {"key": 'None', "base": "LLM_BASE_URL"},
}

def _update_env(key: str, value: str):
    env_file = find_dotenv()
    if not env_file:
        env_file = Path.cwd() / ".env"
        env_file.touch()
    
    set_key(str(env_file), key, value)

def _read_env(key: str) -> Optional[str]:
    
    for name, field in g_settings.model_fields.items():
        if field.alias == key:
            val = getattr(g_settings, name)
            return str(val) if val is not None else None
    return None

def _ask_with_list(title: str, choices: list[str], default: str) -> str:
    console.print(f"\n[bold]{title}[/bold]")
    
    default_index = -1
    for idx, choice in enumerate(choices, 1):
        if choice == default:
            default_index = idx
            console.print(f"  [green]{idx}. {choice}[/green] [dim](current)[/dim]")
        else:
            console.print(f"  {idx}. {choice}")
            
    while True:
        default_str = str(default_index) if default_index != -1 else None
        selection = Prompt.ask("Select number", default=default_str)
        
        try:
            idx = int(selection)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
            console.print("[red]Invalid number. Please try again.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")

@config_app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        interactive_setup()

def interactive_setup():
    console.print(Panel("Memento-S Configuration Wizard", style="bold cyan"))
    
    current_provider = g_settings.llm_api
    provider = _ask_with_list(
        "Select LLM Provider", 
        choices=PROVIDERS, 
        default=current_provider
    )
    _update_env("LLM_API", provider)
    
    env_map = PROVIDER_ENV_MAP.get(provider, {})
    key_env = env_map.get("key")
    base_env = env_map.get("base")
    
    if key_env:
        current_key = getattr(g_settings, f"{provider}_api_key", None)
        
        new_key = Prompt.ask(
            f"Enter {key_env}", 
            password=True, 
            default=current_key if current_key else ""
        )
        if new_key:
            _update_env(key_env, new_key)
            
    if base_env:
        current_base = getattr(g_settings, f"{provider}_base_url", None)
        
        new_base = Prompt.ask(
            f"Enter {base_env} (Optional)", 
            default=current_base if current_base else ""
        )
        if new_base:
            _update_env(base_env, new_base)
            
    default_model = DEFAULT_MODELS.get(provider, "gpt-4o")
    current_model = g_settings.llm_model
    if current_model and current_provider != provider:
         suggested_model = default_model
    else:
         suggested_model = current_model or default_model

    model = Prompt.ask(
        "Enter Model Name", 
        default=suggested_model
    )
    _update_env("LLM_MODEL", model)

    if provider == "openai":
        console.print(Panel("OpenAI Specific Configuration", style="bold cyan"))
        
        current_max_tokens = str(g_settings.llm_max_tokens)
        max_tokens = Prompt.ask("LLM Max Tokens", default=current_max_tokens)
        _update_env("LLM_MAX_TOKENS", max_tokens)
        
        current_temp = str(g_settings.llm_temperature)
        temp = Prompt.ask("LLM Temperature", default=current_temp)
        _update_env("LLM_TEMPERATURE", temp)
        
        current_timeout = str(g_settings.llm_timeout)
        timeout = Prompt.ask("LLM Timeout (seconds)", default=current_timeout)
        _update_env("LLM_TIMEOUT", timeout)
    
    change_workspace = Confirm.ask("Configure Workspace paths?", default=False)
    if change_workspace:
        current_ws = str(g_settings.workspace_dir)
        new_ws = Prompt.ask("Workspace Directory (relative)", default=current_ws)
        _update_env("WORKSPACE_DIR", new_ws)

    console.print(Panel("Skills & Search Configuration", style="bold cyan"))
    
    if Confirm.ask("Configure GitHub Token?", default=False):
        current_gh = g_settings.github_token
        gh_token = Prompt.ask("GitHub Token", password=True, default=current_gh)
        _update_env("GITHUB_TOKEN", gh_token)

    from os import getenv
    
    if Confirm.ask("Configure Search (SerpAPI)?", default=False):
        console.print(Panel("SerpAPI Configuration", style="bold cyan"))
        current_serp = getenv("SERPAPI_API_KEY", "")
        serp_key = Prompt.ask("SerpAPI API Key", password=True, default=current_serp)
        _update_env("SERPAPI_API_KEY", serp_key)
        
    if Confirm.ask("Configure Jina API (for skill)?", default=False):
        console.print(Panel("Jina Configuration", style="bold cyan"))
        current_jina = getenv("JINA_API_KEY", "")
        jina_key = Prompt.ask("Jina API Key", password=True, default=current_jina)
        _update_env("JINA_API_KEY", jina_key)

    if Confirm.ask("Configure Advanced Settings (Retrieval, Execution, Strategy)?", default=False):
        current_top_k = str(g_settings.retrieval_top_k)
        top_k = Prompt.ask("Retrieval Top K", default=current_top_k)
        _update_env("RETRIEVAL_TOP_K", top_k)
        
        current_min_score = str(g_settings.retrieval_min_score)
        min_score = Prompt.ask("Retrieval Min Score", default=current_min_score)
        _update_env("RETRIEVAL_MIN_SCORE", min_score)

        current_sandbox = g_settings.sandbox_provider
        sandbox = _ask_with_list(
            "Sandbox Provider", 
            choices=["local", "e2b", "modal"], 
            default=current_sandbox
        )
        _update_env("SANDBOX_PROVIDER", sandbox)
        
        if sandbox == "e2b":
            current_e2b = g_settings.e2b_api_key
            e2b_key = Prompt.ask("E2B API Key", password=True, default=current_e2b)
            _update_env("E2B_API_KEY", e2b_key)

        current_strategy = g_settings.resolve_strategy
        strategy = _ask_with_list(
            "Resolve Strategy", 
            choices=["local_only", "local_first", "always_search"], 
            default=current_strategy
        )
        _update_env("RESOLVE_STRATEGY", strategy)
        
        current_download = g_settings.skill_download_method
        download = _ask_with_list(
            "Skill Download Method", 
            choices=["github_api", "npx", "auto"], 
            default=current_download
        )
        _update_env("SKILL_DOWNLOAD_METHOD", download)
        
    console.print("\n[green]Configuration updated successfully![/green]")
    console.print("[dim]Changes saved to .env[/dim]")

@config_app.command("list")
def list_config():
    table = Table(title="Current Configuration", show_header=True)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Env Var", style="dim")
    
    for key, field in g_settings.model_fields.items():
        value = getattr(g_settings, key)
        env_var = field.alias
        
        val_str = str(value)
        if "key" in key.lower() or "token" in key.lower() or "password" in key.lower():
            if value:
                val_str = f"{str(value)[:4]}...{str(value)[-4:]}" if len(str(value)) > 8 else "***"
            else:
                val_str = "None"
                
        table.add_row(key, val_str, env_var)
        
    console.print(table)

@config_app.command("get")
def get_config(key: str):
    target_field = None
    target_val = None
    
    for name, field in g_settings.model_fields.items():
        if field.alias == key.upper() or name == key.lower():
            target_field = field
            target_val = getattr(g_settings, name)
            break
            
    if target_field:
        console.print(f"[bold cyan]{target_field.alias}[/bold cyan] ({key}): [green]{target_val}[/green]")
    else:
        console.print(f"[red]Configuration key '{key}' not found.[/red]")

@config_app.command("set")
def set_config(key: str, value: str):
    valid_keys = [f.alias for f in g_settings.model_fields.values()]
    if key not in valid_keys:
        found = False
        for name, field in g_settings.model_fields.items():
            if name == key:
                key = field.alias
                found = True
                break
        if not found:
            console.print(f"[yellow]Warning: '{key}' is not a standard Memento-S config key.[/yellow]")
            if not Confirm.ask("Set it anyway?"):
                return

    _update_env(key, value)
    console.print(f"[green]Set {key}={value}[/green]")

@config_app.command("unset")
def unset_config(key: str):
    env_file = find_dotenv()
    if not env_file:
        console.print("[red]No .env file found.[/red]")
        return
        
    unset_key(str(env_file), key)
    console.print(f"[green]Unset {key}[/green]")
