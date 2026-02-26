"""Memento-S CLI — multi-turn conversation powered by MCPAgent."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.key_binding import KeyBindings

    PROMPT_TOOLKIT_AVAILABLE = True
except Exception:
    PromptSession = None  # type: ignore[assignment]
    Completer = object  # type: ignore[assignment]
    Completion = None  # type: ignore[assignment]
    KeyBindings = None  # type: ignore[assignment]
    PROMPT_TOOLKIT_AVAILABLE = False

from core.config import (
    DEBUG,
    PROJECT_ROOT,
    SEMANTIC_ROUTER_DEBUG,
    SEMANTIC_ROUTER_ENABLED,
    WORKSPACE_DIR,
    refresh_runtime_config,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}
HELP_COMMANDS = {"help", "/help"}
CLEAR_COMMANDS = {"clear", "/clear"}
STATUS_COMMANDS = {"status", "/status"}
SKILLS_COMMANDS = {"skills", "/skills"}
HISTORY_COMMANDS = {"history", "/history"}
CONFIG_COMMANDS = {"config", "/config"}

DEFAULT_HISTORY_FILE = Path(".agent/cli_history.json")
DEFAULT_ENV_FILE = (PROJECT_ROOT / ".env").resolve()
SESSION_MESSAGE_LIMIT = 200
INTERNAL_TURN_LIMIT = 200
SESSION_STORE_LIMIT = 200
ANSI_CYAN = "\033[0;36m"
ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
ANSI_YELLOW = "\033[0;33m"
ANSI_GREEN = "\033[0;32m"
ANSI_MAGENTA = "\033[0;35m"
ANSI_BOLD = "\033[1m"

MODEL_RELATED_KEYS = {
    "OPENROUTER_MODEL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MAX_TOKENS",
    "OPENROUTER_TIMEOUT",
}

CONFIG_KEY_ALIASES: dict[str, str] = {
    "api": "LLM_API",
    "model": "OPENROUTER_MODEL",
    "key": "OPENROUTER_API_KEY",
    "base_url": "OPENROUTER_BASE_URL",
    "timeout": "OPENROUTER_TIMEOUT",
    "max_tokens": "OPENROUTER_MAX_TOKENS",
    "retries": "OPENROUTER_RETRIES",
    "provider": "OPENROUTER_PROVIDER",
    "provider_order": "OPENROUTER_PROVIDER_ORDER",
    "allow_fallbacks": "OPENROUTER_ALLOW_FALLBACKS",
    "site_url": "OPENROUTER_SITE_URL",
    "app_name": "OPENROUTER_APP_NAME",
    "serpapi": "SERPAPI_API_KEY",
}
CONFIG_KEYS: tuple[str, ...] = (
    "LLM_API",
    "OPENROUTER_MODEL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_TIMEOUT",
    "OPENROUTER_MAX_TOKENS",
    "OPENROUTER_RETRIES",
    "OPENROUTER_RETRY_BACKOFF",
    "OPENROUTER_PROVIDER",
    "OPENROUTER_PROVIDER_ORDER",
    "OPENROUTER_ALLOW_FALLBACKS",
    "OPENROUTER_SITE_URL",
    "OPENROUTER_APP_NAME",
    "SERPAPI_API_KEY",
)
CONFIG_KEYS_SET = set(CONFIG_KEYS)
CONFIG_SECRET_KEYS = {"OPENROUTER_API_KEY", "SERPAPI_API_KEY", "OPENAI_API_KEY"}
CONFIG_ATTR_OVERRIDES = {"OPENROUTER_MODEL": "MODEL"}

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show this help"),
    ("/status", "Show session status"),
    ("/skills [query] [-n N]", "Search cloud skills or list local skills"),
    ("/config [show|get|set|unset]", "View/update .env config (api/model/etc.)"),
    ("/history [N]", "Show session history window"),
    ("/history load <index>", "Load one saved session into current context"),
    ("/clear", "Clear conversation context/history"),
    ("/exit", "Exit the CLI"),
]
KNOWN_SLASH_COMMANDS = {cmd.split()[0].strip() for cmd, _ in SLASH_COMMANDS if cmd.split()}


# ---------------------------------------------------------------------------
# TurnInterrupted
# ---------------------------------------------------------------------------

class TurnInterrupted(Exception):
    """Raised when a running turn is interrupted, with optional partial output."""

    def __init__(self, partial_reply: str = "") -> None:
        super().__init__("turn interrupted")
        self.partial_reply = str(partial_reply or "")


# ---------------------------------------------------------------------------
# Prompt toolkit helpers
# ---------------------------------------------------------------------------

class SlashCommandCompleter(Completer):
    """Autocomplete slash commands while typing."""

    def get_completions(self, document, complete_event):  # type: ignore[override]
        text = str(document.text_before_cursor or "")
        stripped = text.lstrip()
        if not stripped.startswith("/"):
            return
        token = stripped.split(maxsplit=1)[0]
        seen: set[str] = set()
        for cmd, desc in SLASH_COMMANDS:
            base = cmd.split()[0].strip()
            if not base or base in seen:
                continue
            if token == "/" or base.startswith(token):
                seen.add(base)
                yield Completion(
                    base,
                    start_position=-len(token),
                    display=base,
                    display_meta=desc,
                )


def _build_prompt_session():
    if not PROMPT_TOOLKIT_AVAILABLE:
        return None
    try:
        kb = KeyBindings()

        @kb.add("/")
        def _slash_autocomplete(event):
            buf = event.app.current_buffer
            buf.insert_text("/")
            try:
                text = str(buf.document.text_before_cursor or "")
                token = text.lstrip().split(maxsplit=1)[0] if text.lstrip() else ""
                if token == "/":
                    buf.start_completion(select_first=False)
            except Exception:
                return

        return PromptSession(
            completer=SlashCommandCompleter(),
            complete_while_typing=True,
            reserve_space_for_menu=10,
            key_bindings=kb,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _split_shell_tokens(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    try:
        return shlex.split(raw)
    except Exception:
        return raw.split()


def _sanitize_history_items(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            out.append({"role": role, "content": content})
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_session_title(first_query: str, *, max_len: int = 80) -> str:
    one_line = " ".join(str(first_query or "").split())
    if not one_line:
        return "Untitled Session"
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 3].rstrip() + "..."


def _new_session() -> dict[str, Any]:
    return {
        "id": f"session-{uuid4().hex[:12]}",
        "title": "",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "messages": [],
        "internal_turns": [],
    }


def _sanitize_session(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    session_id = str(raw.get("id") or "").strip() or f"session-{uuid4().hex[:12]}"
    title = str(raw.get("title") or "").strip()
    created_at = str(raw.get("created_at") or "").strip() or _now_iso()
    updated_at = str(raw.get("updated_at") or "").strip() or created_at
    messages = _sanitize_history_items(raw.get("messages"))
    internal_turns_raw = raw.get("internal_turns")
    internal_turns: list[dict[str, Any]] = []
    if isinstance(internal_turns_raw, list):
        for item in internal_turns_raw:
            if not isinstance(item, dict):
                continue
            user_text = str(item.get("user") or "").strip()
            assistant_text = str(item.get("assistant") or "").strip()
            if not user_text and not assistant_text:
                continue
            internal_turns.append(
                {
                    "ts": str(item.get("ts") or "").strip() or _now_iso(),
                    "user": user_text,
                    "assistant": assistant_text,
                    "interrupted": bool(item.get("interrupted")),
                }
            )

    return {
        "id": session_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "messages": messages[-SESSION_MESSAGE_LIMIT:],
        "internal_turns": internal_turns[-INTERNAL_TURN_LIMIT:],
    }


# ---------------------------------------------------------------------------
# History store persistence
# ---------------------------------------------------------------------------

def _load_history_store(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {"sessions": []}
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            legacy_messages = _sanitize_history_items(raw)[-SESSION_MESSAGE_LIMIT:]
            if not legacy_messages:
                return {"sessions": []}
            first_user = next(
                (m.get("content", "") for m in legacy_messages if m.get("role") == "user"),
                "",
            )
            legacy_session = {
                "id": "legacy-flat-history",
                "title": _build_session_title(first_user) if first_user else "Legacy Session",
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "messages": legacy_messages,
                "internal_turns": [],
            }
            return {"sessions": [legacy_session]}

        sessions_out: list[dict[str, Any]] = []
        if isinstance(raw, dict) and isinstance(raw.get("sessions"), list):
            for item in raw["sessions"]:
                session = _sanitize_session(item)
                if session is not None:
                    sessions_out.append(session)
        return {"sessions": sessions_out[-SESSION_STORE_LIMIT:]}
    except Exception:
        return {"sessions": []}


def _save_history_store(path: Path, store: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        return


def _upsert_session(store: dict[str, Any], session: dict[str, Any]) -> None:
    sessions_raw = store.get("sessions")
    sessions = list(sessions_raw) if isinstance(sessions_raw, list) else []
    sid = str(session.get("id") or "").strip()
    replaced = False
    for idx, item in enumerate(sessions):
        if isinstance(item, dict) and str(item.get("id") or "").strip() == sid:
            sessions[idx] = session
            replaced = True
            break
    if not replaced:
        sessions.append(session)
    store["sessions"] = sessions[-SESSION_STORE_LIMIT:]


# ---------------------------------------------------------------------------
# .env config helpers
# ---------------------------------------------------------------------------

def _parse_env_assignment_line(line: str) -> tuple[str, str] | None:
    text = str(line or "")
    stripped = text.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[7:].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, value.strip()


def _strip_env_quotes(value: str) -> str:
    s = str(value or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {"'", '"'}:
        return s[1:-1]
    return s


def _read_env_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        if not path.exists():
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_assignment_line(line)
            if not parsed:
                continue
            key, value = parsed
            out[key] = _strip_env_quotes(value)
    except Exception:
        return {}
    return out


def _format_env_value(value: str) -> str:
    text = str(value or "")
    if not text:
        return ""
    if any(ch.isspace() for ch in text) or "#" in text:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _upsert_env_key(path: Path, key: str, value: str) -> None:
    lines: list[str] = []
    try:
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = []

    rendered = f"{key}={_format_env_value(value)}"
    out: list[str] = []
    replaced = False
    for line in lines:
        parsed = _parse_env_assignment_line(line)
        if parsed and parsed[0] == key:
            if not replaced:
                out.append(rendered)
                replaced = True
            continue
        out.append(line)
    if not replaced:
        if out and out[-1].strip():
            out.append("")
        out.append(rendered)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")


def _unset_env_key(path: Path, key: str) -> bool:
    try:
        if not path.exists():
            return False
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False

    out: list[str] = []
    removed = False
    for line in lines:
        parsed = _parse_env_assignment_line(line)
        if parsed and parsed[0] == key:
            removed = True
            continue
        out.append(line)

    if removed:
        path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
    return removed


def _normalize_config_key(raw: str) -> str | None:
    token = str(raw or "").strip()
    if not token:
        return None
    alias_hit = CONFIG_KEY_ALIASES.get(token.lower())
    if alias_hit:
        return alias_hit
    direct = token.upper()
    if direct in CONFIG_KEYS_SET:
        return direct
    return None


def _mask_config_value(key: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "(unset)"
    if key in CONFIG_SECRET_KEYS:
        if len(text) <= 8:
            return "*" * len(text)
        return f"{text[:4]}...{text[-4:]}"
    return text


def _effective_config_value(key: str) -> str:
    try:
        import core.config as cfg

        attr = CONFIG_ATTR_OVERRIDES.get(key, key)
        if hasattr(cfg, attr):
            return str(getattr(cfg, attr))
    except Exception:
        pass
    return str(os.getenv(key, "") or "")


def _reload_runtime_config_modules() -> tuple[bool, str]:
    try:
        refresh_runtime_config(override=True)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _print_config_help() -> None:
    print("Usage:")
    print("  /config")
    print("  /config show")
    print("  /config get <key|alias>")
    print("  /config set <key|alias> <value>")
    print("  /config unset <key|alias>")
    print("  /config path")
    print(
        "Aliases: api, model, key, base_url, timeout, max_tokens, retries, serpapi, "
        "provider, provider_order, allow_fallbacks, site_url, app_name"
    )
    print()


def _print_config_show(env_path: Path) -> None:
    saved = _read_env_map(env_path)
    print(f"Config file: {env_path}")
    print("Editable config:")
    for key in CONFIG_KEYS:
        effective = _effective_config_value(key)
        source = "file" if key in saved else "runtime/default"
        print(f"  {key:<25} {_mask_config_value(key, effective):<30} [{source}]")
    print()


def _handle_config_command(raw_args: str, *, env_path: Path) -> str | None:
    """Handle a /config command. Returns the changed key name if model-related, else None."""
    text = str(raw_args or "").strip()
    if not text:
        _print_config_show(env_path)
        _print_config_help()
        return None

    tokens = _split_shell_tokens(text)
    if not tokens:
        _print_config_show(env_path)
        _print_config_help()
        return None

    action = str(tokens[0] or "").strip().lower()
    if action in {"show", "list"}:
        _print_config_show(env_path)
        return None
    if action == "path":
        print(f"{env_path}\n")
        return None

    if action == "get":
        if len(tokens) < 2:
            print("Usage: /config get <key|alias>\n")
            return None
        key = _normalize_config_key(tokens[1])
        if not key:
            print(f"Unsupported key/alias: {tokens[1]!r}\n")
            return None
        value = _effective_config_value(key)
        print(f"{key}={_mask_config_value(key, value)}\n")
        return None

    if action == "set":
        if len(tokens) < 3:
            print("Usage: /config set <key|alias> <value>\n")
            return None
        key = _normalize_config_key(tokens[1])
        if not key:
            print(f"Unsupported key/alias: {tokens[1]!r}\n")
            return None
        value = " ".join(tokens[2:]).strip()
        _upsert_env_key(env_path, key, value)
        os.environ[key] = value
        reloaded, err = _reload_runtime_config_modules()
        if reloaded:
            print(f"Saved: {key}={_mask_config_value(key, value)} (runtime refreshed)\n")
        else:
            print(f"Saved: {key}={_mask_config_value(key, value)}")
            print(f"Runtime refresh warning: {err}")
            print("Restart CLI if the new value does not take effect.\n")
        return key if key in MODEL_RELATED_KEYS else None

    if action == "unset":
        if len(tokens) < 2:
            print("Usage: /config unset <key|alias>\n")
            return None
        key = _normalize_config_key(tokens[1])
        if not key:
            print(f"Unsupported key/alias: {tokens[1]!r}\n")
            return None
        removed = _unset_env_key(env_path, key)
        os.environ.pop(key, None)
        reloaded, err = _reload_runtime_config_modules()
        if removed:
            if reloaded:
                print(f"Removed: {key} (runtime refreshed)\n")
            else:
                print(f"Removed: {key}")
                print(f"Runtime refresh warning: {err}")
                print("Restart CLI if the new value does not take effect.\n")
        else:
            print(f"{key} was not set in {env_path}\n")
        return key if key in MODEL_RELATED_KEYS else None

    _print_config_help()
    return None


# ---------------------------------------------------------------------------
# /skills helpers
# ---------------------------------------------------------------------------

def _parse_skills_args(raw: str, *, default_limit: int = 5) -> tuple[str, int, str | None]:
    text = str(raw or "").strip()
    if not text:
        return "", default_limit, None
    tokens = _split_shell_tokens(text)

    limit = max(1, int(default_limit))
    query_tokens: list[str] = []

    i = 0
    while i < len(tokens):
        tok = str(tokens[i] or "").strip()
        if tok in {"-n", "--n", "--num", "--limit"}:
            if i + 1 >= len(tokens):
                return "", limit, "missing value for -n/--limit"
            val = str(tokens[i + 1] or "").strip()
            try:
                n = int(val)
            except Exception:
                return "", limit, f"invalid number for -n: {val!r}"
            if n <= 0:
                return "", limit, "-n must be >= 1"
            limit = min(n, 50)
            i += 2
            continue
        if tok.startswith("-n=") or tok.startswith("--n=") or tok.startswith("--num=") or tok.startswith("--limit="):
            _, val = tok.split("=", 1)
            val = str(val or "").strip()
            try:
                n = int(val)
            except Exception:
                return "", limit, f"invalid number for -n: {val!r}"
            if n <= 0:
                return "", limit, "-n must be >= 1"
            limit = min(n, 50)
            i += 1
            continue
        query_tokens.append(tok)
        i += 1

    return " ".join(query_tokens).strip(), limit, None


def _print_cloud_skills(query: str, *, top_k: int = 5) -> None:
    from cli.skill_search import load_cloud_skill_catalog, search_cloud_skills

    q = str(query or "").strip()
    entries, meta = load_cloud_skill_catalog()
    if not entries:
        err = str(meta.get("error") or "cloud catalog unavailable").strip()
        ref = str(meta.get("catalog_ref") or "").strip()
        print(f"Cloud skills unavailable: {err}")
        if ref:
            print(f"catalog: {ref}")
        print("Tip: set `SKILL_DYNAMIC_FETCH_CATALOG_JSONL` to a JSONL file path or https URL.")
        print()
        return

    results = search_cloud_skills(q, entries, top_k=top_k)
    if not results:
        print(f"Cloud skills: no matches for query `{q}`.\n")
        return

    source = str(meta.get("source") or "unknown").strip()
    cached = bool(meta.get("cached"))
    stale = bool(meta.get("stale"))
    flags = []
    if cached:
        flags.append("cached")
    if stale:
        flags.append("stale")
    flag_text = f" ({', '.join(flags)})" if flags else ""
    title = q or "(top skills)"
    print(f"Cloud Skills for `{title}`: {len(results)} result(s)  [source={source}{flag_text}, n={top_k}]")
    for idx, item in enumerate(results, 1):
        name = str(item.get("name") or "").strip() or "(unknown)"
        desc = str(item.get("description") or "").strip()
        stars = int(item.get("stars") or 0)
        author = str(item.get("author") or "").strip()
        github_url = str(item.get("githubUrl") or "").strip()

        meta_bits = [f"stars={stars}"]
        if author:
            meta_bits.append(f"author: {author}")

        print(f"{idx}. {name}  ({', '.join(meta_bits)})")
        if desc:
            print(f"   {desc}")
        if github_url:
            print(f"   github: {github_url}")
    print()


# ---------------------------------------------------------------------------
# History display
# ---------------------------------------------------------------------------

def _print_history_window(sessions: list[dict[str, Any]], limit: int = 12) -> None:
    total = len(sessions)
    if total == 0:
        print("+--------------------------------------------------+")
        print("| Session History                                  |")
        print("+--------------------------------------------------+")
        print("| No saved sessions yet.                           |")
        print("+--------------------------------------------------+\n")
        return

    limit = max(1, min(int(limit), 200))
    shown = sessions[-limit:]
    start_idx = total - len(shown) + 1
    width = max(80, min(140, shutil.get_terminal_size((100, 20)).columns))
    inner = width - 2
    text_width = inner - 2

    panel_title = f" Session History (showing {len(shown)}/{total}) "
    panel_title = panel_title[:inner]

    print("+" + "-" * (width - 2) + "+")
    print("|" + panel_title.ljust(inner) + "|")
    print("+" + "-" * (width - 2) + "+")

    for idx, session in enumerate(shown, start=start_idx):
        title = str(session.get("title") or "Untitled Session").strip() or "Untitled Session"
        session_id = str(session.get("id") or "").strip()
        updated_at = str(session.get("updated_at") or "").strip()
        messages = session.get("messages")
        msg_count = len(messages) if isinstance(messages, list) else 0
        turns = msg_count // 2

        header = f"[{idx}] {title}"
        print("| " + header[:text_width].ljust(text_width) + " |")
        meta = f"id={session_id}  turns={turns}  messages={msg_count}  updated={updated_at}"
        wrapped_meta = textwrap.wrap(
            meta,
            width=max(20, text_width),
            replace_whitespace=True,
            drop_whitespace=True,
            break_long_words=False,
        ) or ["(empty)"]
        for line in wrapped_meta:
            print("| " + line[:text_width].ljust(text_width) + " |")
        print("| " + ("-" * text_width) + " |")

    print("+" + "-" * (width - 2) + "+\n")


def _collect_history_sessions(
    store: dict[str, Any],
    *,
    active_session: dict[str, Any],
    history: list[dict[str, str]],
) -> list[dict[str, Any]]:
    sessions_raw = store.get("sessions")
    sessions = list(sessions_raw) if isinstance(sessions_raw, list) else []
    active_id = str(active_session.get("id") or "").strip()
    has_active = any(
        isinstance(item, dict) and str(item.get("id") or "").strip() == active_id
        for item in sessions
    )
    if not has_active and history:
        sessions.append(
            {
                "id": active_id,
                "title": str(active_session.get("title") or "").strip() or "Current Session",
                "updated_at": str(active_session.get("updated_at") or "").strip() or _now_iso(),
                "messages": list(history),
                "internal_turns": list(active_session.get("internal_turns") or []),
            }
        )
    return sessions


# ---------------------------------------------------------------------------
# Banner & display
# ---------------------------------------------------------------------------

def _print_cli_banner() -> None:
    print(ANSI_CYAN)
    print("+" + "=" * 71 + "+")
    print("|" + " " * 71 + "|")
    print("|   MEMENTO-S  —  Multi-turn Agent CLI" + " " * 33 + "|")
    print("|" + " " * 71 + "|")
    print("+" + "=" * 71 + "+")
    print(ANSI_RESET)


def _print_help() -> None:
    print("Commands:")
    print("  /       Show slash command menu (auto popup while typing)")
    print("  /help   Show this help")
    print("  /status Show session status")
    print("  /skills [query] [-n N] Search cloud skills (or list local skills)")
    print("  /config [show|get|set|unset] Manage API/model config in .env")
    print("  /history [N] Show session history window (default: 12 sessions)")
    print("  /history load <index> Load one saved session into current context")
    print("  /clear  Clear conversation context/history")
    print("  /exit   Exit the CLI")
    print("  Ctrl+C  Interrupt current running task")
    print()


def _print_slash_menu() -> None:
    print("Slash commands:")
    for cmd, desc in SLASH_COMMANDS:
        print(f"  {cmd:<12} {desc}")
    print()


def _print_slash_suggestions(raw: str) -> None:
    token = str(raw or "").strip()
    if not token.startswith("/"):
        return
    candidate_cmds = [cmd for cmd, _ in SLASH_COMMANDS]
    matches = [cmd for cmd in candidate_cmds if cmd.startswith(token)]
    if not matches:
        print(f"Unknown command: {token}")
        print("Type `/` to list available commands.\n")
        return
    print(f"Unknown command: {token}")
    print("Did you mean:")
    for cmd in matches:
        print(f"  {cmd}")
    print()


def _print_status(
    *,
    model: str,
    turn_count: int,
    messages_count: int,
    tool_names: list[str],
    debug: bool,
    session_title: str,
) -> None:
    print("Status:")
    print(f"  model: {model}")
    print(f"  turns: {turn_count}")
    print(f"  context_messages: {messages_count}")
    print(f"  tools: {', '.join(tool_names) if tool_names else '(none)'}")
    print(f"  session_title: {session_title or '(untitled)'}")
    print(f"  debug: {debug}")
    print()


# ---------------------------------------------------------------------------
# AgentSession — holds MCPAgent + multi-turn message history
# ---------------------------------------------------------------------------

class AgentSession:
    """Manages the MCPAgent lifecycle and accumulated conversation messages."""

    def __init__(self, *, base_dir: Path | None = None, debug: bool = False) -> None:
        self.messages: list[dict[str, str]] = []
        self.debug = debug
        self._base_dir = base_dir
        self.agent: Any = None  # MCPAgent instance

    async def start(self) -> None:
        from core.model_factory import build_chat_model
        from core.mcp_agent import MCPAgent

        model = build_chat_model()
        self.agent = MCPAgent(model=model, base_dir=self._base_dir)
        await self.agent.start()
        if self.debug:
            print(f"[debug] MCPAgent started, tools: {self.agent.tool_names}")

    async def rebuild(self) -> None:
        """Close and rebuild the agent after config changes."""
        if self.agent is not None:
            await self.agent.close()
        await self.start()
        print("Agent rebuilt with new model config.\n")

    async def close(self) -> None:
        if self.agent is not None:
            await self.agent.close()
            self.agent = None

    def clear(self) -> None:
        """Reset conversation messages."""
        self.messages.clear()

    @property
    def model_name(self) -> str:
        return os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

    @property
    def tool_names(self) -> list[str]:
        if self.agent is not None:
            return self.agent.tool_names
        return []


# ---------------------------------------------------------------------------
# Tool call / result display helpers
# ---------------------------------------------------------------------------


def _match_active_skill(cmd: str, active_skills: list[str]) -> str | None:
    """Try to match a bash command to one of the recently-read skills."""
    if not active_skills or not cmd:
        return None
    cmd_lower = cmd.lower()
    for skill_name in reversed(active_skills):
        sn = skill_name.lower()
        # Match skill name in path fragments: skills/serpapi/, skill_extra/pdf/, etc.
        if f"skills/{sn}/" in cmd_lower or f"skill_extra/{sn}/" in cmd_lower:
            return skill_name
        # Match skill name in script names: serpapi_search, pdf_convert, etc.
        sn_under = sn.replace("-", "_")
        if sn_under in cmd_lower or sn in cmd_lower:
            return skill_name
    return None


def _print_tool_call(step: int, name: str, args: dict, *, active_skills: list[str] | None = None) -> None:
    """Print a compact line showing which tool the agent is calling."""
    label = f"{ANSI_YELLOW}[Step {step}]{ANSI_RESET}"
    tool_label = f"{ANSI_BOLD}{ANSI_CYAN}{name}{ANSI_RESET}"

    detail = ""
    if name == "read_skill":
        skill = args.get("skill_name", "")
        detail = f" {ANSI_MAGENTA}skill={skill}{ANSI_RESET}"
    elif name == "bash_tool":
        cmd = str(args.get("command", "")).strip()
        desc = str(args.get("description", "")).strip()
        # Try to identify which skill is driving this command
        matched_skill = _match_active_skill(cmd, active_skills or [])
        skill_tag = f" {ANSI_MAGENTA}[{matched_skill}]{ANSI_RESET}" if matched_skill else ""
        if cmd:
            first_line = cmd.split("\n")[0]
            if len(first_line) > 80:
                first_line = first_line[:77] + "..."
            detail = f"{skill_tag} {ANSI_DIM}$ {first_line}{ANSI_RESET}"
        elif desc:
            detail = f"{skill_tag} {ANSI_DIM}{desc}{ANSI_RESET}"
        else:
            detail = skill_tag
    elif name == "str_replace":
        path = str(args.get("path", "")).strip()
        if path:
            detail = f" {ANSI_DIM}{path}{ANSI_RESET}"
    elif name == "file_create":
        path = str(args.get("path", "")).strip()
        if path:
            detail = f" {ANSI_DIM}{path}{ANSI_RESET}"
    elif name == "view":
        path = str(args.get("path", "")).strip()
        if path:
            detail = f" {ANSI_DIM}{path}{ANSI_RESET}"

    print(f"  {label} {tool_label}{detail}")


def _print_tool_result(tool_name: str, content: str, *, debug: bool = False) -> None:
    """Print a compact tool result indicator."""
    content = content.strip()
    if not content:
        return

    is_err = content.startswith(f"{tool_name} ERR") or "ERR:" in content[:50]
    is_ok = content.startswith(f"{tool_name} OK") or content == "OK"

    if is_err:
        first_line = content.split("\n")[0]
        if len(first_line) > 100:
            first_line = first_line[:97] + "..."
        print(f"         {ANSI_YELLOW}⚠ {first_line}{ANSI_RESET}")
    elif is_ok:
        print(f"         {ANSI_GREEN}✓ {content.split(chr(10))[0][:100]}{ANSI_RESET}")
    elif debug:
        preview = content[:200] + "..." if len(content) > 200 else content
        preview = preview.replace("\n", " ")
        print(f"         {ANSI_DIM}→ {preview}{ANSI_RESET}")


def _has_read_skill_call(tool_calls: list) -> list[str]:
    """Return skill names from any read_skill calls in the batch."""
    names: list[str] = []
    for tc in tool_calls:
        tc_name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
        if tc_name == "read_skill":
            tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
            if isinstance(tc_args, dict):
                name = str(tc_args.get("skill_name", "")).strip()
                if name:
                    names.append(name)
    return names


def _reroute_for_subtask(
    subtask_num: int,
    skill_names: list[str],
    tool_calls: list,
    *,
    debug: bool = False,
) -> None:
    """Re-route when a new subtask begins (agent reads a new skill)."""
    # Build subtask description from tool call descriptions + skill names
    subtask_parts: list[str] = list(skill_names)
    for tc in tool_calls:
        tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
        if not isinstance(tc_args, dict):
            tc_args = {}
        desc = str(tc_args.get("description", "")).strip()
        if desc:
            subtask_parts.append(desc)

    subtask_text = " ".join(subtask_parts).strip()
    if not subtask_text:
        return

    try:
        from core.skill_engine.skill_catalog import get_skill_catalog
        catalog = get_skill_catalog()
        matched = catalog.route(subtask_text, top_k=3)
        if matched:
            names = [s.get("name", "?") for s in matched]
            skills_str = ", ".join(names)
            print(f"  {ANSI_MAGENTA}⚡ Subtask {subtask_num} re-route: {skills_str}{ANSI_RESET}")
    except Exception as exc:
        if debug:
            print(f"[debug] re-route error: {exc}")


# ---------------------------------------------------------------------------
# Turn execution (streaming)
# ---------------------------------------------------------------------------

async def _execute_turn_streaming(
    session: AgentSession,
    user_text: str,
    *,
    debug: bool = False,
) -> str:
    """Run one user turn through the MCPAgent, streaming output to stdout."""
    augmented_text = user_text
    if SEMANTIC_ROUTER_ENABLED:
        try:
            from core.skill_engine.skill_catalog import get_skill_catalog
            catalog = get_skill_catalog()
            matched = catalog.route(user_text)
            if matched:
                context = catalog.format_skills_context(matched)
                augmented_text = f"{user_text}\n\n{context}"
                names = [s.get("name", "?") for s in matched]
                skills_str = ", ".join(names)
                print(f"  {ANSI_MAGENTA}⚡ Matched skills: {skills_str}{ANSI_RESET}")
        except Exception as exc:
            if debug:
                print(f"[debug] router error: {exc}")

    session.messages.append({"role": "user", "content": augmented_text})

    final_text = ""
    step_num = 0
    subtask_num = 0
    active_skills: list[str] = []  # tracks skills read via read_skill
    try:
        async for chunk in session.agent.stream(session.messages):
            # LangGraph stream_mode="updates" yields dicts keyed by node name.
            # The "agent" node contains LLM output messages.
            if isinstance(chunk, dict):
                for node_name, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    msgs = update.get("messages", [])
                    if not isinstance(msgs, list):
                        continue
                    for msg in msgs:
                        # Tool call messages (from agent deciding to use a tool)
                        tool_calls = getattr(msg, "tool_calls", None)
                        if tool_calls:
                            # Check if this batch contains read_skill → new subtask
                            new_skills = _has_read_skill_call(tool_calls)
                            if new_skills and SEMANTIC_ROUTER_ENABLED:
                                subtask_num += 1
                                if subtask_num > 1:
                                    # Don't re-route for the very first read_skill
                                    # (already covered by initial routing)
                                    _reroute_for_subtask(
                                        subtask_num, new_skills, tool_calls,
                                        debug=debug,
                                    )

                            for tc in tool_calls:
                                step_num += 1
                                tc_name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                                tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                                # Track which skills have been read
                                if tc_name == "read_skill":
                                    skill_name = str(tc_args.get("skill_name", "")).strip()
                                    if skill_name and skill_name not in active_skills:
                                        active_skills.append(skill_name)
                                _print_tool_call(step_num, tc_name, tc_args, active_skills=active_skills)

                        # Tool result messages
                        if hasattr(msg, "type") and msg.type == "tool":
                            result_content = str(getattr(msg, "content", ""))
                            tool_name = getattr(msg, "name", None) or ""
                            _print_tool_result(tool_name, result_content, debug=debug)
                            continue

                        # AI text messages (final or intermediate)
                        content = getattr(msg, "content", None)
                        if isinstance(content, str) and content.strip():
                            # Only print the last AI message (final answer)
                            final_text = content.strip()
    except KeyboardInterrupt:
        raise TurnInterrupted(partial_reply=final_text)

    if final_text:
        print(f"Assistant> {final_text}\n")
        session.messages.append({"role": "assistant", "content": final_text})
    else:
        # Fallback: try non-streaming run if streaming yielded nothing
        try:
            result = await session.agent.run(session.messages)
            msgs = result.get("messages", [])
            if msgs:
                last_msg = msgs[-1]
                content = getattr(last_msg, "content", None) or ""
                if isinstance(content, str) and content.strip():
                    final_text = content.strip()
                    print(f"Assistant> {final_text}\n")
                    session.messages.append({"role": "assistant", "content": final_text})
        except KeyboardInterrupt:
            raise TurnInterrupted(partial_reply="")

    if not final_text:
        final_text = "(no response)"
        print(f"Assistant> {final_text}\n")
        session.messages.append({"role": "assistant", "content": final_text})

    return final_text


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memento-S Multi-turn Agent CLI")
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Run one prompt and exit. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "-p", "--prompt-text",
        dest="prompt_text",
        default=None,
        help="Run a single prompt and exit (alternative to positional args).",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=DEBUG,
        help="Show debug output (tool calls, timing).",
    )
    parser.add_argument("--no-banner", action="store_true", help="Suppress startup banner.")
    return parser


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------

async def _async_main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    debug = bool(args.debug)

    # Persistence
    history_file = (Path.cwd() / DEFAULT_HISTORY_FILE).resolve()
    history_store = _load_history_store(history_file)
    env_file = DEFAULT_ENV_FILE
    active_session_meta: dict[str, Any] = _new_session()
    turn_count = 0

    # Build agent session
    session = AgentSession(base_dir=WORKSPACE_DIR, debug=debug)
    try:
        await session.start()
    except Exception as exc:
        print(f"Error: failed to start MCPAgent: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Helpers for recording turns
    # ------------------------------------------------------------------
    def _record_turn(user_text: str, reply_text: str, *, interrupted: bool = False) -> None:
        nonlocal turn_count, active_session_meta
        turn_count += 1
        if not str(active_session_meta.get("title") or "").strip():
            active_session_meta["title"] = _build_session_title(user_text)
        active_session_meta["updated_at"] = _now_iso()
        active_session_meta["messages"] = list(session.messages)

        internal_turns = active_session_meta.get("internal_turns")
        turns = list(internal_turns) if isinstance(internal_turns, list) else []
        turns.append(
            {
                "ts": _now_iso(),
                "user": user_text,
                "assistant": reply_text,
                "interrupted": interrupted,
            }
        )
        if len(turns) > INTERNAL_TURN_LIMIT:
            turns = turns[-INTERNAL_TURN_LIMIT:]
        active_session_meta["internal_turns"] = turns

        _upsert_session(history_store, active_session_meta)
        _save_history_store(history_file, history_store)

    # ------------------------------------------------------------------
    # Single-turn mode
    # ------------------------------------------------------------------
    _single_prompt = args.prompt_text or (" ".join(args.prompt).strip() if args.prompt else "")
    if _single_prompt:
        user_text = _single_prompt
        if not user_text:
            await session.close()
            return 1
        try:
            reply = await _execute_turn_streaming(session, user_text, debug=debug)
            _record_turn(user_text, reply)
        except TurnInterrupted as intr:
            partial = str(intr.partial_reply or "").strip()
            reply = f"[interrupted] {partial}" if partial else "[interrupted]"
            _record_turn(user_text, reply, interrupted=True)
            print("\nInterrupted.")
            await session.close()
            return 130
        except KeyboardInterrupt:
            print("\nInterrupted.")
            await session.close()
            return 130
        await session.close()
        return 0

    # ------------------------------------------------------------------
    # Interactive REPL
    # ------------------------------------------------------------------
    if not args.no_banner:
        _print_cli_banner()
        print(f"Model: {session.model_name}")
        print(f"Tools: {', '.join(session.tool_names)}")
        print("Type /help for commands.\n")

    prompt_session = _build_prompt_session()
    if prompt_session is None:
        print("Tip: install `prompt_toolkit` to enable dynamic slash menu while typing `/`.\n")

    _consecutive_ctrl_c = 0
    try:
        while True:
            # Read input
            try:
                if prompt_session is not None:
                    user_text = str(await prompt_session.prompt_async("You> ")).strip()
                else:
                    user_text = input("You> ").strip()
                _consecutive_ctrl_c = 0  # reset on successful input
            except EOFError:
                print("\nBye.")
                break
            except KeyboardInterrupt:
                _consecutive_ctrl_c += 1
                if _consecutive_ctrl_c >= 2:
                    print("\nBye.")
                    break
                print("\n(Press Ctrl+C again to exit, or type a message)")
                continue

            if not user_text:
                continue

            # Slash menu
            if user_text == "/":
                _print_slash_menu()
                continue

            # Exit
            if user_text in EXIT_COMMANDS:
                print("Bye.")
                break

            # Help
            if user_text in HELP_COMMANDS:
                _print_help()
                continue

            cmd = user_text.strip().split(maxsplit=1)

            # Status
            if user_text in STATUS_COMMANDS:
                current_title = str(active_session_meta.get("title") or "").strip()
                _print_status(
                    model=session.model_name,
                    turn_count=turn_count,
                    messages_count=len(session.messages),
                    tool_names=session.tool_names,
                    debug=debug,
                    session_title=current_title,
                )
                continue

            # Skills
            if cmd and cmd[0] in SKILLS_COMMANDS:
                arg = cmd[1].strip() if len(cmd) > 1 else ""
                if not arg:
                    try:
                        if prompt_session is not None:
                            arg = str(await prompt_session.prompt_async("Skills(query)> ")).strip()
                        else:
                            arg = input("Skills(query)> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        continue
                    if not arg:
                        # List local skills via MCP tool
                        from core.mcp_server import _list_local_skills_text as list_local_skills
                        result = list_local_skills()
                        print(f"Local skills:\n{result}\n")
                        print("Tip: use `/skills <query> -n 5` to search cloud skills.\n")
                        continue

                arg_norm = arg
                if arg_norm.lower().startswith("cloud "):
                    arg_norm = arg_norm[6:].strip()

                query_text, top_k, parse_err = _parse_skills_args(arg_norm, default_limit=5)
                if parse_err:
                    print(f"Usage: /skills [query|local] [-n N] ({parse_err})\n")
                    continue

                if query_text.lower() == "local":
                    from core.mcp_server import _list_local_skills_text as list_local_skills
                    result = list_local_skills()
                    print(f"Local skills:\n{result}\n")
                    continue
                _print_cloud_skills(query_text, top_k=top_k)
                continue

            # Config
            if cmd and cmd[0] in CONFIG_COMMANDS:
                raw_args = cmd[1] if len(cmd) > 1 else ""
                changed_key = _handle_config_command(raw_args, env_path=env_file)
                if changed_key is not None:
                    await session.rebuild()
                continue

            # Unknown slash command
            if cmd and cmd[0].startswith("/") and cmd[0] not in KNOWN_SLASH_COMMANDS:
                _print_slash_suggestions(cmd[0])
                continue

            # History
            if cmd and cmd[0] in HISTORY_COMMANDS:
                sessions = _collect_history_sessions(
                    history_store,
                    active_session=active_session_meta,
                    history=session.messages,
                )
                history_limit = 12
                if len(cmd) > 1:
                    raw_arg = cmd[1].strip()
                    history_tokens = _split_shell_tokens(raw_arg)

                    if history_tokens and str(history_tokens[0]).strip().lower() == "load":
                        if len(history_tokens) != 2:
                            print("Usage: /history load <index>\n")
                            continue
                        idx_raw = str(history_tokens[1] or "").strip()
                        try:
                            target_index = int(idx_raw)
                        except Exception:
                            print(f"Usage: /history load <index> (invalid index: {idx_raw!r})\n")
                            continue
                        if target_index <= 0:
                            print("Usage: /history load <index> (index must be >= 1)\n")
                            continue
                        if target_index > len(sessions):
                            print(
                                f"History load failed: index {target_index} out of range "
                                f"(1..{len(sessions) if sessions else 0}).\n"
                            )
                            continue
                        selected = _sanitize_session(sessions[target_index - 1])
                        if selected is None:
                            print(f"History load failed: session #{target_index} is invalid.\n")
                            continue

                        loaded_history = _sanitize_history_items(selected.get("messages"))
                        active_session_meta = selected
                        session.messages = loaded_history[-SESSION_MESSAGE_LIMIT:]
                        active_session_meta["messages"] = list(session.messages)
                        turn_count = len(session.messages) // 2
                        loaded_title = str(active_session_meta.get("title") or "").strip() or "Untitled Session"
                        print(
                            f"Loaded session #{target_index}: {loaded_title} "
                            f"(messages={len(session.messages)}, turns={turn_count}).\n"
                        )
                        continue

                    try:
                        history_limit = int(raw_arg)
                    except Exception:
                        print("Usage: /history [N] | /history load <index>\n")
                        continue
                    if history_limit <= 0:
                        print("Usage: /history [N] (N must be >= 1)\n")
                        continue
                _print_history_window(sessions, history_limit)
                continue

            # Clear
            if user_text in CLEAR_COMMANDS:
                session.clear()
                turn_count = 0
                active_session_meta = _new_session()
                print("Context/history cleared.\n")
                continue

            # ----------------------------------------------------------
            # Normal turn — run through MCPAgent
            # ----------------------------------------------------------
            _consecutive_ctrl_c = 0
            interrupted = False
            try:
                reply = await _execute_turn_streaming(session, user_text, debug=debug)
            except TurnInterrupted as intr:
                interrupted = True
                partial = str(intr.partial_reply or "").strip()
                reply = f"[interrupted] {partial}" if partial else "[interrupted]"

            _record_turn(user_text, reply, interrupted=interrupted)

            if interrupted:
                print("\nInterrupted current task. Partial context saved.\n")

    finally:
        await session.close()

    return 0


# ---------------------------------------------------------------------------
# Sync entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
