"""Memento-S MVP CLI (Claude Code-style interaction loop)."""

from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import shutil
import sys
import textwrap
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar
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

from core.config import CHAT_SYSTEM_PROMPT, CLI_CREATE_ON_MISS, DEBUG, PROJECT_ROOT, refresh_runtime_config
from core.llm import openrouter_messages, reset_llm_call_budget
from core.skill_engine.skill_runner import create_skill_on_miss
from cli.skill_search import load_cloud_skill_catalog, search_cloud_skills
from cli.workflow_runner import SkillWorkflowRunner

T = TypeVar("T")

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}
HELP_COMMANDS = {"help", "/help"}
CLEAR_COMMANDS = {"clear", "/clear"}
STATUS_COMMANDS = {"status", "/status"}
RETRY_COMMANDS = {"retry", "/retry", "continue", "/continue"}
LAST_COMMANDS = {"last", "/last"}
SKILLS_COMMANDS = {"skills", "/skills"}
HISTORY_COMMANDS = {"history", "/history"}
CONFIG_COMMANDS = {"config", "/config"}
PREWARM_COMMANDS = {"prewarm", "/prewarm"}
DEFAULT_HISTORY_FILE = Path(".agent/cli_history.json")
DEFAULT_ENV_FILE = (PROJECT_ROOT / ".env").resolve()
SESSION_MESSAGE_LIMIT = 200
INTERNAL_TURN_LIMIT = 200
SESSION_STORE_LIMIT = 200
ANSI_CYAN = "\033[0;36m"
ANSI_RESET = "\033[0m"
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
    "router_method": "SEMANTIC_ROUTER_METHOD",
    "router_top_k": "SEMANTIC_ROUTER_TOP_K",
    "router_qwen_model": "SEMANTIC_ROUTER_QWEN_MODEL_PATH",
    "router_qwen_tokenizer": "SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH",
    "router_memento_model": "SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH",
    "router_memento_tokenizer": "SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH",
    "router_embed_batch": "SEMANTIC_ROUTER_EMBED_BATCH_SIZE",
    "router_embed_max_len": "SEMANTIC_ROUTER_EMBED_MAX_LENGTH",
    "router_embed_cache": "SEMANTIC_ROUTER_EMBED_CACHE_DIR",
    "router_embed_prewarm": "SEMANTIC_ROUTER_EMBED_PREWARM",
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
    "SEMANTIC_ROUTER_METHOD",
    "SEMANTIC_ROUTER_TOP_K",
    "SEMANTIC_ROUTER_QWEN_MODEL_PATH",
    "SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH",
    "SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH",
    "SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH",
    "SEMANTIC_ROUTER_EMBED_BATCH_SIZE",
    "SEMANTIC_ROUTER_EMBED_MAX_LENGTH",
    "SEMANTIC_ROUTER_EMBED_CACHE_DIR",
    "SEMANTIC_ROUTER_EMBED_PREWARM",
)
CONFIG_KEYS_SET = set(CONFIG_KEYS)
CONFIG_SECRET_KEYS = {"OPENROUTER_API_KEY", "SERPAPI_API_KEY", "OPENAI_API_KEY"}
CONFIG_ATTR_OVERRIDES = {"OPENROUTER_MODEL": "MODEL"}
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show this help"),
    ("/status", "Show session status"),
    ("/retry", "Retry the last user request"),
    ("/continue", "Alias of /retry"),
    ("/last", "Show the last assistant reply"),
    ("/skills [query] [-n N]", "Search cloud skills or list local skills"),
    ("/prewarm [auto|qwen|memento|all]", "Prewarm router embedding cache/runtime"),
    ("/config [show|get|set|unset]", "View/update .env config (api/model/etc.)"),
    ("/history [N]", "Show session history window"),
    ("/history load <index>", "Load one saved session into current context"),
    ("/clear", "Clear conversation context/history"),
    ("/exit", "Exit the CLI"),
]
KNOWN_SLASH_COMMANDS = {cmd.split()[0].strip() for cmd, _ in SLASH_COMMANDS if cmd.split()}


class TurnInterrupted(Exception):
    """Raised when a running turn is interrupted, with optional partial output."""

    def __init__(self, partial_reply: str = "") -> None:
        super().__init__("turn interrupted")
        self.partial_reply = str(partial_reply or "")


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
        def _slash_autocomplete(event):  # type: ignore[no-redef]
            buf = event.app.current_buffer
            buf.insert_text("/")
            try:
                text = str(buf.document.text_before_cursor or "")
                token = text.lstrip().split(maxsplit=1)[0] if text.lstrip() else ""
                if token == "/":
                    # Force completion popup immediately after typing "/".
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


def _load_history_store(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {"sessions": []}
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Backward compatibility: previous format stored a flat message list.
        if isinstance(raw, list):
            legacy_messages = _sanitize_history_items(raw)[-SESSION_MESSAGE_LIMIT:]
            if not legacy_messages:
                return {"sessions": []}
            first_user = next((m.get("content", "") for m in legacy_messages if m.get("role") == "user"), "")
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
        # History persistence should never block core interaction flow.
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


def _print_help() -> None:
    print("Commands:")
    print("  /       Show slash command menu (auto popup while typing)")
    print("  /help   Show this help")
    print("  /status Show session status")
    print("  /retry  Retry the last user request")
    print("  /last   Show the last assistant reply")
    print("  /skills [query] [-n N] Search cloud skills (or list local skills)")
    print("  /prewarm [auto|qwen|memento|all] Warm router embedding cache/runtime")
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


def _print_cli_banner() -> None:
    print(ANSI_CYAN)
    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                       ║")
    print("║   ███╗   ███╗███████╗███╗   ███╗███████╗███╗   ██╗████████╗ ██████╗    ║")
    print("║   ████╗ ████║██╔════╝████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗   ║")
    print("║   ██╔████╔██║█████╗  ██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ██║   ██║   ║")
    print("║   ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ██║   ██║   ║")
    print("║   ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ╚██████╔╝   ║")
    print("║   ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝    ║")
    print("║                           Memento-S                                   ║")
    print("║                   One-Click Installer (uv)                            ║")
    print("║                                                                       ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")
    print(ANSI_RESET)


def _print_status(
    *,
    runner: SkillWorkflowRunner,
    max_steps: int,
    create_on_miss: bool,
    optimize_on_error: bool,
    optimize_attempts: int,
    debug: bool,
    turn_count: int,
    history: list[dict[str, str]],
    session_title: str,
    last_user_request: str,
    last_reply: str,
    last_error: str,
) -> None:
    try:
        runner.reload_skills_metadata()
    except Exception:
        pass
    skill_count = len(runner.get_skill_names())
    context_turns = len(history) // 2
    history_messages = len(history)
    print("Status:")
    print(f"  turns: {turn_count}")
    print(f"  context_turns: {context_turns}")
    print(f"  history_messages: {history_messages}")
    print(f"  session_title: {session_title or '(untitled)'}")
    print(f"  loaded_skills: {skill_count}")
    print(f"  max_steps: {max_steps}")
    print(f"  create_on_miss: {create_on_miss}")
    print(f"  optimize_on_error: {optimize_on_error}")
    print(f"  optimize_attempts: {optimize_attempts}")
    print(f"  debug: {debug}")
    print(f"  last_request: {'yes' if last_user_request else 'no'}")
    print(f"  last_reply: {'yes' if last_reply else 'no'}")
    print(f"  last_error: {'yes' if last_error else 'no'}")
    print()


def _print_skills(runner: SkillWorkflowRunner) -> None:
    try:
        runner.reload_skills_metadata()
    except Exception:
        pass
    names = sorted(
        {
            str(name).strip()
            for name in runner.get_skill_names()
            if isinstance(name, str) and str(name).strip()
        }
    )
    if not names:
        print("Skills: none loaded.\n")
        return
    print(f"Skills ({len(names)}):")
    for name in names:
        print(f"  {name}")
    print()


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

        meta_bits = [f"⭐ {stars}"]
        if author:
            meta_bits.append(f"author: {author}")

        print(f"{idx}. {name}  ({', '.join(meta_bits)})")
        if desc:
            print(f"   {desc}")
        if github_url:
            print(f"   github: {github_url}")
    print()


def _resolve_prewarm_methods(raw: str) -> tuple[str, ...]:
    token = str(raw or "").strip().lower()
    if token in {"", "auto", "router", "current"}:
        method = str(os.getenv("SEMANTIC_ROUTER_METHOD") or "bm25").strip().lower()
        if method in {"qwen", "qwen3", "qwen_embedding", "qwen3_embedding"}:
            return ("qwen_embedding",)
        if method in {"memento", "memento_qwen", "memento-qwen", "memento_qwen_embedding"}:
            return ("memento_qwen_embedding",)
        return ()

    if token in {"qwen", "qwen3", "qwen_embedding", "qwen3_embedding"}:
        return ("qwen_embedding",)
    if token in {"memento", "memento_qwen", "memento-qwen", "memento_qwen_embedding"}:
        return ("memento_qwen_embedding",)
    if token in {"all", "both", "embedding", "embeddings"}:
        return ("qwen_embedding", "memento_qwen_embedding")
    return ()


def _build_semantic_catalog_skills(runner: SkillWorkflowRunner) -> tuple[list[dict[str, Any]], str]:
    local_skills = [
        s
        for s in list(getattr(runner, "skills", []) or [])
        if isinstance(s, dict) and str(s.get("name") or "").strip()
    ]
    semantic_catalog_skills = local_skills
    catalog_source = "runtime"

    try:
        from core.config import SEMANTIC_ROUTER_CATALOG_JSONL, SEMANTIC_ROUTER_CATALOG_MD
        from core.skill_engine.skill_catalog import (
            _load_router_catalog_from_jsonl,
            load_available_skills_block_from,
            parse_available_skills,
        )
    except Exception:
        return semantic_catalog_skills, catalog_source

    if SEMANTIC_ROUTER_CATALOG_JSONL:
        try:
            catalog_skills, _ = _load_router_catalog_from_jsonl(SEMANTIC_ROUTER_CATALOG_JSONL)
            if catalog_skills:
                return catalog_skills, "jsonl"
        except Exception:
            pass

    if SEMANTIC_ROUTER_CATALOG_MD:
        try:
            catalog_xml = load_available_skills_block_from(SEMANTIC_ROUTER_CATALOG_MD)
            catalog_skills = parse_available_skills(catalog_xml)
            if catalog_skills:
                return catalog_skills, "xml"
        except Exception:
            pass

    return semantic_catalog_skills, catalog_source


def _run_router_prewarm(
    runner: SkillWorkflowRunner,
    *,
    raw_args: str = "",
    debug: bool = False,
) -> None:
    methods = _resolve_prewarm_methods(raw_args)
    if not methods:
        active = str(os.getenv("SEMANTIC_ROUTER_METHOD") or "bm25").strip().lower()
        if str(raw_args or "").strip():
            print(
                "Usage: /prewarm [auto|qwen|memento|all] "
                f"(invalid method: {raw_args!r})\n"
            )
        else:
            print(
                f"Prewarm skipped: SEMANTIC_ROUTER_METHOD={active!r} "
                "does not use embedding router.\n"
            )
        return

    try:
        from core.skill_engine.skill_catalog import (
            precompute_router_embedding_cache,
            select_embedding_top_skills,
        )
    except Exception as exc:
        print(f"Prewarm failed: cannot import router embedding modules ({exc}).\n")
        return

    try:
        runner.reload_skills_metadata()
    except Exception:
        pass

    semantic_catalog_skills, catalog_source = _build_semantic_catalog_skills(runner)
    if not semantic_catalog_skills:
        print("Prewarm skipped: no catalog skills available.\n")
        return

    total_stages = 2 + len(methods)
    stage_idx = 0
    progress_bar = None

    try:
        from tqdm import tqdm

        progress_bar = tqdm(total=total_stages, desc="router-prewarm", unit="stage")
    except Exception:
        progress_bar = None

    def _stage(label: str) -> None:
        nonlocal stage_idx
        stage_idx += 1
        if progress_bar is not None:
            progress_bar.set_postfix_str(label)
            progress_bar.update(1)
        else:
            print(f"[prewarm] {stage_idx}/{total_stages}: {label}")

    print(
        f"Prewarm start: methods={', '.join(methods)} "
        f"catalog={catalog_source} skills={len(semantic_catalog_skills)}"
    )
    _stage("catalog-ready")

    t_cache = time.perf_counter()
    results = precompute_router_embedding_cache(
        semantic_catalog_skills,
        methods=methods,
        show_progress=True,
    )
    _stage("doc-cache")
    cache_elapsed = time.perf_counter() - t_cache

    warm_query = "__router_prewarm_probe__"
    warmup_times: list[tuple[str, float]] = []
    for method in methods:
        t_warm = time.perf_counter()
        try:
            _ = select_embedding_top_skills(
                warm_query,
                semantic_catalog_skills,
                method=method,
                top_k=1,
            )
            warmup_times.append((method, time.perf_counter() - t_warm))
        except Exception:
            warmup_times.append((method, -1.0))
        _stage(f"{method}-query")

    if progress_bar is not None:
        try:
            progress_bar.close()
        except Exception:
            pass

    print(f"Prewarm done in {cache_elapsed:.2f}s (doc-cache stage).")
    for method, status in results:
        print(f"- {method}: {status}")
    for method, sec in warmup_times:
        if sec >= 0:
            print(f"- {method}: query warmup {sec:.3f}s")
        else:
            print(f"- {method}: query warmup failed")
    if debug:
        print("[debug] prewarm note: offline catalog embedding is reused when cache hits.")
    print()


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
        "router_method, router_top_k, router_qwen_model, router_memento_model, "
        "router_embed_cache, router_embed_prewarm"
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


def _handle_config_command(raw_args: str, *, env_path: Path) -> None:
    text = str(raw_args or "").strip()
    if not text:
        _print_config_show(env_path)
        _print_config_help()
        return

    tokens = _split_shell_tokens(text)

    if not tokens:
        _print_config_show(env_path)
        _print_config_help()
        return

    action = str(tokens[0] or "").strip().lower()
    if action in {"show", "list"}:
        _print_config_show(env_path)
        return
    if action == "path":
        print(f"{env_path}\n")
        return

    if action == "get":
        if len(tokens) < 2:
            print("Usage: /config get <key|alias>\n")
            return
        key = _normalize_config_key(tokens[1])
        if not key:
            print(f"Unsupported key/alias: {tokens[1]!r}\n")
            return
        value = _effective_config_value(key)
        print(f"{key}={_mask_config_value(key, value)}\n")
        return

    if action == "set":
        if len(tokens) < 3:
            print("Usage: /config set <key|alias> <value>\n")
            return
        key = _normalize_config_key(tokens[1])
        if not key:
            print(f"Unsupported key/alias: {tokens[1]!r}\n")
            return
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
        return

    if action == "unset":
        if len(tokens) < 2:
            print("Usage: /config unset <key|alias>\n")
            return
        key = _normalize_config_key(tokens[1])
        if not key:
            print(f"Unsupported key/alias: {tokens[1]!r}\n")
            return
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
        return

    _print_config_help()


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


def _extract_last_turn_fields(history: list[dict[str, str]]) -> tuple[str, str, str]:
    last_user_request = ""
    last_reply = ""
    last_error = ""
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            last_user_request = content
        elif role == "assistant":
            last_reply = content
    if last_reply.startswith("ERR:"):
        last_error = last_reply
    return last_user_request, last_reply, last_error


def _append_history_turn(
    history: list[dict[str, str]],
    *,
    user_text: str,
    assistant_text: str,
    limit: int = SESSION_MESSAGE_LIMIT,
) -> None:
    text = str(assistant_text or "").strip()
    if not text:
        return
    history.extend(
        [
            {"role": "user", "content": str(user_text or "")},
            {"role": "assistant", "content": text},
        ]
    )
    if len(history) > int(limit):
        history[:] = history[-int(limit):]


def _flush_captured(stdout_buf: io.StringIO, stderr_buf: io.StringIO, *, debug: bool) -> None:
    if not debug:
        return
    out = stdout_buf.getvalue().strip()
    err = stderr_buf.getvalue().strip()
    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)


def _run_quiet_call(fn: Callable[[], T], *, debug: bool) -> T:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            result = fn()
    except Exception:
        _flush_captured(stdout_buf, stderr_buf, debug=debug)
        raise
    _flush_captured(stdout_buf, stderr_buf, debug=debug)
    return result


def _iter_workflow_events(
    events: Iterable[tuple[dict[str, Any], Any]],
    *,
    debug: bool,
) -> Iterator[tuple[dict[str, Any], Any]]:
    it = iter(events)
    while True:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                item = next(it)
        except StopIteration:
            _flush_captured(stdout_buf, stderr_buf, debug=debug)
            break
        except Exception:
            _flush_captured(stdout_buf, stderr_buf, debug=debug)
            raise
        _flush_captured(stdout_buf, stderr_buf, debug=debug)
        yield item


def _chat_fallback(user_text: str, history: list[dict[str, str]], *, debug: bool) -> str:
    t_fallback = time.perf_counter()
    messages: list[dict[str, str]] = []
    for item in history[-40:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})
    try:
        t_llm = time.perf_counter()
        result = _run_quiet_call(
            lambda: openrouter_messages(
                CHAT_SYSTEM_PROMPT,
                messages,
            ),
            debug=debug,
        )
        if debug:
            print(f"[debug][timing] chat_fallback.llm: {time.perf_counter() - t_llm:.3f}s")
        text = str(result or "").strip()
        out = text or "No response."
        if debug:
            print(f"[debug][timing] chat_fallback.total: {time.perf_counter() - t_fallback:.3f}s")
        return out
    except Exception as exc:
        if debug:
            print(f"[debug][timing] chat_fallback.total: {time.perf_counter() - t_fallback:.3f}s")
        return f"ERR: {type(exc).__name__}: {exc}"


def _execute_turn(
    runner: SkillWorkflowRunner,
    user_text: str,
    history: list[dict[str, str]],
    *,
    max_steps: int,
    create_on_miss: bool,
    debug: bool,
) -> str:
    t_turn = time.perf_counter()

    def _return_with_timing(value: str, *, label: str = "turn_total") -> str:
        if debug:
            print(f"[debug][timing] {label}: {time.perf_counter() - t_turn:.3f}s")
        return value

    def _try_create_on_miss(
        router_reason: str,
        *,
        start_message: str,
        timing_label: str,
        debug_label: str,
    ) -> bool:
        if not create_on_miss:
            return False

        print(start_message)
        t_creator = time.perf_counter()
        created, created_skill, report = _run_quiet_call(
            lambda: create_skill_on_miss(
                user_text,
                router_reason=router_reason or None,
                available_skill_names=runner.get_skill_names(),
            ),
            debug=debug,
        )
        if debug:
            print(f"[debug][timing] {timing_label}: {time.perf_counter() - t_creator:.3f}s")
        if created and isinstance(created_skill, str) and created_skill.strip():
            print(f"Tool> router: created skill `{created_skill.strip()}`, retrying request")
            try:
                runner.reload_skills_metadata()
            except Exception as exc:
                if debug:
                    print(f"[debug] reload_skills_metadata failed: {exc}")
            return True
        if debug and report:
            print(f"[debug] {debug_label} skipped/failed: {report}")
        return False

    reset_llm_call_budget()
    runner.set_conversation_history(history)
    last_assistant = ""
    try:
        while True:
            should_retry = False
            t_workflow = time.perf_counter()
            events = runner.run_workflow_steps(user_text, max_steps=max_steps)
            try:
                for step_info, result in _iter_workflow_events(events, debug=debug):
                    status = str(step_info.get("status") or "")
                    step_num = int(step_info.get("step_num") or 0)
                    skill_name = str(step_info.get("skill_name") or "").strip()

                    if status == "running":
                        if step_num > 0 and skill_name:
                            print(f"Skill> step {step_num}: {skill_name}")
                        continue

                    if status == "optimizing":
                        attempt = step_info.get("attempt")
                        print(f"Skill> step {step_num}: optimizing {skill_name} (attempt {attempt})")
                        continue

                    if status == "optimized":
                        attempt = step_info.get("attempt")
                        ok = bool(step_info.get("ok"))
                        print(
                            "Skill> step "
                            f"{step_num}: optimization {'ok' if ok else 'failed'} (attempt {attempt})"
                        )
                        continue

                    if status == "completed":
                        text = str(result or "").strip()
                        if text:
                            print(f"Assistant> {text}\n")
                            last_assistant = text
                        continue

                    if status == "done":
                        reason = str(step_info.get("reason") or "").strip()
                        text = str(result or "").strip()
                        done_reason = reason or text
                        done_reason_l = done_reason.lower()

                        # Router "no_skill_needed" should use normal chat fallback instead of
                        # exposing internal router reason to the end user.
                        if done_reason_l in {"no_skill_needed", "no-skill-needed", "no skill needed"}:
                            fallback = _chat_fallback(user_text, history, debug=debug)
                            print(f"Assistant> {fallback}\n")
                            return _return_with_timing(fallback)

                        if text and text != last_assistant and text != done_reason:
                            print(f"Assistant> {text}\n")
                            last_assistant = text

                        complete_reason = (
                            done_reason if done_reason and done_reason_l != "task completed" else "Task completed."
                        )
                        print(f"Assistant> [workflow complete] {complete_reason}\n")
                        return _return_with_timing(last_assistant or complete_reason)

                    if status == "no_match":
                        reason = str(step_info.get("reason") or "").strip()
                        if _try_create_on_miss(
                            reason,
                            start_message="Tool> router: no matching skill, trying skill-creator",
                            timing_label="create_on_miss",
                            debug_label="create_on_miss",
                        ):
                            should_retry = True
                            break
                        fallback = _chat_fallback(user_text, history, debug=debug)
                        print(f"Assistant> {fallback}\n")
                        return _return_with_timing(fallback)

                    if status in {"no_skills", "unknown_action"}:
                        fallback = _chat_fallback(user_text, history, debug=debug)
                        print(f"Assistant> {fallback}\n")
                        return _return_with_timing(fallback)

                    if status == "error":
                        text = str(result or "Skill execution error").strip()
                        if text.startswith("Unknown skill:") and _try_create_on_miss(
                            text,
                            start_message="Tool> skill unavailable, trying skill-creator",
                            timing_label="create_on_miss(on_error)",
                            debug_label="create_on_miss(on_error)",
                        ):
                            should_retry = True
                            break
                        print(f"Assistant> {text}\n")
                        return _return_with_timing(text)

                    if status == "max_steps":
                        text = str(result or "").strip() or f"Stopped after max steps ({max_steps})."
                        print(f"Assistant> {text}\n")
                        return _return_with_timing(text)
            except Exception as exc:
                text = f"ERR: {type(exc).__name__}: {exc}"
                print(f"Assistant> {text}\n")
                return _return_with_timing(text, label="turn_total(error)")

            if debug:
                print(
                    "[debug][timing] workflow_iteration: "
                    f"{time.perf_counter() - t_workflow:.3f}s"
                )
            if should_retry:
                continue
            if last_assistant:
                return _return_with_timing(last_assistant)
            fallback = _chat_fallback(user_text, history, debug=debug)
            print(f"Assistant> {fallback}\n")
            return _return_with_timing(fallback)
    except KeyboardInterrupt as exc:
        raise TurnInterrupted(partial_reply=last_assistant) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memento-S MVP CLI")
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Run one prompt and exit. If omitted, starts interactive mode.",
    )
    parser.add_argument("--max-steps", type=int, default=20, help="Max workflow steps per user turn.")
    parser.add_argument(
        "--create-on-miss",
        action=argparse.BooleanOptionalAction,
        default=CLI_CREATE_ON_MISS,
        help="Try skill-creator when router returns no matching skill.",
    )
    parser.add_argument(
        "--optimize-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-optimize failing skills via skill-creator.",
    )
    parser.add_argument(
        "--optimize-attempts",
        type=int,
        default=1,
        help="Optimization attempts per step when optimize-on-error is enabled.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=DEBUG,
        help="Show captured internal logs/output.",
    )
    parser.add_argument(
        "--prewarm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prewarm router embedding cache/runtime at startup (with progress).",
    )
    parser.add_argument("--no-banner", action="store_true", help="Suppress startup banner.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    max_steps = max(1, int(args.max_steps))
    optimize_attempts = max(0, int(args.optimize_attempts))
    debug = bool(args.debug)

    runner = SkillWorkflowRunner(
        auto_sync=False,
        optimize_on_error=bool(args.optimize_on_error),
        optimize_attempts=optimize_attempts,
        debug=debug,
    )

    history_file = (Path.cwd() / DEFAULT_HISTORY_FILE).resolve()
    history_store = _load_history_store(history_file)
    env_file = DEFAULT_ENV_FILE
    active_session: dict[str, Any] = _new_session()
    history: list[dict[str, str]] = []
    turn_count = 0
    last_user_request = ""
    last_reply = ""
    last_error = ""

    def _run_and_record(request_text: str, *, raise_on_interrupt: bool = False) -> str:
        nonlocal turn_count, last_user_request, last_reply, last_error, active_session
        interrupted = False
        try:
            reply_text = _execute_turn(
                runner,
                request_text,
                history,
                max_steps=max_steps,
                create_on_miss=bool(args.create_on_miss),
                debug=debug,
            )
        except TurnInterrupted as intr:
            interrupted = True
            partial = str(intr.partial_reply or "").strip()
            if partial:
                reply_text = "[interrupted] Partial output before stop:\n" + partial
            else:
                reply_text = "[interrupted] Task stopped before producing output."

        turn_count += 1
        last_user_request = request_text
        last_reply = str(reply_text or "").strip()
        last_error = last_reply if last_reply.startswith("ERR:") else ""
        if last_reply:
            _append_history_turn(
                history,
                user_text=request_text,
                assistant_text=last_reply,
                limit=SESSION_MESSAGE_LIMIT,
            )

            if not str(active_session.get("title") or "").strip():
                active_session["title"] = _build_session_title(request_text)
            active_session["updated_at"] = _now_iso()
            active_session["messages"] = list(history)

            internal_turns = active_session.get("internal_turns")
            turns = list(internal_turns) if isinstance(internal_turns, list) else []
            turns.append(
                {
                    "ts": _now_iso(),
                    "user": request_text,
                    "assistant": last_reply,
                    "interrupted": bool(interrupted),
                }
            )
            if len(turns) > INTERNAL_TURN_LIMIT:
                turns = turns[-INTERNAL_TURN_LIMIT:]
            active_session["internal_turns"] = turns

            _upsert_session(history_store, active_session)
            _save_history_store(history_file, history_store)
        if interrupted:
            if raise_on_interrupt:
                raise KeyboardInterrupt
            print("\nInterrupted current task. Partial context saved.\n")
        return last_reply

    if args.prompt:
        user_text = " ".join(args.prompt).strip()
        if not user_text:
            return 1
        try:
            _run_and_record(user_text, raise_on_interrupt=True)
        except KeyboardInterrupt:
            print("\nInterrupted current task.")
            return 130
        return 0

    if not args.no_banner:
        _print_cli_banner()
        print("Memento-S CLI (MVP)")
        print("Type /help for commands.\n")

    if not runner.has_skills():
        print("Warning: no skills loaded from AGENTS.md; fallback chat will still work.\n")

    if bool(args.prewarm):
        _run_router_prewarm(runner, raw_args="auto", debug=debug)

    prompt_session = _build_prompt_session()
    if prompt_session is None:
        print("Tip: install `prompt_toolkit` to enable dynamic slash menu while typing `/`.\n")

    while True:
        try:
            if prompt_session is not None:
                user_text = str(prompt_session.prompt("You> ")).strip()
            else:
                user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not user_text:
            continue
        if user_text == "/":
            _print_slash_menu()
            continue
        if user_text in EXIT_COMMANDS:
            print("Bye.")
            return 0
        if user_text in HELP_COMMANDS:
            _print_help()
            continue
        cmd = user_text.strip().split(maxsplit=1)
        if user_text in STATUS_COMMANDS:
            current_title = str(active_session.get("title") or "").strip()
            _print_status(
                runner=runner,
                max_steps=max_steps,
                create_on_miss=bool(args.create_on_miss),
                optimize_on_error=bool(args.optimize_on_error),
                optimize_attempts=optimize_attempts,
                debug=debug,
                turn_count=turn_count,
                history=history,
                session_title=current_title,
                last_user_request=last_user_request,
                last_reply=last_reply,
                last_error=last_error,
            )
            continue
        if cmd and cmd[0] in SKILLS_COMMANDS:
            arg = cmd[1].strip() if len(cmd) > 1 else ""
            if not arg:
                try:
                    if prompt_session is not None:
                        arg = str(prompt_session.prompt("Skills(query)> ")).strip()
                    else:
                        arg = input("Skills(query)> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue
                if not arg:
                    _print_skills(runner)
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
                _print_skills(runner)
                continue
            _print_cloud_skills(query_text, top_k=top_k)
            continue
        if cmd and cmd[0] in PREWARM_COMMANDS:
            raw_args = cmd[1] if len(cmd) > 1 else ""
            _run_router_prewarm(runner, raw_args=raw_args, debug=debug)
            continue
        if cmd and cmd[0] in CONFIG_COMMANDS:
            raw_args = cmd[1] if len(cmd) > 1 else ""
            _handle_config_command(raw_args, env_path=env_file)
            continue
        if cmd and cmd[0].startswith("/") and cmd[0] not in KNOWN_SLASH_COMMANDS:
            _print_slash_suggestions(cmd[0])
            continue
        if cmd and cmd[0] in HISTORY_COMMANDS:
            sessions = _collect_history_sessions(
                history_store,
                active_session=active_session,
                history=history,
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
                    active_session = selected
                    history = loaded_history[-SESSION_MESSAGE_LIMIT:]
                    active_session["messages"] = list(history)
                    runner.reset_context()
                    turn_count = len(history) // 2
                    last_user_request, last_reply, last_error = _extract_last_turn_fields(history)
                    loaded_title = str(active_session.get("title") or "").strip() or "Untitled Session"
                    print(
                        f"Loaded session #{target_index}: {loaded_title} "
                        f"(messages={len(history)}, turns={turn_count}).\n"
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
        if user_text in LAST_COMMANDS:
            if last_reply:
                print(f"Assistant(last)> {last_reply}\n")
            else:
                print("No previous assistant reply.\n")
            continue
        if user_text in RETRY_COMMANDS:
            if not last_user_request:
                print("No previous user request to retry.\n")
                continue
            print(f"Tool> retrying: {last_user_request}")
            _run_and_record(last_user_request)
            continue
        if user_text in CLEAR_COMMANDS:
            history.clear()
            runner.reset_context()
            turn_count = 0
            last_user_request = ""
            last_reply = ""
            last_error = ""
            active_session = _new_session()
            print("Context/history cleared.\n")
            continue

        _run_and_record(user_text)


if __name__ == "__main__":
    raise SystemExit(main())
