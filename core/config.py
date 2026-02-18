"""Configuration and constants extracted from agent.py.

This module centralises every environment-variable lookup, compile-time
constant, and small helper used to derive them so that the rest of the
codebase can simply ``from config import …``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = (PROJECT_ROOT / ".env").resolve()
load_dotenv(dotenv_path=ENV_FILE)
_CONFIG_VERSION = 0


def refresh_runtime_config(*, override: bool = True) -> int:
    """Reload environment values from .env and bump runtime config version."""
    global _CONFIG_VERSION
    load_dotenv(dotenv_path=ENV_FILE, override=override)
    _CONFIG_VERSION += 1
    return _CONFIG_VERSION


def get_runtime_config_version() -> int:
    return _CONFIG_VERSION


# ---------------------------------------------------------------------------
# Helpers for parsing env vars
# ---------------------------------------------------------------------------
def _parse_env_path_list(raw: str) -> tuple[Path, ...]:
    if not isinstance(raw, str) or not raw.strip():
        return ()
    out: list[Path] = []
    seen: set[str] = set()
    for part in raw.split(os.pathsep):
        for chunk in part.split(","):
            p = chunk.strip()
            if not p:
                continue
            path = Path(p).expanduser()
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
    return tuple(out)


def _resolve_env_path(name: str, default: str) -> Path:
    raw = os.getenv(name, default)
    text = str(raw or "").strip() or default
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Model / LLM provider
# ---------------------------------------------------------------------------
MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

_LLM_API_ENV = (os.getenv("LLM_API") or "").strip().lower()
LLM_API = _LLM_API_ENV or "openrouter"

# ---------------------------------------------------------------------------
# Skills / Agents
# ---------------------------------------------------------------------------
AGENTS_MD = os.getenv("AGENTS_MD", "AGENTS.md")
SKILLS_DIR = Path(os.getenv("SKILLS_DIR", "skills"))
SKILLS_EXTRA_DIRS = _parse_env_path_list(os.getenv("SKILLS_EXTRA_DIRS", ""))
WORKSPACE_DIR = _resolve_env_path("WORKSPACE_DIR", "workspace")

# ---------------------------------------------------------------------------
# LLM API
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MAX_TOKENS = _env_int("OPENROUTER_MAX_TOKENS", 100000)
OPENROUTER_TIMEOUT = _env_int("OPENROUTER_TIMEOUT", 60)
OPENROUTER_RETRIES = _env_int("OPENROUTER_RETRIES", 3)
OPENROUTER_RETRY_BACKOFF = _env_float("OPENROUTER_RETRY_BACKOFF", 2.0)
LLM_MAX_CALLS_PER_TURN = max(1, _env_int("LLM_MAX_CALLS_PER_TURN", 24))
LLM_ENFORCE_CALL_BUDGET = _env_flag("LLM_ENFORCE_CALL_BUDGET", True)

# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------
OPENROUTER_PROVIDER = (os.getenv("OPENROUTER_PROVIDER") or "").strip()
OPENROUTER_PROVIDER_ORDER = (os.getenv("OPENROUTER_PROVIDER_ORDER") or "").strip()
OPENROUTER_ALLOW_FALLBACKS = _env_flag("OPENROUTER_ALLOW_FALLBACKS", True)
OPENROUTER_SITE_URL = (os.getenv("OPENROUTER_SITE_URL") or "").strip()
OPENROUTER_APP_NAME = (os.getenv("OPENROUTER_APP_NAME") or "").strip()

# ---------------------------------------------------------------------------
# Semantic router
# ---------------------------------------------------------------------------
DEBUG = _env_flag("DEBUG", False)
SEMANTIC_ROUTER_ENABLED = _env_flag("SEMANTIC_ROUTER_ENABLED", True)
SEMANTIC_ROUTER_TOP_K = max(1, _env_int("SEMANTIC_ROUTER_TOP_K", 4))
SEMANTIC_ROUTER_METHOD = (os.getenv("SEMANTIC_ROUTER_METHOD") or "bm25").strip().lower()
SEMANTIC_ROUTER_DEBUG = _env_flag("SEMANTIC_ROUTER_DEBUG", DEBUG)
SEMANTIC_ROUTER_WRITE_VISIBLE_AGENTS = _env_flag("SEMANTIC_ROUTER_WRITE_VISIBLE_AGENTS", False)
SEMANTIC_ROUTER_CATALOG_MD = (os.getenv("SEMANTIC_ROUTER_CATALOG_MD") or "").strip()
SEMANTIC_ROUTER_CATALOG_JSONL = (
    os.getenv("SEMANTIC_ROUTER_CATALOG_JSONL") or "router_data/skills_catalog.jsonl"
).strip()
SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH = (os.getenv("SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH") or "").strip()
SEMANTIC_ROUTER_QWEN_MODEL_PATH = (os.getenv("SEMANTIC_ROUTER_QWEN_MODEL_PATH") or "").strip()
SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH = (
    os.getenv("SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH") or ""
).strip()
SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH = (
    os.getenv("SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH") or ""
).strip()
SEMANTIC_ROUTER_EMBED_MAX_LENGTH = max(256, _env_int("SEMANTIC_ROUTER_EMBED_MAX_LENGTH", 8192))
SEMANTIC_ROUTER_EMBED_BATCH_SIZE = max(1, _env_int("SEMANTIC_ROUTER_EMBED_BATCH_SIZE", 128))
SEMANTIC_ROUTER_EMBED_CACHE_DIR = _resolve_env_path(
    "SEMANTIC_ROUTER_EMBED_CACHE_DIR",
    "router_data/embeddings",
)
SEMANTIC_ROUTER_EMBED_PREWARM = _env_flag("SEMANTIC_ROUTER_EMBED_PREWARM", True)
SEMANTIC_ROUTER_EMBED_QUERY_INSTRUCTION = (
    os.getenv("SEMANTIC_ROUTER_EMBED_QUERY_INSTRUCTION")
    or "Given a user query, retrieve relevant skill descriptions that match the query"
).strip()
ROUTER_DYNAMIC_GAP_ENABLED = _env_flag("ROUTER_DYNAMIC_GAP_ENABLED", True)
ROUTER_DYNAMIC_GAP_MAX_CHARS = max(400, _env_int("ROUTER_DYNAMIC_GAP_MAX_CHARS", 2400))
_DEFAULT_BASE_SKILLS = "filesystem,terminal,web-search,uv-pip-install,skill-creator"
SEMANTIC_ROUTER_BASE_SKILLS = tuple(
    s.strip()
    for s in (os.getenv("SEMANTIC_ROUTER_BASE_SKILLS") or _DEFAULT_BASE_SKILLS).split(",")
    if s.strip()
)

# ---------------------------------------------------------------------------
# Skill dynamic fetch
# ---------------------------------------------------------------------------
SKILL_DYNAMIC_FETCH_ENABLED = _env_flag("SKILL_DYNAMIC_FETCH_ENABLED", True)
SKILL_DYNAMIC_FETCH_CATALOG_JSONL = (
    os.getenv("SKILL_DYNAMIC_FETCH_CATALOG_JSONL")
    or SEMANTIC_ROUTER_CATALOG_JSONL
    or "router_data/skills_catalog.jsonl"
).strip()
_DEFAULT_DYNAMIC_SKILL_ROOT = str(SKILLS_EXTRA_DIRS[0]) if SKILLS_EXTRA_DIRS else "skill_extra"
SKILL_DYNAMIC_FETCH_ROOT = Path(
    (os.getenv("SKILL_DYNAMIC_FETCH_ROOT") or _DEFAULT_DYNAMIC_SKILL_ROOT).strip()
).expanduser()
SKILL_DYNAMIC_FETCH_TIMEOUT_SEC = max(30, _env_int("SKILL_DYNAMIC_FETCH_TIMEOUT_SEC", 180))
SKILL_DYNAMIC_FETCH_ALLOWED_REPOS = tuple(
    s.strip()
    for s in (os.getenv("SKILL_DYNAMIC_FETCH_ALLOWED_REPOS") or "").replace(";", ",").split(",")
    if s.strip()
)

# ---------------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------------
CLI_CREATE_ON_MISS = _env_flag("CLI_CREATE_ON_MISS", True)

# ---------------------------------------------------------------------------
# Op-type sets (used by the bridge dispatcher)
# ---------------------------------------------------------------------------
FILESYSTEM_OP_TYPES = {
    "read_file",
    "write_file",
    "edit_file",
    "replace_text",
    "append_file",
    "mkdir",
    "list_directory",
    "directory_tree",
    "create_directory",
    "move_file",
    "copy_file",
    "delete_file",
    "file_info",
    "search_files",
    "file_exists",
}

TERMINAL_OP_TYPES = {
    "run_command",
    "shell",
    "ensure_uv_available",
    "setup_initial_env_with_uv",
    "setup_initial_env_with_venv",
    "clone_current_environment",
    "is_uv_environment",
    "check_nodejs_availability",
}

WEB_OP_TYPES = {
    "web_search",
    "google_search",
    "search",
    "fetch",
    "fetch_url",
    "fetch_markdown",
}

UV_PIP_OP_TYPES = {"check", "install", "list"}

# ---------------------------------------------------------------------------
# Built-in bridge skills & local-path helpers
# ---------------------------------------------------------------------------
BUILTIN_BRIDGE_SKILLS = {
    "skill-creator",
    "filesystem",
    "terminal",
    "web-search",
    "uv-pip-install",
}

SKILL_LOCAL_DIR_PREFIXES = ("scripts", "references", "assets", "templates", "examples")

SKILL_LOCAL_COMMAND_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])((?:\\./)?(?:scripts|references|assets|templates|examples)/[^\s\"'|&;<>`]+)"
)

# ---------------------------------------------------------------------------
# Execution logging
# ---------------------------------------------------------------------------
EXEC_LOG_ENABLED = _env_flag("EXEC_LOG_ENABLED", False)
EXEC_LOG_DIR = Path(os.getenv("EXEC_LOG_DIR", "logs"))
EXEC_LOG_MAX_CHARS = max(0, _env_int("EXEC_LOG_MAX_CHARS", 0))

# ---------------------------------------------------------------------------
# Chat / workflow constants
# ---------------------------------------------------------------------------
CHAT_SYSTEM_PROMPT = """You are Memento-S, an intelligent assistant.
You can help with coding, analysis, and general questions.
Be concise but thorough. Use markdown for code blocks."""

MAX_WORKFLOW_STEPS = 50  # Safety limit to prevent infinite loops

# ---------------------------------------------------------------------------
# Router action constants
# ---------------------------------------------------------------------------
ROUTER_ACTION_NEXT_STEP = "next_step"
ROUTER_ACTION_DONE = "done"
ROUTER_ACTION_NONE = "none"

STEP_SUMMARY_MAX_TOKENS = _env_int("STEP_SUMMARY_MAX_TOKENS", 2000)
STEP_SUMMARY_THRESHOLD = _env_int("STEP_SUMMARY_THRESHOLD", 15000)

SKILL_LOOP_FEEDBACK_CHARS = max(200, _env_int("SKILL_LOOP_FEEDBACK_CHARS", 2000))

# ---------------------------------------------------------------------------
# Terminal toolkit (optional dependency)
# NOTE: terminal_utils import moved to core/skill_engine/skill_executor.py
# for lazy loading. Import it there, not here.
# ---------------------------------------------------------------------------
