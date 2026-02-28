"""Memento-S configuration.

Centralises all environment-variable lookups and compile-time constants.
Usage: ``from core.config import …``
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ===================================================================
# Bootstrap
# ===================================================================

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


# ===================================================================
# Env-var parsing helpers (private)
# ===================================================================

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


# ===================================================================
# General
# ===================================================================

DEBUG = _env_flag("DEBUG", False)

# ===================================================================
# LLM / Model
# ===================================================================

MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
LLM_API = (os.getenv("LLM_API") or "").strip().lower() or "openrouter"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MAX_TOKENS = _env_int("OPENROUTER_MAX_TOKENS", 100000)
OPENROUTER_TIMEOUT = _env_int("OPENROUTER_TIMEOUT", 60)
OPENROUTER_RETRIES = _env_int("OPENROUTER_RETRIES", 3)
OPENROUTER_RETRY_BACKOFF = _env_float("OPENROUTER_RETRY_BACKOFF", 2.0)

OPENROUTER_PROVIDER = (os.getenv("OPENROUTER_PROVIDER") or "").strip()
OPENROUTER_PROVIDER_ORDER = (os.getenv("OPENROUTER_PROVIDER_ORDER") or "").strip()
OPENROUTER_ALLOW_FALLBACKS = _env_flag("OPENROUTER_ALLOW_FALLBACKS", True)
OPENROUTER_SITE_URL = (os.getenv("OPENROUTER_SITE_URL") or "").strip()
OPENROUTER_APP_NAME = (os.getenv("OPENROUTER_APP_NAME") or "").strip()

LLM_MAX_CALLS_PER_TURN = max(1, _env_int("LLM_MAX_CALLS_PER_TURN", 24))
LLM_ENFORCE_CALL_BUDGET = _env_flag("LLM_ENFORCE_CALL_BUDGET", True)

# ===================================================================
# Skills & workspace
# ===================================================================

SKILLS_DIR = Path(os.getenv("SKILLS_DIR", "skills"))
SKILLS_EXTRA_DIRS = _parse_env_path_list(os.getenv("SKILLS_EXTRA_DIRS", ""))
WORKSPACE_DIR = _resolve_env_path("WORKSPACE_DIR", "workspace")

# ===================================================================
# Semantic router
# ===================================================================

SEMANTIC_ROUTER_ENABLED = _env_flag("SEMANTIC_ROUTER_ENABLED", True)
SEMANTIC_ROUTER_TOP_K = max(1, _env_int("SEMANTIC_ROUTER_TOP_K", 4))
SEMANTIC_ROUTER_METHOD = (os.getenv("SEMANTIC_ROUTER_METHOD") or "bm25").strip().lower()
SEMANTIC_ROUTER_DEBUG = _env_flag("SEMANTIC_ROUTER_DEBUG", DEBUG)

SEMANTIC_ROUTER_CATALOG_JSONL = (
    os.getenv("SEMANTIC_ROUTER_CATALOG_JSONL") or "router_data/skills_catalog.jsonl"
).strip()

# Qwen embedding model paths
SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH = (os.getenv("SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH") or "").strip()
SEMANTIC_ROUTER_QWEN_MODEL_PATH = (os.getenv("SEMANTIC_ROUTER_QWEN_MODEL_PATH") or "").strip()
SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH = (
    os.getenv("SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH") or ""
).strip()
SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH = (
    os.getenv("SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH") or ""
).strip()

# Embedding parameters
SEMANTIC_ROUTER_EMBED_MAX_LENGTH = max(256, _env_int("SEMANTIC_ROUTER_EMBED_MAX_LENGTH", 8192))
SEMANTIC_ROUTER_EMBED_BATCH_SIZE = max(1, _env_int("SEMANTIC_ROUTER_EMBED_BATCH_SIZE", 128))
SEMANTIC_ROUTER_EMBED_CACHE_DIR = _resolve_env_path("SEMANTIC_ROUTER_EMBED_CACHE_DIR", "router_data/embeddings")
SEMANTIC_ROUTER_EMBED_PREWARM = _env_flag("SEMANTIC_ROUTER_EMBED_PREWARM", True)
SEMANTIC_ROUTER_EMBED_QUERY_INSTRUCTION = (
    os.getenv("SEMANTIC_ROUTER_EMBED_QUERY_INSTRUCTION")
    or "Given a user query, retrieve relevant skill descriptions that match the query"
).strip()

ROUTER_DYNAMIC_GAP_ENABLED = _env_flag("ROUTER_DYNAMIC_GAP_ENABLED", True)
ROUTER_DYNAMIC_GAP_MAX_CHARS = max(400, _env_int("ROUTER_DYNAMIC_GAP_MAX_CHARS", 2400))

# ===================================================================
# Skill dynamic fetch
# ===================================================================

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

# ===================================================================
# Execution logging
# ===================================================================

EXEC_LOG_ENABLED = _env_flag("EXEC_LOG_ENABLED", False)
EXEC_LOG_DIR = Path(os.getenv("EXEC_LOG_DIR", "logs"))
EXEC_LOG_MAX_CHARS = max(0, _env_int("EXEC_LOG_MAX_CHARS", 0))

# ===================================================================
# Agent system prompt
# ===================================================================
# Use {workspace} as placeholder — filled at runtime by MementoAgent.__init__.

AGENT_SYSTEM_PROMPT_TEMPLATE = (
    "You are Memento-S, an intelligent assistant with a skill system.\n"
    "\n"
    "## Working directory\n"
    "Your working directory is `{workspace}`. "
    "**Always** use relative paths (e.g. `output/report.md`) or paths "
    "under this directory for any files you create, read, or modify. "
    "NEVER write to `/tmp` or other system directories.\n"
    "\n"
    "## Core tools\n"
    "You have tools for: bash commands, editing files, creating files, "
    "and viewing files/directories.\n"
    "\n"
    "## Skill system\n"
    "Relevant skills are automatically suggested in user messages under "
    "a `[Matched Skills]` section. Each skill has a name and description.\n"
    "\n"
    "**When you see matched skills relevant to the task:**\n"
    "1. Use `read_skill` with the skill name to learn how it works.\n"
    "2. Execute the skill's scripts/commands via `bash_tool`.\n"
    "\n"
    "**IMPORTANT — when to use skills:**\n"
    "If you are not one hundred percent certain about the answer, or the question "
    "involves specific people, organizations, current events, or facts "
    "you are not fully confident about, always use a matched skill "
    "(such as web search) rather than guessing.\n"
    "\n"
    "**When no matched skills appear or none are relevant:**\n"
    "If the task involves a repeatable workflow, pipeline, or specialized "
    "operation that would benefit from a reusable skill, do NOT simply "
    "answer from your own knowledge. Instead:\n"
    "1. Use `read_skill` with `skill-creator` to learn the skill creation workflow.\n"
    "2. Follow the skill-creator guidance to create a new skill under "
    "`skill_extra/` (e.g. `skill_extra/my-new-skill/SKILL.md`).\n"
    "3. Immediately use the newly created skill to complete the current task.\n"
    "\n"
    "Only answer from your own knowledge or use core tools directly when "
    "the task is a simple one-off question or action that does not warrant "
    "a reusable skill.\n"
    "\n"
    "Use the tools to accomplish the user's request. "
    "Be concise but thorough."
)
