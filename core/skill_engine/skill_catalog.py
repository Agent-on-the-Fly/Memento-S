"""Skill catalog/XML parsing and semantic router functions.

Extracted from agent.py to improve modularity.
"""

import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from hashlib import sha1
from math import log, sqrt
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from core.config import (
    AGENTS_MD,
    PROJECT_ROOT,
    ROUTER_DYNAMIC_GAP_ENABLED,
    ROUTER_DYNAMIC_GAP_MAX_CHARS,
    SEMANTIC_ROUTER_METHOD,
    SEMANTIC_ROUTER_BASE_SKILLS,
    SEMANTIC_ROUTER_CATALOG_JSONL,
    SEMANTIC_ROUTER_CATALOG_MD,
    SEMANTIC_ROUTER_DEBUG,
    SEMANTIC_ROUTER_EMBED_BATCH_SIZE,
    SEMANTIC_ROUTER_EMBED_CACHE_DIR,
    SEMANTIC_ROUTER_EMBED_MAX_LENGTH,
    SEMANTIC_ROUTER_EMBED_PREWARM,
    SEMANTIC_ROUTER_EMBED_QUERY_INSTRUCTION,
    SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH,
    SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH,
    SEMANTIC_ROUTER_QWEN_MODEL_PATH,
    SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH,
    SEMANTIC_ROUTER_ENABLED,
    SEMANTIC_ROUTER_TOP_K,
    SEMANTIC_ROUTER_WRITE_VISIBLE_AGENTS,
)
from core.utils.json_utils import parse_json_output
from core.llm import openrouter_messages
from core.utils.logging_utils import log_event
from core.utils.path_utils import _truncate_middle, _xml_escape

# ---------------------------------------------------------------------------
# Skills XML functions
# ---------------------------------------------------------------------------


def load_available_skills_block_from(path: str) -> str:
    """Read an <available_skills> XML block from *path*."""
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r"<available_skills>.*?</available_skills>", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"{path} missing <available_skills>. Run: npx openskills sync")
    return m.group(0)


def load_available_skills_block() -> str:
    """Load the <available_skills> block from AGENTS_MD."""
    return load_available_skills_block_from(AGENTS_MD)


def write_visible_skills_block(skills_xml: str, target_path: str = AGENTS_MD) -> None:
    """
    Replace (or append) <available_skills> block in AGENTS.md with a routed visible subset.
    This is optional and guarded by SEMANTIC_ROUTER_WRITE_VISIBLE_AGENTS.
    """
    global _LAST_VISIBLE_AGENTS_SIG
    xml = str(skills_xml or "").strip()
    if not xml:
        return

    signature = sha1(f"{target_path}\n{xml}".encode("utf-8")).hexdigest()
    if signature == _LAST_VISIBLE_AGENTS_SIG:
        return

    path = Path(target_path)
    try:
        original = path.read_text(encoding="utf-8") if path.exists() else ""
        m = re.search(r"<available_skills>.*?</available_skills>", original, re.DOTALL)
        if m:
            updated = original[: m.start()] + xml + original[m.end() :]
        else:
            if original and not original.endswith("\n"):
                original += "\n"
            updated = (original + "\n" if original.strip() else "") + xml + "\n"
        if updated != original:
            path.write_text(updated, encoding="utf-8")
        _LAST_VISIBLE_AGENTS_SIG = signature
    except Exception as exc:
        print(f"[warn] failed to write visible skills to {target_path!r}: {exc}")


def parse_available_skills(skills_xml: str) -> list[dict]:
    """Parse <skill> blocks from an <available_skills> XML string."""
    skills: list[dict] = []
    for block in re.findall(r"<skill>.*?</skill>", skills_xml, re.DOTALL):
        name_m = re.search(r"<name>(.*?)</name>", block, re.DOTALL)
        desc_m = re.search(r"<description>(.*?)</description>", block, re.DOTALL)
        if not name_m:
            continue
        skills.append(
            {
                "name": name_m.group(1).strip(),
                "description": (desc_m.group(1).strip() if desc_m else ""),
            }
        )
    return skills


def build_available_skills_xml(skills: list[dict]) -> str:
    """Build an <available_skills> XML string from a list of skill dicts."""
    lines = ["<available_skills>", ""]
    for s in skills:
        name = _xml_escape(str(s.get("name") or "").strip())
        if not name:
            continue
        desc = _xml_escape(str(s.get("description") or "").strip())
        lines.extend(
            [
                "<skill>",
                f"<name>{name}</name>",
                f"<description>{desc}</description>",
                "<location>project</location>",
                "</skill>",
                "",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Semantic router core tokenization/index utilities
# ---------------------------------------------------------------------------

_ROUTER_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "this",
    "these",
    "those",
    "need",
    "needs",
    "using",
    "use",
    "help",
    "please",
}


def _tokenize_for_semantic(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [tok for tok in raw if tok and tok not in _ROUTER_STOPWORDS and len(tok) > 1]


def _catalog_signature(skills: list[dict]) -> str:
    digest = sha1()
    for s in skills:
        name = str(s.get("name") or "").strip()
        desc = str(s.get("description") or "").strip()
        digest.update(name.encode("utf-8", errors="ignore"))
        digest.update(b"|")
        digest.update(desc.encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()


# Module-level mutable state for semantic index caching
_SEMANTIC_INDEX_SIG: str | None = None
_SEMANTIC_INDEX: dict[str, Any] | None = None
_SEMANTIC_INDEX_SOURCE_ID: int | None = None
_BM25_INDEX_SIG: str | None = None
_BM25_INDEX: dict[str, Any] | None = None
_BM25_INDEX_SOURCE_ID: int | None = None
_LAST_VISIBLE_AGENTS_SIG: str | None = None
_JSONL_CATALOG_CACHE: dict[str, dict[str, Any]] = {}
_JSONL_CATALOG_LOCK = threading.Lock()
_EMBEDDING_DOC_CACHE: dict[str, dict[str, Any]] = {}
_EMBEDDING_RUNTIME_CACHE: dict[str, dict[str, Any]] = {}
_EMBEDDING_LOCK = threading.Lock()
_EMBEDDING_PREWARM_LOCK = threading.Lock()
_EMBEDDING_PREWARM_IN_PROGRESS: set[str] = set()
_EMBEDDING_PREWARM_COMPLETED: set[str] = set()
_DOTENV_STATE_LOCK = threading.Lock()
_DOTENV_LAST_MTIME_NS: int | None = None
_DOTENV_PATH = (PROJECT_ROOT / ".env").resolve()


def _refresh_dotenv_if_changed() -> None:
    global _DOTENV_LAST_MTIME_NS
    try:
        st = _DOTENV_PATH.stat()
    except Exception:
        return

    with _DOTENV_STATE_LOCK:
        if _DOTENV_LAST_MTIME_NS == st.st_mtime_ns:
            return
        load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
        _DOTENV_LAST_MTIME_NS = st.st_mtime_ns


def _env_str(name: str, fallback: str = "") -> str:
    _refresh_dotenv_if_changed()
    val = os.getenv(name)
    if val is None:
        return str(fallback or "").strip()
    return str(val).strip()


def _env_int(name: str, fallback: int) -> int:
    raw = _env_str(name, str(fallback))
    try:
        return int(raw)
    except Exception:
        return int(fallback)


def _env_flag(name: str, fallback: bool) -> bool:
    raw = _env_str(name, "1" if fallback else "0").lower()
    return raw not in {"0", "false", "no", "off"}


def _router_method() -> str:
    return (_env_str("SEMANTIC_ROUTER_METHOD", SEMANTIC_ROUTER_METHOD) or "bm25").lower()


def _router_embed_max_length() -> int:
    return max(256, _env_int("SEMANTIC_ROUTER_EMBED_MAX_LENGTH", SEMANTIC_ROUTER_EMBED_MAX_LENGTH))


def _router_embed_batch_size() -> int:
    return max(1, _env_int("SEMANTIC_ROUTER_EMBED_BATCH_SIZE", SEMANTIC_ROUTER_EMBED_BATCH_SIZE))


def _router_embed_query_instruction() -> str:
    return _env_str(
        "SEMANTIC_ROUTER_EMBED_QUERY_INSTRUCTION",
        SEMANTIC_ROUTER_EMBED_QUERY_INSTRUCTION,
    ) or "Given a user query, retrieve relevant skill descriptions that match the query"


def _router_embed_prewarm_enabled() -> bool:
    return _env_flag("SEMANTIC_ROUTER_EMBED_PREWARM", SEMANTIC_ROUTER_EMBED_PREWARM)


def _router_embed_cache_dir() -> Path:
    raw = _env_str("SEMANTIC_ROUTER_EMBED_CACHE_DIR", str(SEMANTIC_ROUTER_EMBED_CACHE_DIR))
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def _build_semantic_index(skills: list[dict]) -> dict[str, Any]:
    docs_tf: list[Counter] = []
    df: Counter = Counter()
    postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    name_tokens: list[set[str]] = []
    names_lower: list[str] = []

    for s in skills:
        name = str(s.get("name") or "").strip()
        desc = str(s.get("description") or "").strip()
        tokens = _tokenize_for_semantic(f"{name} {desc}")
        tf = Counter(tokens)
        docs_tf.append(tf)
        for tok in tf:
            df[tok] += 1
        name_tokens.append(set(_tokenize_for_semantic(name)))
        names_lower.append(name.lower())

    n_docs = max(1, len(skills))
    idf = {tok: log((1.0 + n_docs) / (1.0 + float(freq))) + 1.0 for tok, freq in df.items()}
    doc_norms: list[float] = []

    for doc_idx, tf in enumerate(docs_tf):
        norm_sq = 0.0
        for tok, cnt in tf.items():
            weight = (1.0 + log(float(cnt))) * idf.get(tok, 0.0)
            postings[tok].append((doc_idx, weight))
            norm_sq += weight * weight
        doc_norms.append(sqrt(norm_sq) if norm_sq > 0.0 else 1.0)

    return {
        "idf": idf,
        "postings": postings,
        "doc_norms": doc_norms,
        "name_tokens": name_tokens,
        "names_lower": names_lower,
    }


def _get_semantic_index(skills: list[dict]) -> dict[str, Any]:
    global _SEMANTIC_INDEX_SIG, _SEMANTIC_INDEX, _SEMANTIC_INDEX_SOURCE_ID
    if _SEMANTIC_INDEX is not None and _SEMANTIC_INDEX_SOURCE_ID == id(skills):
        return _SEMANTIC_INDEX
    sig = _catalog_signature(skills)
    if _SEMANTIC_INDEX is not None and _SEMANTIC_INDEX_SIG == sig:
        _SEMANTIC_INDEX_SOURCE_ID = id(skills)
        return _SEMANTIC_INDEX
    _SEMANTIC_INDEX = _build_semantic_index(skills)
    _SEMANTIC_INDEX_SIG = sig
    _SEMANTIC_INDEX_SOURCE_ID = id(skills)
    return _SEMANTIC_INDEX


def select_semantic_top_skills(
    goal_text: str, skills: list[dict], top_k: int = SEMANTIC_ROUTER_TOP_K
) -> list[dict]:
    if not skills:
        return []

    top_k = max(1, min(int(top_k), len(skills)))
    index = _get_semantic_index(skills)
    q_tokens = _tokenize_for_semantic(goal_text)
    q_tf: Counter = Counter(q_tokens)

    base_names = {name for name in SEMANTIC_ROUTER_BASE_SKILLS if name}
    name_to_skill = {str(s.get("name") or "").strip(): s for s in skills}
    forced = [name_to_skill[n] for n in SEMANTIC_ROUTER_BASE_SKILLS if n in name_to_skill]

    if not q_tf:
        selected = forced[:]
        for s in skills:
            if s not in selected:
                selected.append(s)
            if len(selected) >= top_k + len(forced):
                break
        return selected

    idf = index["idf"]
    postings = index["postings"]
    doc_norms = index["doc_norms"]
    name_tokens = index["name_tokens"]
    names_lower = index["names_lower"]

    q_weights: dict[str, float] = {}
    q_norm_sq = 0.0
    for tok, cnt in q_tf.items():
        tok_idf = idf.get(tok)
        if tok_idf is None:
            continue
        weight = (1.0 + log(float(cnt))) * tok_idf
        q_weights[tok] = weight
        q_norm_sq += weight * weight

    if not q_weights:
        selected = forced[:]
        for s in skills:
            if s not in selected:
                selected.append(s)
            if len(selected) >= top_k + len(forced):
                break
        return selected

    q_norm = sqrt(q_norm_sq) if q_norm_sq > 0.0 else 1.0
    scores: dict[int, float] = defaultdict(float)
    for tok, q_w in q_weights.items():
        for doc_idx, d_w in postings.get(tok, []):
            scores[doc_idx] += q_w * d_w

    goal_lower = str(goal_text or "").lower()
    ranked: list[tuple[float, int]] = []
    for doc_idx, dot in scores.items():
        cosine = dot / (q_norm * doc_norms[doc_idx])
        # Bias toward exact skill-name hits and partial name-token hits.
        bonus = 0.0
        skill_name_l = names_lower[doc_idx]
        if skill_name_l and skill_name_l in goal_lower:
            bonus += 0.35
        overlap = len(name_tokens[doc_idx].intersection(q_tf.keys()))
        if overlap:
            bonus += min(0.2, 0.05 * overlap)
        ranked.append((cosine + bonus, doc_idx))

    ranked.sort(key=lambda x: x[0], reverse=True)
    chosen: list[dict] = []
    seen_names: set[str] = set()
    for score, doc_idx in ranked[: max(top_k * 3, top_k)]:
        if score <= 0:
            continue
        skill = skills[doc_idx]
        name = str(skill.get("name") or "").strip()
        if not name or name in seen_names:
            continue
        chosen.append(skill)
        seen_names.add(name)
        if len(chosen) >= top_k:
            break

    for skill in forced:
        name = str(skill.get("name") or "").strip()
        if name and name not in seen_names:
            chosen.append(skill)
            seen_names.add(name)

    # Fallback fill (rare): ensure we always provide enough candidates.
    if len(chosen) < min(len(skills), top_k):
        for s in skills:
            name = str(s.get("name") or "").strip()
            if not name or name in seen_names:
                continue
            chosen.append(s)
            seen_names.add(name)
            if len(chosen) >= top_k + len(forced):
                break

    return chosen


# ---------------------------------------------------------------------------
# Semantic router methods: BM25 / Embedding / Dispatcher
# ---------------------------------------------------------------------------


def _resolve_forced_skills(skills: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    name_to_skill = {
        str(s.get("name") or "").strip(): s
        for s in skills
        if isinstance(s, dict) and str(s.get("name") or "").strip()
    }
    forced = [name_to_skill[n] for n in SEMANTIC_ROUTER_BASE_SKILLS if n in name_to_skill]
    return forced, name_to_skill


def _append_forced_skills_and_fill(
    chosen: list[dict],
    skills: list[dict],
    *,
    top_k: int,
    forced: list[dict],
) -> list[dict]:
    seen_names: set[str] = {
        str(s.get("name") or "").strip() for s in chosen if isinstance(s, dict)
    }
    for skill in forced:
        name = str(skill.get("name") or "").strip()
        if name and name not in seen_names:
            chosen.append(skill)
            seen_names.add(name)

    if len(chosen) < min(len(skills), top_k):
        for skill in skills:
            name = str(skill.get("name") or "").strip()
            if not name or name in seen_names:
                continue
            chosen.append(skill)
            seen_names.add(name)
            if len(chosen) >= top_k + len(forced):
                break
    return chosen


def _tokenize_for_bm25(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    # Chinese/Japanese heavy text: prefer jieba if available.
    if re.search(r"[\u4e00-\u9fff]", raw):
        try:
            import jieba

            tokens = [tok.strip() for tok in jieba.cut(raw) if str(tok).strip()]
            if tokens:
                return tokens
        except Exception:
            pass

    tokens = _tokenize_for_semantic(raw)
    if tokens:
        return tokens
    return [tok for tok in re.split(r"\s+", raw.lower()) if tok]


def _build_bm25_index(skills: list[dict]) -> dict[str, Any] | None:
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        return None

    docs_tokens: list[list[str]] = []
    name_tokens: list[set[str]] = []
    names_lower: list[str] = []

    for s in skills:
        name = str(s.get("name") or "").strip()
        desc = str(s.get("description") or "").strip()
        doc_tokens = _tokenize_for_bm25(f"{name} {desc}")
        if not doc_tokens:
            doc_tokens = ["_"]
        docs_tokens.append(doc_tokens)
        name_tokens.append(set(_tokenize_for_bm25(name)))
        names_lower.append(name.lower())

    bm25 = BM25Okapi(docs_tokens)
    return {
        "bm25": bm25,
        "name_tokens": name_tokens,
        "names_lower": names_lower,
    }


def _get_bm25_index(skills: list[dict]) -> dict[str, Any] | None:
    global _BM25_INDEX_SIG, _BM25_INDEX, _BM25_INDEX_SOURCE_ID
    if _BM25_INDEX is not None and _BM25_INDEX_SOURCE_ID == id(skills):
        return _BM25_INDEX
    sig = _catalog_signature(skills)
    if _BM25_INDEX is not None and _BM25_INDEX_SIG == sig:
        _BM25_INDEX_SOURCE_ID = id(skills)
        return _BM25_INDEX
    _BM25_INDEX = _build_bm25_index(skills)
    _BM25_INDEX_SIG = sig
    _BM25_INDEX_SOURCE_ID = id(skills)
    return _BM25_INDEX


def select_bm25_top_skills(
    goal_text: str,
    skills: list[dict],
    top_k: int = SEMANTIC_ROUTER_TOP_K,
) -> list[dict]:
    if not skills:
        return []

    top_k = max(1, min(int(top_k), len(skills)))
    forced, _name_to_skill = _resolve_forced_skills(skills)

    bm25_index = _get_bm25_index(skills)
    if not bm25_index:
        if SEMANTIC_ROUTER_DEBUG:
            print("[semantic-router] bm25 dependencies missing, fallback to tfidf")
        return select_semantic_top_skills(goal_text, skills, top_k=top_k)

    q_tokens = _tokenize_for_bm25(goal_text)
    if not q_tokens:
        selected = forced[:]
        return _append_forced_skills_and_fill(selected, skills, top_k=top_k, forced=forced)

    bm25 = bm25_index["bm25"]
    name_tokens = bm25_index["name_tokens"]
    names_lower = bm25_index["names_lower"]
    try:
        raw_scores = bm25.get_scores(q_tokens)
        scores = [float(v) for v in raw_scores]
    except Exception:
        if SEMANTIC_ROUTER_DEBUG:
            print("[semantic-router] bm25 scoring failed, fallback to tfidf")
        return select_semantic_top_skills(goal_text, skills, top_k=top_k)

    goal_lower = str(goal_text or "").lower()
    q_token_set = set(q_tokens)
    ranked: list[tuple[float, int]] = []
    for idx, score in enumerate(scores):
        bonus = 0.0
        skill_name_l = names_lower[idx]
        if skill_name_l and skill_name_l in goal_lower:
            bonus += 0.35
        overlap = len(name_tokens[idx].intersection(q_token_set))
        if overlap:
            bonus += min(0.2, 0.05 * overlap)
        ranked.append((score + bonus, idx))
    ranked.sort(key=lambda x: x[0], reverse=True)

    chosen: list[dict] = []
    seen_names: set[str] = set()
    for score, doc_idx in ranked[: max(top_k * 3, top_k)]:
        if score <= 0 and len(chosen) >= top_k:
            break
        skill = skills[doc_idx]
        name = str(skill.get("name") or "").strip()
        if not name or name in seen_names:
            continue
        chosen.append(skill)
        seen_names.add(name)
        if len(chosen) >= top_k:
            break

    return _append_forced_skills_and_fill(chosen, skills, top_k=top_k, forced=forced)


def _resolve_embedding_paths(method: str) -> tuple[str, str]:
    if method == "qwen_embedding":
        tokenizer_path = (
            _env_str("SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH", SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH)
            or _env_str("SEMANTIC_ROUTER_QWEN_MODEL_PATH", SEMANTIC_ROUTER_QWEN_MODEL_PATH)
        ).strip()
        model_path = _env_str("SEMANTIC_ROUTER_QWEN_MODEL_PATH", SEMANTIC_ROUTER_QWEN_MODEL_PATH).strip()
        return tokenizer_path, model_path

    if method == "memento_qwen_embedding":
        tokenizer_path = (
            _env_str(
                "SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH",
                SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH,
            )
            or _env_str("SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH", SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH)
            or _env_str("SEMANTIC_ROUTER_QWEN_MODEL_PATH", SEMANTIC_ROUTER_QWEN_MODEL_PATH)
            or _env_str(
                "SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH",
                SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH,
            )
        ).strip()
        model_path = _env_str(
            "SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH",
            SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH,
        ).strip()
        return tokenizer_path, model_path

    return "", ""


def _resolve_embedding_cache_file(method: str, model_path: str) -> Path:
    cache_dir = _router_embed_cache_dir().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_hash = sha1(str(model_path or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
    method_slug = re.sub(r"[^a-z0-9_-]+", "-", str(method or "").lower()) or "embedding"
    return cache_dir / f"skills_catalog.{method_slug}.{model_hash}.npz"


def _get_model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _last_token_pool(last_hidden_states: Any, attention_mask: Any, torch_mod: Any) -> Any:
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch_mod.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _load_embedding_runtime(tokenizer_path: str, model_path: str) -> tuple[dict[str, Any] | None, str | None]:
    cache_key = f"{tokenizer_path}::{model_path}"
    with _EMBEDDING_LOCK:
        cached = _EMBEDDING_RUNTIME_CACHE.get(cache_key)
        if isinstance(cached, dict):
            if SEMANTIC_ROUTER_DEBUG:
                print("[semantic-router] embedding runtime cache hit")
            return cached, None

    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as hf_logging
    except Exception as exc:
        return None, f"embedding dependencies missing: {type(exc).__name__}: {exc}"

    try:
        hf_logging.disable_progress_bar()
    except Exception:
        pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
    except Exception as exc:
        return None, f"failed to load tokenizer: {exc}"

    model_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        model_kwargs["dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = torch.float32

    load_errors: list[str] = []
    model = None
    t0 = time.perf_counter()
    if SEMANTIC_ROUTER_DEBUG:
        print(f"[semantic-router] loading embedding runtime model={model_path}")
    for attn_impl in ("flash_attention_2", "sdpa", None):
        try:
            kwargs = dict(model_kwargs)
            if attn_impl:
                kwargs["attn_implementation"] = attn_impl
            model = AutoModel.from_pretrained(model_path, **kwargs)
            break
        except Exception as exc:
            load_errors.append(f"{attn_impl or 'default'}: {type(exc).__name__}: {exc}")
            continue

    if model is None:
        return None, "failed to load embedding model: " + " | ".join(load_errors)

    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()

    runtime = {
        "torch": torch,
        "F": F,
        "tokenizer": tokenizer,
        "model": model,
    }
    device = _get_model_device(model)
    with _EMBEDDING_LOCK:
        _EMBEDDING_RUNTIME_CACHE[cache_key] = runtime
    if SEMANTIC_ROUTER_DEBUG:
        print(
            "[semantic-router] embedding runtime loaded in "
            f"{time.perf_counter() - t0:.2f}s (device={device})"
        )
    return runtime, None


def _encode_texts_with_embedding(
    runtime: dict[str, Any],
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    progress_hook: Callable[[int, int], None] | None = None,
) -> tuple[Any | None, str | None]:
    try:
        import numpy as np
    except Exception as exc:
        return None, f"numpy missing: {exc}"

    if not texts:
        return np.zeros((0, 0), dtype="float32"), None

    torch_mod = runtime["torch"]
    func = runtime["F"]
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    device = _get_model_device(model)
    if device is None:
        return None, "unable to determine embedding model device"

    embs: list[Any] = []
    batch_step = max(1, int(batch_size))
    total_batches = (len(texts) + batch_step - 1) // batch_step
    with torch_mod.no_grad():
        for batch_index, i in enumerate(range(0, len(texts), batch_step), start=1):
            batch_texts = texts[i : i + batch_step]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max(128, int(max_length)),
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pooled = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"], torch_mod)
            pooled = func.normalize(pooled, p=2, dim=1)
            embs.append(pooled.detach().to("cpu", dtype=torch_mod.float32))
            if progress_hook is not None:
                try:
                    progress_hook(batch_index, total_batches)
                except Exception:
                    pass

    if not embs:
        return np.zeros((0, 0), dtype="float32"), None
    arr = torch_mod.cat(embs, dim=0).numpy()
    return arr.astype("float32", copy=False), None


def _load_embedding_doc_cache(
    cache_file: Path,
    *,
    expected_catalog_sig: str,
    expected_tokenizer_path: str,
    expected_model_path: str,
    expected_names: list[str],
) -> Any | None:
    try:
        import numpy as np
    except Exception:
        return None
    if not cache_file.exists():
        return None
    try:
        data = np.load(cache_file, allow_pickle=False)
    except Exception:
        return None
    try:
        catalog_sig = str(data["catalog_sig"].item())
        tokenizer_path = str(data["tokenizer_path"].item())
        model_path = str(data["model_path"].item())
        names = [str(x) for x in data["names"].tolist()]
        embeddings = data["embeddings"].astype("float32")
    except Exception:
        return None

    if catalog_sig != expected_catalog_sig:
        return None
    if tokenizer_path != expected_tokenizer_path:
        return None
    if model_path != expected_model_path:
        return None
    if names != expected_names:
        return None
    if embeddings.shape[0] != len(expected_names):
        return None
    return embeddings


def _save_embedding_doc_cache(
    cache_file: Path,
    *,
    catalog_sig: str,
    tokenizer_path: str,
    model_path: str,
    names: list[str],
    embeddings: Any,
) -> None:
    try:
        import numpy as np
    except Exception:
        return
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            catalog_sig=np.asarray(catalog_sig),
            tokenizer_path=np.asarray(tokenizer_path),
            model_path=np.asarray(model_path),
            names=np.asarray(names, dtype=str),
            embeddings=np.asarray(embeddings, dtype="float32"),
        )
    except Exception:
        return


def _get_embedding_doc_matrix(
    skills: list[dict],
    method: str,
    *,
    show_progress: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    tokenizer_path, model_path = _resolve_embedding_paths(method)
    if not tokenizer_path or not model_path:
        return None, (
            "missing embedding model/tokenizer path in env "
            f"for method={method!r}"
        )

    names = [str(s.get("name") or "").strip() for s in skills]
    doc_texts = []
    for skill in skills:
        name = str(skill.get("name") or "").strip()
        desc = str(skill.get("description") or "").strip()
        doc_texts.append(f"Skill: {name}\nDescription: {desc}".strip())

    catalog_sig = _catalog_signature(skills)
    embed_max_length = _router_embed_max_length()
    embed_batch_size = _router_embed_batch_size()
    cache_key = f"{method}|{catalog_sig}|{tokenizer_path}|{model_path}|{embed_max_length}"
    with _EMBEDDING_LOCK:
        cached = _EMBEDDING_DOC_CACHE.get(cache_key)
        if isinstance(cached, dict):
            if SEMANTIC_ROUTER_DEBUG:
                print(f"[semantic-router] {method} doc-embedding memory cache hit")
            return cached, None

    cache_file = _resolve_embedding_cache_file(method, model_path)
    cached_embeddings = _load_embedding_doc_cache(
        cache_file,
        expected_catalog_sig=catalog_sig,
        expected_tokenizer_path=tokenizer_path,
        expected_model_path=model_path,
        expected_names=names,
    )

    if cached_embeddings is None:
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] {method} doc-embedding cache miss; encoding catalog")
        runtime, err = _load_embedding_runtime(tokenizer_path, model_path)
        if runtime is None:
            return None, err
        progress_bar: Any | None = None
        progress_hook: Callable[[int, int], None] | None = None
        if show_progress:
            total_batches = (len(doc_texts) + embed_batch_size - 1) // max(1, embed_batch_size)
            try:
                from tqdm import tqdm

                progress_bar = tqdm(
                    total=max(1, total_batches),
                    desc=f"{method}",
                    unit="batch",
                )

                def _progress_hook(done: int, total: int) -> None:
                    if progress_bar is None:
                        return
                    progress_bar.total = max(1, int(total))
                    progress_bar.n = min(int(done), progress_bar.total)
                    progress_bar.refresh()

                progress_hook = _progress_hook
            except Exception:
                progress_bar = None
                progress_hook = None
        embeddings, enc_err = _encode_texts_with_embedding(
            runtime,
            doc_texts,
            batch_size=embed_batch_size,
            max_length=embed_max_length,
            progress_hook=progress_hook,
        )
        if progress_bar is not None:
            try:
                progress_bar.close()
            except Exception:
                pass
        if embeddings is None:
            return None, enc_err
        _save_embedding_doc_cache(
            cache_file,
            catalog_sig=catalog_sig,
            tokenizer_path=tokenizer_path,
            model_path=model_path,
            names=names,
            embeddings=embeddings,
        )
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] {method} doc-embedding saved: {cache_file}")
    else:
        embeddings = cached_embeddings
        if SEMANTIC_ROUTER_DEBUG:
            shape = getattr(embeddings, "shape", None)
            print(f"[semantic-router] {method} doc-embedding disk cache hit: {cache_file} shape={shape}")

    payload: dict[str, Any] = {
        "method": method,
        "tokenizer_path": tokenizer_path,
        "model_path": model_path,
        "names": names,
        "embeddings": embeddings,
        "cache_file": str(cache_file),
    }
    with _EMBEDDING_LOCK:
        _EMBEDDING_DOC_CACHE[cache_key] = payload
    return payload, None


def _prewarm_embedding_catalog_sync(
    skills: list[dict],
    *,
    methods: tuple[str, ...] = ("qwen_embedding", "memento_qwen_embedding"),
) -> None:
    for method in methods:
        tokenizer_path, model_path = _resolve_embedding_paths(method)
        if not tokenizer_path or not model_path:
            if SEMANTIC_ROUTER_DEBUG:
                print(
                    f"[semantic-router] prewarm skip {method}: "
                    "missing tokenizer/model path"
                )
            continue

        payload, err = _get_embedding_doc_matrix(skills, method)
        if err:
            if SEMANTIC_ROUTER_DEBUG:
                print(f"[semantic-router] prewarm failed {method}: {err}")
            continue
        if SEMANTIC_ROUTER_DEBUG and payload:
            print(
                f"[semantic-router] prewarm ready {method}: "
                f"{payload.get('cache_file')}"
            )


def _router_method_to_embedding_methods(method: str) -> tuple[str, ...]:
    m = str(method or "").strip().lower()
    if m in {"qwen", "qwen3", "qwen_embedding", "qwen3_embedding"}:
        return ("qwen_embedding",)
    if m in {"memento", "memento_qwen", "memento-qwen", "memento_qwen_embedding"}:
        return ("memento_qwen_embedding",)
    return ()


def ensure_router_embedding_prewarm(
    skills: list[dict],
    *,
    methods: tuple[str, ...] | None = None,
) -> None:
    """Precompute embedding caches once per catalog signature in background."""
    if not _router_embed_prewarm_enabled() or not skills:
        return
    if methods is None:
        methods = _router_method_to_embedding_methods(_router_method())
    methods = tuple(str(x).strip() for x in methods if str(x).strip())
    if not methods:
        return

    sig = _catalog_signature(skills)
    prewarm_key = f"{sig}|{','.join(sorted(methods))}"
    with _EMBEDDING_PREWARM_LOCK:
        if (
            prewarm_key in _EMBEDDING_PREWARM_COMPLETED
            or prewarm_key in _EMBEDDING_PREWARM_IN_PROGRESS
        ):
            return
        _EMBEDDING_PREWARM_IN_PROGRESS.add(prewarm_key)

    def _worker() -> None:
        try:
            _prewarm_embedding_catalog_sync(skills, methods=methods)
        finally:
            with _EMBEDDING_PREWARM_LOCK:
                _EMBEDDING_PREWARM_IN_PROGRESS.discard(prewarm_key)
                _EMBEDDING_PREWARM_COMPLETED.add(prewarm_key)

    thread = threading.Thread(
        target=_worker,
        name=f"router-embed-prewarm-{sig[:8]}",
        daemon=True,
    )
    thread.start()


def precompute_router_embedding_cache(
    skills: list[dict],
    *,
    methods: tuple[str, ...] = ("qwen_embedding", "memento_qwen_embedding"),
    show_progress: bool = False,
) -> list[tuple[str, str]]:
    """Synchronously build embedding caches for selected methods.

    Returns a list of (method, status) entries for CLI/logging.
    """
    results: list[tuple[str, str]] = []
    for method in methods:
        tokenizer_path, model_path = _resolve_embedding_paths(method)
        if not tokenizer_path or not model_path:
            results.append((method, "skipped: missing tokenizer/model path"))
            continue
        payload, err = _get_embedding_doc_matrix(skills, method, show_progress=show_progress)
        if err:
            results.append((method, f"failed: {err}"))
            continue
        cache_file = str(payload.get("cache_file") or "") if isinstance(payload, dict) else ""
        results.append((method, f"ok: {cache_file}".strip()))
    return results


def select_embedding_top_skills(
    goal_text: str,
    skills: list[dict],
    *,
    method: str,
    top_k: int = SEMANTIC_ROUTER_TOP_K,
) -> list[dict]:
    if not skills:
        return []

    top_k = max(1, min(int(top_k), len(skills)))
    forced, name_to_skill = _resolve_forced_skills(skills)

    docs_payload, err = _get_embedding_doc_matrix(skills, method)
    if docs_payload is None:
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] {method} unavailable ({err}); fallback to tfidf")
        return select_semantic_top_skills(goal_text, skills, top_k=top_k)

    runtime, runtime_err = _load_embedding_runtime(
        docs_payload["tokenizer_path"],
        docs_payload["model_path"],
    )
    if runtime is None:
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] {method} runtime error ({runtime_err}); fallback to tfidf")
        return select_semantic_top_skills(goal_text, skills, top_k=top_k)

    embed_max_length = _router_embed_max_length()
    query_text = (
        f"Instruct: {_router_embed_query_instruction()}\n"
        f"Query:{goal_text}"
    )
    t1 = time.perf_counter()
    query_emb, enc_err = _encode_texts_with_embedding(
        runtime,
        [query_text],
        batch_size=1,
        max_length=embed_max_length,
    )
    if query_emb is None or enc_err:
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] {method} query encode failed ({enc_err}); fallback to tfidf")
        return select_semantic_top_skills(goal_text, skills, top_k=top_k)
    if SEMANTIC_ROUTER_DEBUG:
        print(f"[semantic-router] {method} query embedding time: {time.perf_counter() - t1:.3f}s")

    try:
        t2 = time.perf_counter()
        sims = (query_emb @ docs_payload["embeddings"].T).reshape(-1)
        ranked_indices = sorted(
            range(len(sims)),
            key=lambda i: float(sims[i]),
            reverse=True,
        )
        if SEMANTIC_ROUTER_DEBUG:
            print(
                f"[semantic-router] {method} similarity over {len(sims)} skills: "
                f"{time.perf_counter() - t2:.3f}s"
            )
    except Exception as exc:
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] {method} similarity failed ({exc}); fallback to tfidf")
        return select_semantic_top_skills(goal_text, skills, top_k=top_k)

    chosen: list[dict] = []
    seen_names: set[str] = set()
    for doc_idx in ranked_indices[: max(top_k * 3, top_k)]:
        name = str(docs_payload["names"][doc_idx] or "").strip()
        if not name or name in seen_names:
            continue
        skill = name_to_skill.get(name)
        if skill is None:
            continue
        chosen.append(skill)
        seen_names.add(name)
        if len(chosen) >= top_k:
            break

    return _append_forced_skills_and_fill(chosen, skills, top_k=top_k, forced=forced)


def select_router_top_skills(
    goal_text: str,
    skills: list[dict],
    top_k: int = SEMANTIC_ROUTER_TOP_K,
) -> list[dict]:
    method = str(_router_method() or "bm25").strip().lower()
    if SEMANTIC_ROUTER_DEBUG:
        print(f"[semantic-router] method={method}")

    if method in {"tfidf", "semantic", "legacy"}:
        return select_semantic_top_skills(goal_text, skills, top_k=top_k)

    if method in {"bm25", ""}:
        return select_bm25_top_skills(goal_text, skills, top_k=top_k)

    if method in {"qwen", "qwen3", "qwen_embedding", "qwen3_embedding"}:
        return select_embedding_top_skills(goal_text, skills, method="qwen_embedding", top_k=top_k)

    if method in {
        "memento",
        "memento_qwen",
        "memento-qwen",
        "memento_qwen_embedding",
    }:
        return select_embedding_top_skills(
            goal_text,
            skills,
            method="memento_qwen_embedding",
            top_k=top_k,
        )

    if SEMANTIC_ROUTER_DEBUG:
        print(f"[semantic-router] unknown SEMANTIC_ROUTER_METHOD={method!r}; fallback to bm25")
    return select_bm25_top_skills(goal_text, skills, top_k=top_k)


# ---------------------------------------------------------------------------
# JSONL catalog helpers
# ---------------------------------------------------------------------------


def _resolve_catalog_jsonl_path(path_str: str) -> Path | None:
    raw = str(path_str or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def _parse_int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _choose_catalog_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    return max(
        entries,
        key=lambda e: (
            _parse_int_or_zero(e.get("stars")),
            _parse_int_or_zero(e.get("updatedAt")),
            len(str(e.get("description") or "")),
            -_parse_int_or_zero(e.get("_line")),
        ),
    )


def _load_router_catalog_from_jsonl(
    path_str: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    path = _resolve_catalog_jsonl_path(path_str)
    if path is None:
        return [], {}
    try:
        st = path.stat()
    except Exception:
        return [], {}

    cache_key = str(path)
    with _JSONL_CATALOG_LOCK:
        cached = _JSONL_CATALOG_CACHE.get(cache_key)
        if cached:
            if cached.get("mtime_ns") == st.st_mtime_ns and cached.get("size") == st.st_size:
                skills = cached.get("skills")
                by_name = cached.get("by_name")
                if isinstance(skills, list) and isinstance(by_name, dict):
                    return skills, by_name

    name_order: list[str] = []
    by_name: dict[str, list[dict[str, Any]]] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get("name") or "").strip()
                if not name:
                    continue
                if name not in by_name:
                    by_name[name] = []
                    name_order.append(name)
                by_name[name].append(
                    {
                        "name": name,
                        "description": str(obj.get("description") or "").strip(),
                        "githubUrl": str(
                            obj.get("githubUrl") or obj.get("github_url") or ""
                        ).strip(),
                        "skillUrl": str(obj.get("skillUrl") or "").strip(),
                        "id": str(obj.get("id") or "").strip(),
                        "author": str(obj.get("author") or "").strip(),
                        "stars": _parse_int_or_zero(obj.get("stars")),
                        "updatedAt": _parse_int_or_zero(obj.get("updatedAt")),
                        "_line": line_no,
                    }
                )
    except Exception as exc:
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] failed to parse catalog jsonl {path!r}: {exc}")
        return [], {}

    skills: list[dict[str, Any]] = []
    for name in name_order:
        preferred = _choose_catalog_entry(by_name.get(name) or [])
        if preferred is None:
            continue
        skill: dict[str, Any] = {
            "name": name,
            "description": str(preferred.get("description") or "").strip(),
        }
        github_url = str(preferred.get("githubUrl") or "").strip()
        if github_url:
            skill["githubUrl"] = github_url
        skills.append(skill)

    with _JSONL_CATALOG_LOCK:
        _JSONL_CATALOG_CACHE[cache_key] = {
            "mtime_ns": st.st_mtime_ns,
            "size": st.st_size,
            "skills": skills,
            "by_name": by_name,
        }
    return skills, by_name


def _merge_skill_catalog(primary: list[dict], fallback: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen_names: set[str] = set()
    for source in (primary, fallback):
        for raw in source:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if not name or name in seen_names:
                continue
            item: dict[str, Any] = {
                "name": name,
                "description": str(raw.get("description") or "").strip(),
            }
            github_url = str(raw.get("githubUrl") or "").strip()
            if github_url:
                item["githubUrl"] = github_url
            merged.append(item)
            seen_names.add(name)
    return merged


# ---------------------------------------------------------------------------
# Router step helpers
# ---------------------------------------------------------------------------


def build_router_step_note(
    *,
    step_num: int,
    step_skill: str,
    step_instruction: str,
    step_output: str,
    original_goal: str,
) -> str:
    def _generate_next_todo() -> str | None:
        if not ROUTER_DYNAMIC_GAP_ENABLED:
            return None

        prompt = f"""You are deriving the next actionable subtask for a workflow router.
Return ONLY JSON with one key:
{{"next_todo":"<one short actionable sentence>"}}

Rules:
- Focus only on the next concrete action.
- Keep it to one sentence and under 180 characters if possible.
- Do NOT include "Original objective:".
- If the task appears complete, return {{"next_todo":"Task complete"}}.

Original objective:
{_truncate_middle(str(original_goal or ""), 380)}

Last step skill:
{step_skill}

Last step instruction:
{_truncate_middle(str(step_instruction or ""), 260).replace(chr(10), " ")}

Last step output:
{_truncate_middle(str(step_output or ""), ROUTER_DYNAMIC_GAP_MAX_CHARS)}
""".strip()
        try:
            raw = openrouter_messages(
                "Return only valid JSON.",
                [{"role": "user", "content": prompt}],
            )
            parsed = parse_json_output(raw)
            obj: dict[str, Any] | None = None
            if isinstance(parsed, dict):
                obj = parsed
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        obj = item
                        break

            todo = obj.get("next_todo") if isinstance(obj, dict) else None
            if not isinstance(todo, str):
                return None

            one_line = " ".join(todo.split())
            one_line = re.sub(
                r"(?i)^original objective\\s*:\\s*", "", one_line
            ).strip(" -")
            if not one_line:
                return None
            return _truncate_middle(one_line, 220).replace("\n", " ")
        except Exception as exc:
            if SEMANTIC_ROUTER_DEBUG:
                print(f"[semantic-router] next_todo generation failed: {exc}")
            return None

    output = str(step_output or "").strip()
    first_line = next(
        (line.strip() for line in output.splitlines() if line.strip()),
        "(empty output)",
    )
    first_line = _truncate_middle(first_line, 220).replace("\n", " ")
    failed = output.startswith("ERR:") or "Traceback" in output
    status = (
        "failed"
        if failed
        else (
            "partial"
            if any(k in output for k in ("SKIP", "NOOP", "unknown op"))
            else "success"
        )
    )
    if failed:
        gap = f"Resolve failure from {step_skill} and continue: {original_goal}"
    else:
        gap = f"Continue remaining work for: {original_goal}"
    dynamic_gap = _generate_next_todo()
    if dynamic_gap:
        gap = dynamic_gap

    return (
        f"[Step {step_num}]\n"
        f"Skill: {step_skill}\n"
        f"Status: {status}\n"
        f"Done: {first_line}\n"
        f"Gap: {gap}\n"
        f"Instruction: {_truncate_middle(str(step_instruction or ''), 180).replace(chr(10), ' ')}"
    )


def derive_semantic_goal(original_goal: str, router_context: list[str]) -> str:
    if not router_context:
        return original_goal
    last = str(router_context[-1])
    m = re.search(r"^Gap:\s*(.+)$", last, re.MULTILINE)
    gap = m.group(1).strip() if m else ""
    if not gap:
        return original_goal
    return gap
