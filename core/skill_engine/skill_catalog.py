from __future__ import annotations

import os
import re
import threading
import time
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable

from core.config import (
    PROJECT_ROOT,
    SEMANTIC_ROUTER_CATALOG_JSONL,
    SEMANTIC_ROUTER_DEBUG,
    SEMANTIC_ROUTER_EMBED_BATCH_SIZE,
    SEMANTIC_ROUTER_EMBED_CACHE_DIR,
    SEMANTIC_ROUTER_EMBED_MAX_LENGTH,
    SEMANTIC_ROUTER_EMBED_PREWARM,
    SEMANTIC_ROUTER_EMBED_QUERY_INSTRUCTION,
    SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH,
    SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH,
    SEMANTIC_ROUTER_METHOD,
    SEMANTIC_ROUTER_QWEN_MODEL_PATH,
    SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH,
    SEMANTIC_ROUTER_TOP_K,
)

# ---------------------------------------------------------------------------
# Stopwords & tokenization
# ---------------------------------------------------------------------------

_ROUTER_STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "to",
    "with", "this", "these", "those", "need", "needs", "using", "use",
    "help", "please",
}


def _tokenize_for_semantic(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [tok for tok in raw if tok and tok not in _ROUTER_STOPWORDS and len(tok) > 1]


def _tokenize_for_bm25(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
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




# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_BM25_INDEX_SIG: str | None = None
_BM25_INDEX: dict[str, Any] | None = None
_BM25_INDEX_SOURCE_ID: int | None = None
_EMBEDDING_DOC_CACHE: dict[str, dict[str, Any]] = {}
_EMBEDDING_RUNTIME_CACHE: dict[str, dict[str, Any]] = {}
_EMBEDDING_LOCK = threading.Lock()
_EMBEDDING_PREWARM_LOCK = threading.Lock()
_EMBEDDING_PREWARM_IN_PROGRESS: set[str] = set()
_EMBEDDING_PREWARM_COMPLETED: set[str] = set()


# ---------------------------------------------------------------------------
# Env helpers (live-reload friendly)
# ---------------------------------------------------------------------------

def _router_method() -> str:
    return (os.getenv("SEMANTIC_ROUTER_METHOD") or SEMANTIC_ROUTER_METHOD or "bm25").strip().lower()


def _router_embed_max_length() -> int:
    try:
        return max(256, int(os.getenv("SEMANTIC_ROUTER_EMBED_MAX_LENGTH") or SEMANTIC_ROUTER_EMBED_MAX_LENGTH))
    except Exception:
        return SEMANTIC_ROUTER_EMBED_MAX_LENGTH


def _router_embed_batch_size() -> int:
    try:
        return max(1, int(os.getenv("SEMANTIC_ROUTER_EMBED_BATCH_SIZE") or SEMANTIC_ROUTER_EMBED_BATCH_SIZE))
    except Exception:
        return SEMANTIC_ROUTER_EMBED_BATCH_SIZE


def _router_embed_prewarm_enabled() -> bool:
    raw = os.getenv("SEMANTIC_ROUTER_EMBED_PREWARM")
    if raw is None:
        return SEMANTIC_ROUTER_EMBED_PREWARM
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _router_embed_query_instruction() -> str:
    return (os.getenv("SEMANTIC_ROUTER_EMBED_QUERY_INSTRUCTION") or SEMANTIC_ROUTER_EMBED_QUERY_INSTRUCTION).strip()


def _router_embed_cache_dir() -> Path:
    raw = os.getenv("SEMANTIC_ROUTER_EMBED_CACHE_DIR") or str(SEMANTIC_ROUTER_EMBED_CACHE_DIR)
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


# ---------------------------------------------------------------------------
# BM25 routing
# ---------------------------------------------------------------------------

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


def _route_bm25(
    goal_text: str,
    skills: list[dict],
    top_k: int,
) -> list[dict]:
    if not skills:
        return []
    top_k = max(1, min(int(top_k), len(skills)))

    bm25_index = _get_bm25_index(skills)
    if not bm25_index:
        if SEMANTIC_ROUTER_DEBUG:
            print("[semantic-router] bm25 dependencies missing, no matches")
        return []

    q_tokens = _tokenize_for_bm25(goal_text)
    if not q_tokens:
        return []

    bm25 = bm25_index["bm25"]
    name_tokens = bm25_index["name_tokens"]
    names_lower = bm25_index["names_lower"]
    try:
        raw_scores = bm25.get_scores(q_tokens)
        scores = [float(v) for v in raw_scores]
    except Exception:
        if SEMANTIC_ROUTER_DEBUG:
            print("[semantic-router] bm25 scoring failed, no matches")
        return []

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

    return chosen


# ---------------------------------------------------------------------------
# Qwen embedding routing
# ---------------------------------------------------------------------------

def _resolve_embedding_paths(method: str) -> tuple[str, str]:
    if method == "qwen_embedding":
        tokenizer_path = (
            (os.getenv("SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH") or SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH)
            or (os.getenv("SEMANTIC_ROUTER_QWEN_MODEL_PATH") or SEMANTIC_ROUTER_QWEN_MODEL_PATH)
        ).strip()
        model_path = (os.getenv("SEMANTIC_ROUTER_QWEN_MODEL_PATH") or SEMANTIC_ROUTER_QWEN_MODEL_PATH).strip()
        return tokenizer_path, model_path

    if method == "memento_qwen_embedding":
        tokenizer_path = (
            (os.getenv("SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH") or SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH)
            or (os.getenv("SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH") or SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH)
            or (os.getenv("SEMANTIC_ROUTER_QWEN_MODEL_PATH") or SEMANTIC_ROUTER_QWEN_MODEL_PATH)
            or (os.getenv("SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH") or SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH)
        ).strip()
        model_path = (
            os.getenv("SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH") or SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH
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


def _load_embedding_runtime(
    tokenizer_path: str, model_path: str,
) -> tuple[dict[str, Any] | None, str | None]:
    cache_key = f"{tokenizer_path}::{model_path}"
    with _EMBEDDING_LOCK:
        cached = _EMBEDDING_RUNTIME_CACHE.get(cache_key)
        if isinstance(cached, dict):
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
            f"[semantic-router] embedding runtime loaded in "
            f"{time.perf_counter() - t0:.2f}s (device={device})"
        )
    return runtime, None


def _encode_texts(
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
        return None, f"missing embedding model/tokenizer path for method={method!r}"

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
        embeddings, enc_err = _encode_texts(
            runtime,
            doc_texts,
            batch_size=embed_batch_size,
            max_length=embed_max_length,
        )
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


def _route_qwen(
    goal_text: str,
    skills: list[dict],
    *,
    method: str,
    top_k: int,
) -> list[dict]:
    if not skills:
        return []
    top_k = max(1, min(int(top_k), len(skills)))
    name_to_skill = {
        str(s.get("name") or "").strip(): s
        for s in skills
        if isinstance(s, dict) and str(s.get("name") or "").strip()
    }

    docs_payload, err = _get_embedding_doc_matrix(skills, method)
    if docs_payload is None:
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] {method} unavailable ({err}); fallback to bm25")
        return _route_bm25(goal_text, skills, top_k)

    runtime, runtime_err = _load_embedding_runtime(
        docs_payload["tokenizer_path"],
        docs_payload["model_path"],
    )
    if runtime is None:
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] {method} runtime error ({runtime_err}); fallback to bm25")
        return _route_bm25(goal_text, skills, top_k)

    embed_max_length = _router_embed_max_length()
    query_text = (
        f"Instruct: {_router_embed_query_instruction()}\n"
        f"Query:{goal_text}"
    )
    t1 = time.perf_counter()
    query_emb, enc_err = _encode_texts(
        runtime,
        [query_text],
        batch_size=1,
        max_length=embed_max_length,
    )
    if query_emb is None or enc_err:
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] {method} query encode failed ({enc_err}); fallback to bm25")
        return _route_bm25(goal_text, skills, top_k)
    if SEMANTIC_ROUTER_DEBUG:
        print(f"[semantic-router] {method} query embedding time: {time.perf_counter() - t1:.3f}s")

    try:
        sims = (query_emb @ docs_payload["embeddings"].T).reshape(-1)
        ranked_indices = sorted(
            range(len(sims)),
            key=lambda i: float(sims[i]),
            reverse=True,
        )
    except Exception as exc:
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] {method} similarity failed ({exc}); fallback to bm25")
        return _route_bm25(goal_text, skills, top_k)

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

    return chosen


def _router_method_to_embedding_methods(method: str) -> tuple[str, ...]:
    m = str(method or "").strip().lower()
    if m in {"qwen", "qwen3", "qwen_embedding", "qwen3_embedding"}:
        return ("qwen_embedding",)
    if m in {"memento", "memento_qwen", "memento-qwen", "memento_qwen_embedding"}:
        return ("memento_qwen_embedding",)
    return ()


# ---------------------------------------------------------------------------
# SkillCatalog class
# ---------------------------------------------------------------------------

class SkillCatalog:
    """Unified skill catalog with semantic routing."""

    def __init__(self) -> None:
        self._catalog: list[dict] | None = None
        self._local_skills: list[dict] | None = None
        self._cloud_skills: list[dict] | None = None
        self._lock = threading.Lock()

    # -- public API --

    def route(self, query: str, top_k: int | None = None) -> list[dict]:
        """Route a user query to matching skills.

        Local skills are always included. The router only selects from
        cloud skills, appending the top_k most relevant ones after the
        full local list.
        """
        self._ensure_catalog()
        local = self._local_skills or []
        cloud = self._cloud_skills or []

        if top_k is None:
            top_k = SEMANTIC_ROUTER_TOP_K

        # Always include all local skills
        result = list(local)
        local_names = {str(s.get("name") or "").strip() for s in local}

        # Route cloud skills if available
        if cloud:
            method = _router_method()
            if SEMANTIC_ROUTER_DEBUG:
                print(f"[semantic-router] method={method} local={len(local)} cloud={len(cloud)}")

            if method in {"bm25", ""}:
                matched = _route_bm25(query, cloud, top_k)
            elif method in {"qwen", "qwen3", "qwen_embedding", "qwen3_embedding"}:
                matched = _route_qwen(query, cloud, method="qwen_embedding", top_k=top_k)
            elif method in {"memento", "memento_qwen", "memento-qwen", "memento_qwen_embedding"}:
                matched = _route_qwen(query, cloud, method="memento_qwen_embedding", top_k=top_k)
            else:
                if SEMANTIC_ROUTER_DEBUG:
                    print(f"[semantic-router] unknown method={method!r}; fallback to bm25")
                matched = _route_bm25(query, cloud, top_k)

            # Append cloud matches that aren't already in local
            for s in matched:
                name = str(s.get("name") or "").strip()
                if name and name not in local_names:
                    result.append(s)
                    local_names.add(name)

        return result

    @staticmethod
    def format_skills_context(skills: list[dict]) -> str:
        """Format matched skills as injection text for the user message."""
        if not skills:
            return ""
        lines = [
            "[Matched Skills]",
            "Skills relevant to your query have been pre-selected. "
            "Use `read_skill` to read full documentation before using a skill.",
            "",
        ]
        for s in skills:
            name = str(s.get("name") or "").strip()
            desc = str(s.get("description") or "").strip()
            if desc and len(desc) > 200:
                desc = desc[:197] + "..."
            entry = f"- {name}: {desc}" if desc else f"- {name}"
            lines.append(entry)
        lines.append("[/Matched Skills]")
        return "\n".join(lines)

    def ensure_prewarm(self) -> None:
        """Prewarm embedding caches in a background thread if configured."""
        if not _router_embed_prewarm_enabled():
            return
        skills = self._ensure_catalog()
        if not skills:
            return

        methods = _router_method_to_embedding_methods(_router_method())
        methods = tuple(str(x).strip() for x in methods if str(x).strip())
        if not methods:
            return

        sig = _catalog_signature(skills)
        prewarm_key = f"{sig}|{','.join(sorted(methods))}"
        with _EMBEDDING_PREWARM_LOCK:
            if prewarm_key in _EMBEDDING_PREWARM_COMPLETED or prewarm_key in _EMBEDDING_PREWARM_IN_PROGRESS:
                return
            _EMBEDDING_PREWARM_IN_PROGRESS.add(prewarm_key)

        def _worker() -> None:
            try:
                for method in methods:
                    tp, mp = _resolve_embedding_paths(method)
                    if not tp or not mp:
                        continue
                    _get_embedding_doc_matrix(skills, method)
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

    # -- private catalog build --

    def _ensure_catalog(self) -> list[dict]:
        with self._lock:
            if self._catalog is not None:
                return self._catalog
            self._local_skills = self._parse_local_skills()
            self._cloud_skills = self._load_cloud_skills()
            self._catalog = _merge_catalogs(self._local_skills, self._cloud_skills)
            return self._catalog

    def _parse_local_skills(self) -> list[dict]:
        from core.skill_engine.skill_resolver import _iter_skill_roots

        skills: list[dict] = []
        seen: set[str] = set()
        for root in _iter_skill_roots():
            if not root.exists() or not root.is_dir():
                continue
            try:
                for skill_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                    if not skill_dir.is_dir():
                        continue
                    skill_md = skill_dir / "SKILL.md"
                    if not skill_md.exists():
                        continue
                    name = skill_dir.name
                    if name in seen:
                        continue
                    seen.add(name)
                    desc = _extract_skill_description(skill_md)
                    entry: dict[str, Any] = {"name": name}
                    if desc:
                        entry["description"] = desc
                    skills.append(entry)
            except Exception:
                continue
        return skills

    def _load_cloud_skills(self) -> list[dict]:
        from core.skill_engine.catalog_jsonl import load_catalog_from_jsonl

        jsonl_path = SEMANTIC_ROUTER_CATALOG_JSONL
        if not jsonl_path:
            return []
        skills, _ = load_catalog_from_jsonl(jsonl_path)
        return skills


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_skill_description(skill_md: Path) -> str:
    """Extract description from SKILL.md YAML frontmatter."""
    try:
        text = skill_md.read_text(encoding="utf-8")
    except Exception:
        return ""
    if not text.startswith("---"):
        return ""
    end = text.find("---", 3)
    if end == -1:
        return ""
    for line in text[3:end].splitlines():
        line = line.strip()
        if line.lower().startswith("description:"):
            desc = line[len("description:"):].strip().strip("\"'")
            return desc[:200]
    return ""


def _merge_catalogs(primary: list[dict], fallback: list[dict]) -> list[dict]:
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
# Singleton
# ---------------------------------------------------------------------------

_CATALOG_INSTANCE: SkillCatalog | None = None
_CATALOG_LOCK = threading.Lock()


def get_skill_catalog() -> SkillCatalog:
    global _CATALOG_INSTANCE
    if _CATALOG_INSTANCE is not None:
        return _CATALOG_INSTANCE
    with _CATALOG_LOCK:
        if _CATALOG_INSTANCE is None:
            _CATALOG_INSTANCE = SkillCatalog()
        return _CATALOG_INSTANCE
