"""Embedding router implementation extracted from skill_catalog.py."""

from __future__ import annotations

import re
import threading
import time
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable

from core.config import (
    SEMANTIC_ROUTER_BASE_SKILLS,
    SEMANTIC_ROUTER_DEBUG,
    SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH,
    SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH,
    SEMANTIC_ROUTER_QWEN_MODEL_PATH,
    SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH,
    SEMANTIC_ROUTER_TOP_K,
)

from . import catalog_cache as _cache


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


def _resolve_embedding_paths(method: str) -> tuple[str, str]:
    if method == "qwen_embedding":
        tokenizer_path = (
            _cache.env_str("SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH", SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH)
            or _cache.env_str("SEMANTIC_ROUTER_QWEN_MODEL_PATH", SEMANTIC_ROUTER_QWEN_MODEL_PATH)
        ).strip()
        model_path = _cache.env_str("SEMANTIC_ROUTER_QWEN_MODEL_PATH", SEMANTIC_ROUTER_QWEN_MODEL_PATH).strip()
        return tokenizer_path, model_path

    if method == "memento_qwen_embedding":
        tokenizer_path = (
            _cache.env_str(
                "SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH",
                SEMANTIC_ROUTER_MEMENTO_QWEN_TOKENIZER_PATH,
            )
            or _cache.env_str("SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH", SEMANTIC_ROUTER_QWEN_TOKENIZER_PATH)
            or _cache.env_str("SEMANTIC_ROUTER_QWEN_MODEL_PATH", SEMANTIC_ROUTER_QWEN_MODEL_PATH)
            or _cache.env_str(
                "SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH",
                SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH,
            )
        ).strip()
        model_path = _cache.env_str(
            "SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH",
            SEMANTIC_ROUTER_MEMENTO_QWEN_MODEL_PATH,
        ).strip()
        return tokenizer_path, model_path

    return "", ""


def _resolve_embedding_cache_file(method: str, model_path: str) -> Path:
    cache_dir = _cache.router_embed_cache_dir().resolve()
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
    cached_runtime = _cache.get_embedding_runtime_cache(cache_key)
    if isinstance(cached_runtime, dict):
        if SEMANTIC_ROUTER_DEBUG:
            print("[semantic-router] embedding runtime cache hit")
        return cached_runtime, None

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
    _cache.put_embedding_runtime_cache(cache_key, runtime)
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

    catalog_sig = _cache.catalog_signature(skills)
    embed_max_length = _cache.router_embed_max_length()
    embed_batch_size = _cache.router_embed_batch_size()
    cache_key = f"{method}|{catalog_sig}|{tokenizer_path}|{model_path}|{embed_max_length}"
    cached_doc = _cache.get_embedding_doc_cache(cache_key)
    if isinstance(cached_doc, dict):
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] {method} doc-embedding memory cache hit")
        return cached_doc, None

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
    _cache.put_embedding_doc_cache(cache_key, payload)
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
    if not _cache.router_embed_prewarm_enabled() or not skills:
        return
    if methods is None:
        methods = _router_method_to_embedding_methods(_cache.router_method())
    methods = tuple(str(x).strip() for x in methods if str(x).strip())
    if not methods:
        return

    sig = _cache.catalog_signature(skills)
    prewarm_key = f"{sig}|{','.join(sorted(methods))}"
    if not _cache.begin_embedding_prewarm(prewarm_key):
        return

    def _worker() -> None:
        try:
            _prewarm_embedding_catalog_sync(skills, methods=methods)
        finally:
            _cache.finish_embedding_prewarm(prewarm_key)

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


def _fallback_select_semantic_top_skills(
    goal_text: str,
    skills: list[dict],
    top_k: int,
) -> list[dict]:
    from .catalog_router import select_semantic_top_skills

    return select_semantic_top_skills(goal_text, skills, top_k=top_k)


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
        return _fallback_select_semantic_top_skills(goal_text, skills, top_k=top_k)

    runtime, runtime_err = _load_embedding_runtime(
        docs_payload["tokenizer_path"],
        docs_payload["model_path"],
    )
    if runtime is None:
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] {method} runtime error ({runtime_err}); fallback to tfidf")
        return _fallback_select_semantic_top_skills(goal_text, skills, top_k=top_k)

    embed_max_length = _cache.router_embed_max_length()
    query_text = (
        f"Instruct: {_cache.router_embed_query_instruction()}\n"
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
        return _fallback_select_semantic_top_skills(goal_text, skills, top_k=top_k)
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
        return _fallback_select_semantic_top_skills(goal_text, skills, top_k=top_k)

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


__all__ = [
    "_resolve_embedding_paths",
    "_resolve_embedding_cache_file",
    "_get_model_device",
    "_last_token_pool",
    "_load_embedding_runtime",
    "_encode_texts_with_embedding",
    "_load_embedding_doc_cache",
    "_save_embedding_doc_cache",
    "_get_embedding_doc_matrix",
    "_prewarm_embedding_catalog_sync",
    "_router_method_to_embedding_methods",
    "ensure_router_embedding_prewarm",
    "precompute_router_embedding_cache",
    "select_embedding_top_skills",
]
