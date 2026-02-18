"""Catalog router/XML/jsonl functions extracted from skill_catalog.py."""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from hashlib import sha1
from math import log, sqrt
from pathlib import Path
from typing import Any

from core.config import (
    AGENTS_MD,
    PROJECT_ROOT,
    ROUTER_DYNAMIC_GAP_ENABLED,
    ROUTER_DYNAMIC_GAP_MAX_CHARS,
    SEMANTIC_ROUTER_BASE_SKILLS,
    SEMANTIC_ROUTER_CATALOG_JSONL,
    SEMANTIC_ROUTER_CATALOG_MD,
    SEMANTIC_ROUTER_DEBUG,
    SEMANTIC_ROUTER_ENABLED,
    SEMANTIC_ROUTER_TOP_K,
    SEMANTIC_ROUTER_WRITE_VISIBLE_AGENTS,
)
from core.llm import get_llm_call_budget, openrouter_messages
from core.utils.json_utils import parse_json_output
from core.utils.logging_utils import log_event
from core.utils.path_utils import _truncate_middle, _xml_escape

from .. import catalog_jsonl as _catalog_jsonl
from . import catalog_cache as _cache
from .catalog_embedding import (
    ensure_router_embedding_prewarm,
    precompute_router_embedding_cache,
    select_embedding_top_skills,
)


def load_available_skills_block_from(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r"<available_skills>.*?</available_skills>", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"{path} missing <available_skills>. Run: npx openskills sync")
    return m.group(0)


def load_available_skills_block() -> str:
    return load_available_skills_block_from(AGENTS_MD)


def write_visible_skills_block(skills_xml: str, target_path: str = AGENTS_MD) -> None:
    xml = str(skills_xml or "").strip()
    if not xml:
        return

    signature = sha1(f"{target_path}\n{xml}".encode("utf-8")).hexdigest()
    if signature == _cache.get_last_visible_agents_sig():
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
        _cache.set_last_visible_agents_sig(signature)
    except Exception as exc:
        log_event(
            "visible_skills_write_error",
            target_path=target_path,
            error=f"{type(exc).__name__}: {exc}",
        )


def parse_available_skills(skills_xml: str) -> list[dict]:
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


def _build_semantic_index(skills: list[dict]) -> dict[str, Any]:
    docs_tf: list[Counter] = []
    df: Counter = Counter()
    postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    name_tokens: list[set[str]] = []
    names_lower: list[str] = []

    for s in skills:
        name = str(s.get("name") or "").strip()
        desc = str(s.get("description") or "").strip()
        tokens = _cache.tokenize_for_semantic(f"{name} {desc}")
        tf = Counter(tokens)
        docs_tf.append(tf)
        for tok in tf:
            df[tok] += 1
        name_tokens.append(set(_cache.tokenize_for_semantic(name)))
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
    return _cache.get_or_build_semantic_index(skills, _build_semantic_index)


def select_semantic_top_skills(
    goal_text: str,
    skills: list[dict],
    top_k: int = SEMANTIC_ROUTER_TOP_K,
) -> list[dict]:
    if not skills:
        return []

    top_k = max(1, min(int(top_k), len(skills)))
    index = _get_semantic_index(skills)
    q_tokens = _cache.tokenize_for_semantic(goal_text)
    q_tf: Counter = Counter(q_tokens)

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

    if re.search(r"[\u4e00-\u9fff]", raw):
        try:
            import jieba

            tokens = [tok.strip() for tok in jieba.cut(raw) if str(tok).strip()]
            if tokens:
                return tokens
        except Exception:
            pass

    tokens = _cache.tokenize_for_semantic(raw)
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
    return _cache.get_or_build_bm25_index(skills, _build_bm25_index)


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


def select_router_top_skills(
    goal_text: str,
    skills: list[dict],
    top_k: int = SEMANTIC_ROUTER_TOP_K,
) -> list[dict]:
    method = str(_cache.router_method() or "bm25").strip().lower()
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
    return _catalog_jsonl._parse_int_or_zero(value)


def _choose_catalog_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    return _catalog_jsonl.choose_catalog_entry(entries)


def parse_catalog_jsonl_text(
    text: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    return _catalog_jsonl.parse_catalog_jsonl_text(text)


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
    now_ts = int(time.time())
    cached_payload = _cache.get_jsonl_catalog_cache(
        cache_key,
        mtime_ns=st.st_mtime_ns,
        size=st.st_size,
        now_ts=now_ts,
    )
    if cached_payload is not None:
        return cached_payload

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        if SEMANTIC_ROUTER_DEBUG:
            print(f"[semantic-router] failed to parse catalog jsonl {path!r}: {exc}")
        return [], {}

    skills, by_name = parse_catalog_jsonl_text(text)

    _cache.put_jsonl_catalog_cache(
        cache_key,
        now_ts=now_ts,
        mtime_ns=st.st_mtime_ns,
        size=st.st_size,
        skills=skills,
        by_name=by_name,
    )
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
        budget = get_llm_call_budget()
        if isinstance(budget, int) and budget <= 1:
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


__all__ = [
    "load_available_skills_block_from",
    "load_available_skills_block",
    "write_visible_skills_block",
    "parse_available_skills",
    "build_available_skills_xml",
    "_build_semantic_index",
    "_get_semantic_index",
    "select_semantic_top_skills",
    "_resolve_forced_skills",
    "_append_forced_skills_and_fill",
    "_tokenize_for_bm25",
    "_build_bm25_index",
    "_get_bm25_index",
    "select_bm25_top_skills",
    "select_router_top_skills",
    "_resolve_catalog_jsonl_path",
    "_parse_int_or_zero",
    "_choose_catalog_entry",
    "parse_catalog_jsonl_text",
    "_load_router_catalog_from_jsonl",
    "_merge_skill_catalog",
    "build_router_step_note",
    "derive_semantic_goal",
    "ensure_router_embedding_prewarm",
    "precompute_router_embedding_cache",
    "select_embedding_top_skills",
]
