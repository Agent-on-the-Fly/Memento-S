"""Failure feedback construction — building feedback blobs, target selection, tips."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import (
    _collect_attachment_refs,
    _extract_attachment_refs,
    _looks_like_image_ref,
)
from .skill_helpers import resolve_local_skill_name as _resolve_local_skill_name
from .text_utils import (
    dedupe_keep_order,
    json_fragment,
    normalize_skill_name,
    normalize_space,
    strip_markdown_fence,
)


# ---------------------------------------------------------------------------
# Feedback blob
# ---------------------------------------------------------------------------

def build_failure_feedback_blob(
    *,
    task: dict[str, Any],
    user_text: str,
    model_answer: str,
    judge: dict[str, Any] | None,
    trace: list[dict[str, Any]],
    skill_judgement: dict[str, Any] | None = None,
    include_judge_rationale: bool = True,
) -> str:
    judge_payload: dict[str, Any] = {}
    if isinstance(judge, dict):
        for k, v in judge.items():
            key = str(k)
            if (not include_judge_rationale) and key.lower() in {"rationale", "reasoning"}:
                continue
            judge_payload[key] = v
        if not include_judge_rationale:
            judge_payload["_rationale_redacted"] = True

    payload = {
        "task": {
            "id": task.get("id"),
            "question": task.get("question"),
            "answer_type": task.get("answer_type"),
            "category": task.get("category"),
            "level": task.get("level"),
            "attachments": task.get("attachments"),
        },
        "user_text": user_text,
        "previous_model_answer": model_answer,
        "judge": judge_payload,
        "trace": trace,
        "skill_judgement": (dict(skill_judgement) if isinstance(skill_judgement, dict) else None),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def append_target_selection_feedback(
    *,
    feedback: str,
    selection_context: dict[str, Any] | None,
) -> str:
    base = str(feedback or "").rstrip()
    if not isinstance(selection_context, dict) or not selection_context:
        return base
    fields = [
        "status", "reason", "selected", "skill_name", "skill_path",
        "failure_type", "optimisation_suggestion", "optimization_suggestion",
        "llm_reason", "confidence_0_to_1", "selection_scope",
    ]
    compact: dict[str, Any] = {}
    for key in fields:
        val = selection_context.get(key)
        if val is None:
            continue
        if isinstance(val, str) and not val.strip():
            continue
        compact[key] = val
    if not compact:
        compact = dict(selection_context)
    return (
        f"{base}\n\n"
        "=== TARGET SKILL SELECTION CONTEXT ===\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
    )


# ---------------------------------------------------------------------------
# Target skill selector
# ---------------------------------------------------------------------------

async def select_target_skill_for_feedback(
    *,
    used_skills: list[str],
    trace: list[dict[str, Any]],
    judge: dict[str, Any],
    base_skills_root: Path,
    extra_skills_root: Path,
) -> tuple[str | None, dict[str, Any]]:
    from core.evolve import get_evolve_llm

    candidates: list[str] = []
    seen_names: set[str] = set()
    for raw in used_skills:
        name = str(raw or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        candidates.append(name)

    if not candidates:
        return None, {"status": "no_candidates"}

    norm_to_candidate: dict[str, str] = {}
    candidate_records: list[dict[str, str]] = []
    for name in candidates:
        norm = normalize_skill_name(name)
        norm_to_candidate[norm] = name
        if (extra_skills_root / name / "SKILL.md").exists():
            skill_path = (extra_skills_root / name).resolve()
        elif (base_skills_root / name / "SKILL.md").exists():
            skill_path = (base_skills_root / name).resolve()
        else:
            skill_path = (extra_skills_root / name).resolve()
        candidate_records.append({"skill_name": name, "skill_path": str(skill_path)})

    trace_payload = [entry for entry in trace if isinstance(entry, dict)]
    judge_payload = judge if isinstance(judge, dict) else {}

    system_prompt = (
        "You are a failure attribution selector for skill optimization. "
        "Read the full trajectory and pick one skill from candidates that is most likely responsible for failure. "
        "If there is no clear owner, return an empty skill_name and explain why. "
        "Return JSON only with keys: "
        "skill_name, failure_type, skill_path, reason, optimisation_suggestion, confidence_0_to_1."
    )
    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "Candidates (must select from skill_name values here):\n"
                f"{json.dumps(candidate_records, ensure_ascii=False, indent=2)}\n\n"
                "Judge payload:\n"
                f"{json.dumps(judge_payload, ensure_ascii=False, indent=2)}\n\n"
                "Trace payload:\n"
                f"{json.dumps(trace_payload, ensure_ascii=False, indent=2)}\n\n"
                "Selection constraints:\n"
                "- Choose exactly one candidate skill_name, or empty string if no clear owner.\n"
                "- skill_path should match the chosen candidate path.\n"
                "- failure_type should be a concise label like routing_error/execution_error/tool_error/unknown.\n"
                "- reason should reference concrete trace evidence.\n"
                "- optimisation_suggestion should be actionable and generic.\n"
                "- confidence_0_to_1 should be a float between 0 and 1.\n"
                "Return JSON only."
            ),
        }
    ]

    llm = get_evolve_llm()
    raw = ""
    parsed: dict[str, Any] | None = None
    llm_error = ""
    for _ in range(2):
        try:
            resp = await llm.chat(messages, system=system_prompt)
            raw = resp.content or ""
            llm_error = ""
        except Exception as exc:
            llm_error = f"{type(exc).__name__}: {exc}"
            raw = ""
            continue
        parsed = json_fragment(raw)
        if isinstance(parsed, dict):
            break
        messages.append({"role": "assistant", "content": str(raw or "")})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Return only one JSON object with keys exactly: "
                    "skill_name, failure_type, skill_path, reason, "
                    "optimisation_suggestion, confidence_0_to_1."
                ),
            }
        )

    if not isinstance(parsed, dict):
        return None, {
            "status": "selector_failed",
            "reason": "llm_non_json_output",
            "candidates": candidates,
            "error": llm_error,
            "llm_selector_raw": str(raw or "")[:2000],
        }

    chosen_raw = str(parsed.get("skill_name") or "").strip()
    if not chosen_raw:
        path_hint = str(parsed.get("skill_path") or "").strip()
        if path_hint:
            chosen_raw = Path(path_hint).name.strip()

    selected: str | None = None
    if chosen_raw:
        selected = norm_to_candidate.get(normalize_skill_name(chosen_raw))

    reason = str(parsed.get("reason") or parsed.get("rationale") or "").strip()
    failure_type = str(parsed.get("failure_type") or "unknown").strip() or "unknown"
    optimisation_suggestion = str(
        parsed.get("optimisation_suggestion") or parsed.get("optimization_suggestion") or ""
    ).strip()

    confidence_raw = parsed.get("confidence_0_to_1")
    confidence_val: float | None = None
    if confidence_raw is not None:
        try:
            confidence_val = float(confidence_raw)
        except Exception:
            confidence_val = None
    if confidence_val is not None:
        confidence_val = max(0.0, min(1.0, confidence_val))

    if not selected:
        return None, {
            "status": "no_clear_target",
            "reason": "llm_no_clear_target",
            "candidates": candidates,
            "failure_type": failure_type,
            "optimisation_suggestion": optimisation_suggestion,
            "llm_reason": reason,
            "confidence_0_to_1": confidence_val,
            "llm_selector_output": parsed,
            "llm_selector_raw": str(raw or "")[:2000],
        }

    selected_path = ""
    for item in candidate_records:
        if str(item.get("skill_name") or "") == selected:
            selected_path = str(item.get("skill_path") or "")
            break

    return selected, {
        "status": "selected",
        "reason": "llm_trace_selector",
        "selected": selected,
        "skill_name": selected,
        "skill_path": selected_path,
        "failure_type": failure_type,
        "optimisation_suggestion": optimisation_suggestion,
        "llm_reason": reason,
        "confidence_0_to_1": confidence_val,
        "candidates": candidates,
        "llm_selector_output": parsed,
    }


# ---------------------------------------------------------------------------
# Skill judgement after answer failure
# ---------------------------------------------------------------------------

def build_skill_judgement_after_answer_failure(
    *,
    trace: list[dict[str, Any]],
    judge: dict[str, Any],
    target_skill: str | None,
    target_selector: dict[str, Any] | None,
) -> dict[str, Any]:
    selector_obj = target_selector if isinstance(target_selector, dict) else {}
    selected = str(
        target_skill or selector_obj.get("skill_name") or selector_obj.get("selected") or ""
    ).strip()
    if not selected:
        return {
            "status": "no_target_skill",
            "reason": "no_selected_skill_after_answer_failure",
        }

    selected_norm = normalize_skill_name(selected)
    hits: list[tuple[int, dict[str, Any]]] = []
    for idx, entry in enumerate(trace or []):
        if not isinstance(entry, dict):
            continue
        skill_name = str(entry.get("skill_name") or "").strip()
        if normalize_skill_name(skill_name) != selected_norm:
            continue
        hits.append((idx, entry))

    evidence_steps: list[int] = []
    for idx, entry in hits[-3:]:
        raw_step = entry.get("step_num")
        step = raw_step if isinstance(raw_step, int) else idx
        if step not in evidence_steps:
            evidence_steps.append(step)

    last_idx = int(hits[-1][0]) if hits else -1
    last_entry = hits[-1][1] if hits else {}
    last_status = (
        str(last_entry.get("status") or "").strip().lower()
        if isinstance(last_entry, dict)
        else ""
    )
    preview = (
        normalize_space(str(last_entry.get("result_preview") or ""))
        if isinstance(last_entry, dict)
        else ""
    )
    failure_type = str(selector_obj.get("failure_type") or "unknown").strip() or "unknown"
    suggestion = str(
        selector_obj.get("optimisation_suggestion") or selector_obj.get("optimization_suggestion") or ""
    ).strip()
    if not suggestion:
        suggestion = "Apply selector-provided failure evidence to update this skill deterministically."

    selector_reason = str(
        selector_obj.get("llm_reason") or selector_obj.get("reason") or selector_obj.get("status") or ""
    ).strip()
    judge_rationale = normalize_space(str((judge or {}).get("rationale") or ""))
    reason_parts: list[str] = []
    if selector_reason:
        reason_parts.append(f"selector={selector_reason}")
    if hits:
        reason_parts.append(f"trace_status={last_status or 'unknown'}")
    if preview:
        reason_parts.append(f"trace_preview={preview[:220]}")
    if judge_rationale:
        reason_parts.append(f"answer_judge={judge_rationale[:220]}")

    confidence_raw = selector_obj.get("confidence_0_to_1")
    confidence = 0.5
    if confidence_raw is not None:
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.5

    return {
        "status": ("judged" if hits else "judged_selector_only"),
        "skill_name": selected,
        "failure_type": failure_type,
        "evidence_steps": evidence_steps,
        "latest_trace_index": (last_idx if hits else None),
        "latest_trace_status": last_status,
        "result_preview": preview[:260],
        "reason": "; ".join(reason_parts),
        "optimisation_suggestion": suggestion,
        "confidence_0_to_1": float(max(0.0, min(1.0, confidence))),
        "selector_source": "llm_target_selector",
        "answer_judge_is_correct": bool((judge or {}).get("is_correct")),
        "answer_judge_score_0_to_10": (judge or {}).get("score_0_to_10"),
    }


# ---------------------------------------------------------------------------
# Generic tip rules
# ---------------------------------------------------------------------------

def _dedupe_normalize(items: list[str]) -> list[str]:
    """Deduplicate strings after normalizing whitespace."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = normalize_space(str(raw or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_generic_tip_rules(
    *,
    task: dict[str, Any],
    judge: dict[str, Any],
    trace: list[dict[str, Any]],
) -> list[str]:
    answer_type = str(task.get("answer_type") or "").strip().lower()
    image_ref = str(task.get("image") or "").strip()
    attachments = _extract_attachment_refs(task.get("attachments"))
    if not attachments:
        attachments = _collect_attachment_refs(task)
    if image_ref and image_ref not in attachments:
        attachments.insert(0, image_ref)
    has_image_attachment = any(_looks_like_image_ref(ref) for ref in attachments)
    rationale = normalize_space(str((judge or {}).get("rationale") or ""))

    rules: list[str] = [
        "A frequent failure mode is missing constraints (format, units, precision, ranges); extract constraints before solving.",
        "The evaluator is sensitive to final-answer extraction, so keep the final answer clearly separated from explanation text.",
        "Before submitting, run a quick coverage check to ensure every part of the question is addressed with no irrelevant additions.",
    ]

    if "choice" in answer_type or "multiple" in answer_type or "option" in answer_type:
        rules.append("For multiple-choice questions, keep the answer zone to a single option expression and avoid extra narrative.")
    if any(tok in answer_type for tok in ("number", "numeric", "float", "int", "integer")):
        rules.append("For numeric tasks, normalize units first and then verify decimal places or significant figures.")
    if has_image_attachment:
        rules.append("For image-based tasks, validate visual evidence first, then do textual reasoning to avoid missing key details.")
    elif attachments:
        rules.append("For attachment-based tasks, inspect the referenced files first and cite extracted evidence before concluding.")

    if any(str(entry.get("status") or "").strip().lower() == "max_steps" for entry in (trace or [])):
        rules.append("When trajectories are long, split work into short verifiable loops and validate state after each step.")

    rationale_l = rationale.lower()
    if rationale:
        if "equivalent" in rationale_l or "semantic" in rationale_l:
            rules.append("For semantic-equivalence judgments, normalize symbols/case/units before finalizing the answer.")
        if "format" in rationale_l or "parse" in rationale_l:
            rules.append("Format failures usually come from extra prefixes/suffixes; keep output structure stable.")
        if "unit" in rationale_l:
            rules.append("For physical quantities, unit consistency should be an explicit standalone check.")
        if "incomplete" in rationale_l or "missing" in rationale_l:
            rules.append("If judgment indicates missing content, first re-check sub-condition coverage before adding details.")

    return _dedupe_normalize(rules)


# ---------------------------------------------------------------------------
# Tip writing
# ---------------------------------------------------------------------------

async def write_tip_md(
    *,
    tip_path: Path,
    task: dict[str, Any],
    judge: dict[str, Any],
    trace: list[dict[str, Any]],
    no_update_reason: str,
) -> dict[str, Any]:
    from core.evolve import get_evolve_llm

    rules = _dedupe_normalize(build_generic_tip_rules(task=task, judge=judge, trace=trace))
    if not rules:
        return {"status": "tip_no_rules", "tip_file": str(tip_path)}

    reason = normalize_space(str(no_update_reason or "")) or "no_skill_update"
    tip_path.parent.mkdir(parents=True, exist_ok=True)
    existing = tip_path.read_text(encoding="utf-8") if tip_path.exists() else ""
    existing_norm = normalize_space(existing).lower()
    new_rules = [
        rule for rule in rules if normalize_space(rule).lower() not in existing_norm
    ]
    existing_doc = existing.strip()

    tip_context = {
        "task": {
            "id": task.get("id"),
            "question": task.get("question"),
            "answer_type": task.get("answer_type"),
            "category": task.get("category"),
            "level": task.get("level"),
        },
        "judge": {
            "is_correct": judge.get("is_correct"),
            "score": judge.get("score"),
            "score_0_to_10": judge.get("score_0_to_10"),
            "rationale": judge.get("rationale"),
        },
        "trace": list(trace or []),
        "reason": reason,
    }
    tip_system_prompt = (
        "You maintain a persistent TIP.md for reusable failure learnings. "
        "Rewrite the entire TIP.md as one complete Markdown document. "
        "Merge old tips with new insights, deduplicate, keep guidance generic and concise. "
        "Do not include code fences."
    )
    tip_user_prompt = (
        "Current TIP.md full text:\n"
        "<tip_md>\n"
        f"{existing_doc or '[EMPTY]'}\n"
        "</tip_md>\n\n"
        "New failure insights to integrate:\n"
        "<new_rules>\n"
        f"{json.dumps(rules, ensure_ascii=False, indent=2)}\n"
        "</new_rules>\n\n"
        "Context:\n"
        "<context>\n"
        f"{json.dumps(tip_context, ensure_ascii=False, indent=2)}\n"
        "</context>\n\n"
        "Return only the full updated TIP.md Markdown text."
    )
    out = ""
    try:
        llm = get_evolve_llm()
        resp = await llm.chat(
            [{"role": "user", "content": tip_user_prompt}],
            system=tip_system_prompt,
        )
        out = strip_markdown_fence(str(resp.content or "")).strip()
    except Exception as exc:
        return {
            "status": "tip_llm_failed",
            "tip_file": str(tip_path),
            "error": f"{type(exc).__name__}: {exc}",
            "rules_total": len(rules),
            "rules_added": len(new_rules),
            "reason": reason,
        }
    if not out:
        return {
            "status": "tip_llm_empty",
            "tip_file": str(tip_path),
            "rules_total": len(rules),
            "rules_added": len(new_rules),
            "reason": reason,
        }
    tip_path.write_text(out, encoding="utf-8")
    return {
        "status": "tip_written",
        "tip_file": str(tip_path),
        "rules_total": len(rules),
        "rules_added": len(new_rules),
        "rules_merged_total": len(rules),
        "reason": reason,
    }


async def write_tip_for_failure(
    *,
    tip_write_enabled: bool,
    tip_path: Path,
    task: dict[str, Any],
    judge: dict[str, Any],
    trace: list[dict[str, Any]],
    no_update_reason: str,
) -> dict[str, Any]:
    if tip_write_enabled:
        return await write_tip_md(
            tip_path=tip_path, task=task, judge=judge,
            trace=trace, no_update_reason=no_update_reason,
        )
    return {
        "status": "tip_write_disabled",
        "tip_file": str(tip_path),
        "reason": normalize_space(str(no_update_reason or "")) or "no_skill_update",
    }


# ---------------------------------------------------------------------------
# Tip summarization — condense TIP.md when it grows too large
# ---------------------------------------------------------------------------

_TIP_SUMMARIZE_CHAR_THRESHOLD = 3000


async def summarize_tips(
    *,
    tip_path: Path,
    max_chars: int = 2000,
) -> dict[str, Any]:
    """Condense TIP.md if it exceeds the size threshold.

    Asks the LLM to merge redundant tips, remove low-value entries,
    and produce a concise document within *max_chars*.
    """
    from core.evolve import get_evolve_llm

    if not tip_path.exists():
        return {"status": "no_tip_file"}
    try:
        existing = tip_path.read_text(encoding="utf-8").strip()
    except Exception:
        return {"status": "read_error"}
    if not existing:
        return {"status": "empty"}
    if len(existing) <= _TIP_SUMMARIZE_CHAR_THRESHOLD:
        return {"status": "below_threshold", "chars": len(existing)}

    system_prompt = (
        "You are condensing a TIP.md file that contains accumulated failure learnings. "
        "Merge redundant tips, remove low-value or overly specific entries, "
        "and keep only the most impactful, general-purpose guidance. "
        f"The output MUST be under {max_chars} characters. "
        "Return only the condensed Markdown text, no code fences."
    )
    user_prompt = (
        "Current TIP.md (too long, needs condensing):\n"
        "<tip_md>\n"
        f"{existing}\n"
        "</tip_md>\n\n"
        f"Condense to under {max_chars} characters. "
        "Keep the most useful, general tips. Remove duplicates and overly specific advice. "
        "Return only the condensed Markdown."
    )
    try:
        llm = get_evolve_llm()
        resp = await llm.chat(
            [{"role": "user", "content": user_prompt}],
            system=system_prompt,
        )
        out = strip_markdown_fence(str(resp.content or "")).strip()
    except Exception as exc:
        return {
            "status": "summarize_llm_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "chars_before": len(existing),
        }
    if not out:
        return {"status": "summarize_llm_empty", "chars_before": len(existing)}
    tip_path.write_text(out, encoding="utf-8")
    return {
        "status": "summarized",
        "chars_before": len(existing),
        "chars_after": len(out),
    }


# ---------------------------------------------------------------------------
# Collect used skills for feedback
# ---------------------------------------------------------------------------

def collect_used_skills_for_feedback(
    *,
    used_skills: list[str],
    base_skills_root: Path,
    extra_skills_root: Path,
) -> tuple[list[str], set[str], list[str]]:
    resolved_used: list[str] = []
    resolved_used_norms: set[str] = set()
    extra_used: list[str] = []
    seen_norms: set[str] = set()
    for raw in used_skills:
        resolved = (
            _resolve_local_skill_name(
                str(raw or ""),
                base_skills_root=base_skills_root,
                extra_skills_root=extra_skills_root,
            )
            or str(raw or "").strip()
        )
        norm = normalize_skill_name(resolved)
        if not norm or norm in seen_norms:
            continue
        seen_norms.add(norm)
        resolved_used.append(resolved)
        resolved_used_norms.add(norm)
        # Only treat as "extra" (optimizable) if it exists in extra but NOT in base.
        # Skills copied from builtin are fixed and should not be optimized.
        in_extra = (extra_skills_root / resolved / "SKILL.md").exists()
        in_base = (base_skills_root / resolved / "SKILL.md").exists()
        if in_extra and not in_base:
            extra_used.append(resolved)
    return resolved_used, resolved_used_norms, extra_used
