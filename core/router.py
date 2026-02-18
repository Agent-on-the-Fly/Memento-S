"""Skill routing logic: explicit matching, semantic selection, LLM-based routing."""

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from core.config import (
    AGENTS_MD,
    DEBUG,
    SEMANTIC_ROUTER_ENABLED,
    SEMANTIC_ROUTER_TOP_K,
    SEMANTIC_ROUTER_DEBUG,
    SEMANTIC_ROUTER_WRITE_VISIBLE_AGENTS,
    SEMANTIC_ROUTER_CATALOG_MD,
    SEMANTIC_ROUTER_CATALOG_JSONL,
    SEMANTIC_ROUTER_BASE_SKILLS,
)
from core.utils.logging_utils import log_event
from core.llm import openrouter_messages
from core.utils.json_utils import parse_json_output
from core.utils.path_utils import _truncate_middle
from core.skill_engine.skill_catalog import (
    load_available_skills_block_from,
    parse_available_skills,
    build_available_skills_xml,
    write_visible_skills_block,
    ensure_router_embedding_prewarm,
    select_bm25_top_skills,
    select_router_top_skills,
    _load_router_catalog_from_jsonl,
    _merge_skill_catalog,
    build_router_step_note,
    derive_semantic_goal,
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _normalize_router_decision(obj: Any) -> dict:
    """Normalize router JSON output into a standard decision dict.

    Accepts various LLM response shapes (dict with action, list of steps,
    old workflow format, etc.) and returns a dict with at least an ``action``
    key set to one of ``"next_step"``, ``"done"``, or ``"none"``.
    """
    if isinstance(obj, dict):
        action = obj.get("action")
        # Normalize action names
        if action in ("next_step", "step", "execute"):
            obj = {**obj, "action": "next_step"}
        elif action in ("done", "complete", "finished"):
            obj = {**obj, "action": "done"}
        elif action in ("load_skill", "skill"):
            obj = {**obj, "action": "next_step"}  # Treat as next_step
        elif isinstance(action, str) and action.strip():
            # Some models emit the skill name directly as "action"
            # (e.g. {"action":"web-search","name":"web-search",...}).
            # Normalize this shape to next_step so executors will run it.
            skill_name = str(obj.get("name") or "").strip()
            action_name = action.strip()
            if skill_name:
                obj = {**obj, "action": "next_step"}
            elif re.fullmatch(r"[a-z0-9][a-z0-9-]*", action_name):
                obj = {
                    **obj,
                    "action": "next_step",
                    "name": action_name,
                    "reason": obj.get("reason") or "router_action_as_skill_name",
                }
        elif not action:
            # Infer action from structure
            if isinstance(obj.get("steps"), list):
                # Old workflow format - convert first step to next_step
                steps = obj.get("steps")
                if steps:
                    first = steps[0]
                    return {
                        "action": "next_step",
                        "name": first.get("name") or first.get("skill"),
                        "user": first.get("user") or first.get("user_text"),
                        "reason": obj.get("reason") or "router_normalized",
                    }
                return {"action": "none", "reason": "empty_steps"}
            elif isinstance(obj.get("name"), str) and obj.get("name").strip():
                obj = {
                    **obj,
                    "action": "next_step",
                    "reason": obj.get("reason") or "router_normalized",
                }
            else:
                obj = {
                    **obj,
                    "action": "none",
                    "reason": obj.get("reason") or "router_normalized",
                }
        return obj

    if isinstance(obj, list):
        if not obj:
            return {"action": "none", "reason": "router_empty_list"}
        for item in obj:
            if isinstance(item, dict) and item.get("action"):
                return _normalize_router_decision(item)
        # Old workflow list format - convert first step
        if all(isinstance(item, dict) for item in obj):
            first = obj[0]
            return {
                "action": "next_step",
                "name": first.get("name") or first.get("skill"),
                "user": first.get("user") or first.get("user_text"),
                "reason": "router_list_first_step",
            }
        return {"action": "none", "reason": "router_invalid_list"}

    return {"action": "none", "reason": f"router_invalid_type:{type(obj).__name__}"}


def route_skill(
    user_text: str,
    skills: list[dict],
    skills_xml: str,
    *,
    allow_new_skills: bool = True,
    context: list[str] | None = None,
    routing_goal: str | None = None,
    debug: bool = False,
) -> dict:
    """Route user request to skill(s). Can be called iteratively for dynamic workflows.

    Args:
        user_text: Original user request
        skills: List of available skills
        skills_xml: XML representation of skills
        allow_new_skills: Whether to allow creating new skills
        context: List of previous step outputs (for dynamic workflow continuation)
        routing_goal: Goal text used for semantic candidate retrieval before LLM routing
    """
    goal_text = str(routing_goal or user_text or "").strip() or user_text
    debug_enabled = bool(debug) or DEBUG
    semantic_router_enabled = _env_flag("SEMANTIC_ROUTER_ENABLED", SEMANTIC_ROUTER_ENABLED)
    semantic_router_top_k = max(1, _env_int("SEMANTIC_ROUTER_TOP_K", SEMANTIC_ROUTER_TOP_K))
    write_visible_agents = _env_flag(
        "SEMANTIC_ROUTER_WRITE_VISIBLE_AGENTS",
        SEMANTIC_ROUTER_WRITE_VISIBLE_AGENTS,
    )
    t_route = time.perf_counter()

    def _debug_timing(label: str, started_at: float) -> None:
        if not debug_enabled:
            return
        elapsed = max(0.0, time.perf_counter() - float(started_at))
        print(f"[debug][timing] {label}: {elapsed:.3f}s")

    log_event(
        "route_skill_input",
        user_text=user_text,
        routing_goal=routing_goal,
        computed_goal=goal_text,
        context=context,
        skills_count=len(skills),
        allow_new_skills=allow_new_skills,
    )

    visible_skills = skills
    visible_skills_xml = skills_xml
    semantic_catalog_skills = skills
    skill_extra_catalog = [
        s
        for s in skills
        if isinstance(s, dict) and str(s.get("_source") or "").strip().lower() == "skill_extra"
    ]
    skill_extra_topk: list[dict[str, Any]] = []
    catalog_loaded_ok = False
    catalog_source = "runtime"
    t_catalog = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load extended catalog from JSONL (preferred) or MD fallback
    # ------------------------------------------------------------------
    if SEMANTIC_ROUTER_CATALOG_JSONL:
        try:
            catalog_skills, _by_name = _load_router_catalog_from_jsonl(SEMANTIC_ROUTER_CATALOG_JSONL)
            if catalog_skills:
                semantic_catalog_skills = catalog_skills
                catalog_loaded_ok = True
                catalog_source = "jsonl"
            elif SEMANTIC_ROUTER_DEBUG:
                print(
                    f"[semantic-router] empty JSONL catalog at {SEMANTIC_ROUTER_CATALOG_JSONL!r}; "
                    "fallback to AGENTS catalog"
                )
        except Exception as exc:
            if SEMANTIC_ROUTER_DEBUG:
                print(f"[semantic-router] failed to load JSONL catalog {SEMANTIC_ROUTER_CATALOG_JSONL!r}: {exc}")

    if not catalog_loaded_ok and SEMANTIC_ROUTER_CATALOG_MD:
        try:
            catalog_xml = load_available_skills_block_from(SEMANTIC_ROUTER_CATALOG_MD)
            catalog_skills = parse_available_skills(catalog_xml)
            if catalog_skills:
                semantic_catalog_skills = catalog_skills
                catalog_loaded_ok = True
                catalog_source = "xml"
            elif SEMANTIC_ROUTER_DEBUG:
                print(f"[semantic-router] empty catalog at {SEMANTIC_ROUTER_CATALOG_MD!r}; fallback to AGENTS_MD")
        except Exception as exc:
            if SEMANTIC_ROUTER_DEBUG:
                print(f"[semantic-router] failed to load catalog {SEMANTIC_ROUTER_CATALOG_MD!r}: {exc}")
    _debug_timing("route.catalog_load", t_catalog)

    # ------------------------------------------------------------------
    # 1.5. Always pick top-K from skill_extra via BM25
    # ------------------------------------------------------------------
    if skill_extra_catalog:
        t_extra = time.perf_counter()
        try:
            skill_extra_topk = select_bm25_top_skills(
                goal_text,
                skill_extra_catalog,
                top_k=max(1, min(semantic_router_top_k, len(skill_extra_catalog))),
            )
            if skill_extra_topk:
                visible_skills = _merge_skill_catalog(skill_extra_topk, visible_skills)
                visible_skills_xml = build_available_skills_xml(visible_skills)
            if SEMANTIC_ROUTER_DEBUG:
                names = ", ".join(str(s.get("name") or "").strip() for s in skill_extra_topk)
                print(f"[semantic-router] skill_extra bm25 topk={len(skill_extra_topk)}: {names}")
        except Exception as exc:
            if SEMANTIC_ROUTER_DEBUG:
                print(f"[semantic-router] skill_extra bm25 failed: {exc}")
        _debug_timing("route.skill_extra_bm25", t_extra)

    # ------------------------------------------------------------------
    # 2. Semantic top-K selection
    # ------------------------------------------------------------------
    if semantic_router_enabled and semantic_catalog_skills:
        t_semantic = time.perf_counter()
        ensure_router_embedding_prewarm(semantic_catalog_skills)
        _debug_timing("route.embedding_prewarm_trigger", t_semantic)

    if semantic_router_enabled and semantic_catalog_skills:
        t_semantic = time.perf_counter()
        selected = select_router_top_skills(
            goal_text,
            semantic_catalog_skills,
            top_k=semantic_router_top_k,
        )
        if selected:
            visible_skills = _merge_skill_catalog(selected, skills)
            if skill_extra_topk:
                visible_skills = _merge_skill_catalog(skill_extra_topk, visible_skills)
            visible_skills_xml = build_available_skills_xml(visible_skills)
            if write_visible_agents:
                if catalog_source == "xml" and SEMANTIC_ROUTER_CATALOG_MD and catalog_loaded_ok:
                    same_file = False
                    try:
                        same_file = Path(SEMANTIC_ROUTER_CATALOG_MD).resolve() == Path(AGENTS_MD).resolve()
                    except Exception:
                        same_file = SEMANTIC_ROUTER_CATALOG_MD == AGENTS_MD
                    if same_file:
                        if SEMANTIC_ROUTER_DEBUG:
                            print(
                                "[semantic-router] skip writing visible AGENTS because "
                                "SEMANTIC_ROUTER_CATALOG_MD == AGENTS_MD"
                            )
                    else:
                        write_visible_skills_block(visible_skills_xml, AGENTS_MD)
                elif SEMANTIC_ROUTER_DEBUG:
                    print(
                        "[semantic-router] skip writing visible AGENTS because "
                        "catalog source is not AGENTS-style XML"
                    )
            if SEMANTIC_ROUTER_DEBUG:
                names = ", ".join(str(s.get("name") or "").strip() for s in visible_skills)
                print(f"[semantic-router] goal={goal_text!r} visible={len(visible_skills)} skills: {names}")
            log_event(
                "semantic_router_selected",
                goal_text=goal_text,
                selected_skills=[str(s.get("name") or "").strip() for s in visible_skills],
                selected_count=len(visible_skills),
                catalog_count=len(semantic_catalog_skills),
                catalog_source=catalog_source,
            )
        _debug_timing("route.semantic_select", t_semantic)
    if debug_enabled:
        visible_names = [
            str(s.get("name") or "").strip()
            for s in visible_skills
            if isinstance(s, dict) and str(s.get("name") or "").strip()
        ]
        preview_n = 40
        preview = visible_names[:preview_n]
        more = len(visible_names) - len(preview)
        suffix = f", ... (+{more} more)" if more > 0 else ""
        print(
            f"[debug] router visible_skills({len(visible_names)}) "
            f"[catalog_source={catalog_source}]: "
            + ", ".join(preview)
            + suffix
        )

    # ------------------------------------------------------------------
    # 3. Build prompt for LLM-based routing
    # ------------------------------------------------------------------
    if allow_new_skills:
        rules = (
            "- Prefer using existing skills in <available_skills> below.\n"
            "- If none fit, you MAY invent a new skill name (lowercase kebab-case) and plan to use it.\n"
            "- New skill names should be short (2-4 words) and capability-focused (not task-id-specific).\n"
        )
        available_label = "Available skills (existing):"
    else:
        rules = "- Use ONLY skills listed in <available_skills> below.\n"
        available_label = "Available skills (authoritative):"

    # Build context section if we have previous steps
    context_section = ""
    if context:
        context_section = (
            "\n\n=== COMPLETED STEPS AND THEIR OUTPUTS ===\n"
            + "\n\n".join(context)
            + "\n=== END OF COMPLETED STEPS ===\n\n"
            "IMPORTANT: Analyze the outputs above carefully.\n"
            '- If the original task is FULLY COMPLETED based on these outputs, return {"action":"done"}\n'
            "- If more work is needed, return the NEXT step (do NOT repeat a completed step)\n"
            "- Each step should make NEW progress, not repeat previous work"
        )

    prompt = f"""
You are a skill router and workflow planner.

Return ONLY JSON in ONE of these forms:

1) Execute next skill step (only if more work needed):
{{"action":"next_step","name":"<skill-name>","user":"instruction for this step only","reason":"short"}}

2) No skill needed (answer directly in normal chat):
{{"action":"done","reason":"no_skill_needed"}}

3) Task complete (when the user's request has been fulfilled):
{{"action":"done","reason":"explain what was accomplished"}}

4) No skill matches but a skill/tool would be required to complete the task:
{{"action":"none","reason":"short"}}

CRITICAL Rules:
{rules}- CAREFULLY analyze completed steps before deciding.
- IMPORTANT: If the user request involves ACTUAL ACTIONS like saving/writing files, running commands, searching the web, installing packages, etc., you MUST use the appropriate skill (e.g., filesystem for file operations, terminal for shell commands). Do NOT return "done" for action requests - that would skip the actual execution!
- If you are uncertain about correctness/completeness/freshness, first check available skills and route to a helpful one instead of answering directly (especially use web-search for factual lookups).
- For non-trivial questions or instructions, prefer a skill step when any available skill can materially improve answer quality.
- Only return {{"action":"done","reason":"no_skill_needed"}} for very simple, high-confidence conversational requests (e.g., greetings, small talk, basic explanations) that clearly do not need tool support.
- Return "done" if the user's original request has been satisfied (after skills have executed).
- Return "none" ONLY when the user clearly needs an external action/tool AND none of the available skills apply.
- Do NOT repeat steps that have already been completed successfully.
- Each new step must make DIFFERENT progress than previous steps.

{available_label}
{visible_skills_xml}

User request:
{user_text}
{context_section}
""".strip()
    log_event(
        "route_prompt",
        goal_text=goal_text,
        available_label=available_label,
        visible_skills=[str(s.get("name") or "").strip() for s in visible_skills],
        context_count=(len(context) if isinstance(context, list) else 0),
    )

    # ------------------------------------------------------------------
    # 4. LLM call for routing decision
    # ------------------------------------------------------------------
    t_llm = time.perf_counter()
    output = openrouter_messages(
        "Return only valid JSON.",
        [{"role": "user", "content": prompt}],
    )
    _debug_timing("route.llm_decision_call", t_llm)
    log_event("route_raw_output", raw_output=output)

    # ------------------------------------------------------------------
    # 5. Parse and normalize the response
    # ------------------------------------------------------------------
    try:
        t_parse = time.perf_counter()
        decision = _normalize_router_decision(json.loads(output))
        _debug_timing("route.parse_json", t_parse)
        log_event("route_skill_output", decision=decision, mode="json")
        _debug_timing("route.total", t_route)
        return decision
    except json.JSONDecodeError:
        t_parse = time.perf_counter()
        parsed = parse_json_output(output)
        _debug_timing("route.parse_fallback", t_parse)
        if parsed:
            decision = _normalize_router_decision(parsed)
            log_event("route_skill_output", decision=decision, mode="parsed_json_fragment")
            _debug_timing("route.total", t_route)
            return decision
        preview = (output or "").strip().replace("\n", "\\n")
        if DEBUG:
            print(f"[debug] route_skill invalid JSON output preview={preview[:500]!r}")
        decision = {"action": "none", "reason": "router_invalid_json"}
        log_event("route_skill_output", decision=decision, mode="invalid_json")
        _debug_timing("route.total(invalid_json)", t_route)
        return decision
