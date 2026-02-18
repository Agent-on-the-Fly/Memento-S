"""Refactored agent.py - slim entry point.

This module re-exports every public symbol from the new sub-modules so that
existing code doing ``from agent import X`` continues to work unchanged.
The only logic that lives here is ``main()`` (the CLI REPL loop) and its
helper ``_reload_skills_catalog()``.
"""
from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Re-exports from config
# ---------------------------------------------------------------------------
from core.config import (  # noqa: F401
    PROJECT_ROOT,
    MODEL,
    DEBUG,
    AGENTS_MD,
    SKILLS_DIR,
    SKILLS_EXTRA_DIRS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MAX_TOKENS,
    OPENROUTER_TIMEOUT,
    OPENROUTER_RETRIES,
    OPENROUTER_RETRY_BACKOFF,
    LLM_API,
    _LLM_API_ENV,
    OPENROUTER_PROVIDER,
    OPENROUTER_PROVIDER_ORDER,
    OPENROUTER_ALLOW_FALLBACKS,
    OPENROUTER_SITE_URL,
    OPENROUTER_APP_NAME,
    FILESYSTEM_OP_TYPES,
    TERMINAL_OP_TYPES,
    WEB_OP_TYPES,
    UV_PIP_OP_TYPES,
    BUILTIN_BRIDGE_SKILLS,
    SKILL_LOCAL_DIR_PREFIXES,
    SKILL_LOCAL_COMMAND_PATH_RE,
    CHAT_SYSTEM_PROMPT,
    MAX_WORKFLOW_STEPS,
    STEP_SUMMARY_MAX_TOKENS,
    STEP_SUMMARY_THRESHOLD,
    SKILL_LOOP_FEEDBACK_CHARS,
    SEMANTIC_ROUTER_ENABLED,
    SEMANTIC_ROUTER_TOP_K,
    SEMANTIC_ROUTER_DEBUG,
    SEMANTIC_ROUTER_WRITE_VISIBLE_AGENTS,
    SEMANTIC_ROUTER_CATALOG_MD,
    SEMANTIC_ROUTER_CATALOG_JSONL,
    SEMANTIC_ROUTER_BASE_SKILLS,
    ROUTER_DYNAMIC_GAP_ENABLED,
    ROUTER_DYNAMIC_GAP_MAX_CHARS,
    SKILL_DYNAMIC_FETCH_ENABLED,
    SKILL_DYNAMIC_FETCH_CATALOG_JSONL,
    SKILL_DYNAMIC_FETCH_ROOT,
    SKILL_DYNAMIC_FETCH_TIMEOUT_SEC,
    CLI_CREATE_ON_MISS,
    EXEC_LOG_ENABLED,
    EXEC_LOG_DIR,
    EXEC_LOG_MAX_CHARS,
    _env_flag,
    _env_int,
    _parse_env_path_list,
)

# ---------------------------------------------------------------------------
# Re-exports from json_utils
# ---------------------------------------------------------------------------
from core.utils.json_utils import (  # noqa: F401
    parse_json_output,
    extract_json_candidates,
    repair_json_string,
)

# ---------------------------------------------------------------------------
# Re-exports from logging_utils
# ---------------------------------------------------------------------------
from core.utils.logging_utils import (  # noqa: F401
    log_event,
    get_exec_log_path,
)

# ---------------------------------------------------------------------------
# Re-exports from llm
# ---------------------------------------------------------------------------
from core.llm import (  # noqa: F401
    openrouter_messages,
    _openrouter_chat_completions,
    _normalize_openrouter_base,
)

# ---------------------------------------------------------------------------
# Re-exports from path_utils
# ---------------------------------------------------------------------------
from core.utils.path_utils import (  # noqa: F401
    _truncate,
    _truncate_middle,
    _truncate_text,
    _stringify_result,
    _xml_escape,
    _resolve_dir,
    _resolve_runtime_path,
    _skill_local_rel_path,
    _rewrite_command_paths_for_skill,
    _windows_path_to_wsl,
    _shell_command,
    _find_venv,
    _venv_bin_dir,
    _is_valid_venv_dir,
    _safe_subpath,
    _run_command_capture,
    _no_git_prompt_env,
    _NO_GIT_PROMPT_ENV,
    _parse_json_object,
)

# ---------------------------------------------------------------------------
# Re-exports from skill_catalog
# ---------------------------------------------------------------------------
from core.skill_engine.skill_catalog import (  # noqa: F401
    load_available_skills_block,
    load_available_skills_block_from,
    write_visible_skills_block,
    parse_available_skills,
    build_available_skills_xml,
    _ROUTER_STOPWORDS,
    _tokenize_for_semantic,
    _catalog_signature,
    _build_semantic_index,
    _get_semantic_index,
    select_semantic_top_skills,
    _resolve_catalog_jsonl_path,
    _parse_int_or_zero,
    _choose_catalog_entry,
    _load_router_catalog_from_jsonl,
    _merge_skill_catalog,
    build_router_step_note,
    derive_semantic_goal,
)

# ---------------------------------------------------------------------------
# Re-exports from skill_resolver
# ---------------------------------------------------------------------------
from core.skill_engine.skill_resolver import (  # noqa: F401
    _iter_skill_roots,
    _resolve_skill_dir,
    has_local_skill_dir,
    _parse_github_tree_url,
    _pick_skill_dir_from_checkout,
    ensure_skill_available,
    openskills_read,
    install_or_update_skill,
)

# ---------------------------------------------------------------------------
# Re-exports from skill_executor
# ---------------------------------------------------------------------------
from core.skill_engine.skill_executor import (  # noqa: F401
    _normalize_op_dict,
    _tool_call_to_op,
    normalize_plan_shape,
    _coerce_skill_context,
    _extract_skill_context,
    _execute_skill_creator_plan,
    _filesystem_tree,
    _execute_filesystem_op,
    _execute_filesystem_ops,
    _execute_terminal_ops,
    _convert_pip_to_uv,
    _run_uv_pip,
    _execute_uv_pip_ops,
    _web_google_search,
    _fetch_async,
    _web_fetch,
    _execute_web_ops,
    _dispatch_bridge_op,
    execute_skill_plan_result,
    execute_skill_plan,
)

# ---------------------------------------------------------------------------
# Re-exports from router
# ---------------------------------------------------------------------------
from core.router import (  # noqa: F401
    explicit_skill_match,
    route_skill,
)

# ---------------------------------------------------------------------------
# Re-exports from skill_runner
# ---------------------------------------------------------------------------
from core.skill_engine.skill_runner import (  # noqa: F401
    ask_for_plan,
    validate_plan_for_skill,
    build_strict_schema_prompt,
    normalize_skill_creator_plan,
    run_one_skill,
    run_one_skill_loop,
    run_skill_once_with_plan,
    should_auto_continue_skill_result,
    _count_approx_tokens,
    summarize_step_output,
    should_create_skill_on_miss,
    _should_create_skill_on_miss_fallback,
    create_skill_on_miss,
)


# ---------------------------------------------------------------------------
# CLI REPL
# ---------------------------------------------------------------------------

def _reload_skills_catalog(
    prev_skills: list[dict] | None = None,
    prev_skills_xml: str | None = None,
) -> tuple[list[dict], str, set[str]]:
    """
    Reload skill catalog from AGENTS.md.
    If reload fails but we already have a previous snapshot, keep running with that snapshot.
    """
    try:
        current_skills_xml = load_available_skills_block()
        current_skills = parse_available_skills(current_skills_xml)
        current_skill_names = {
            s.get("name") for s in current_skills if isinstance(s, dict) and s.get("name")
        }
        return current_skills, current_skills_xml, current_skill_names
    except Exception as exc:
        if prev_skills is not None and prev_skills_xml is not None:
            prev_skill_names = {
                s.get("name") for s in prev_skills if isinstance(s, dict) and s.get("name")
            }
            print(f"[warn] failed to reload {AGENTS_MD!r}; using previous catalog. error={exc}")
            log_event(
                "skills_reload_warning",
                agents_md=AGENTS_MD,
                error=str(exc),
                using_previous_snapshot=True,
            )
            return prev_skills, prev_skills_xml, prev_skill_names
        log_event(
            "skills_reload_error",
            agents_md=AGENTS_MD,
            error=str(exc),
            using_previous_snapshot=False,
        )
        raise


def main() -> None:
    # Keep backward compatibility for `python agent.py` while using the unified CLI runtime.
    from cli.main import main as cli_main

    cli_main()
    return

    log_path = get_exec_log_path()
    if log_path and DEBUG:
        print(f"[debug] exec log file: {log_path}")
    log_event(
        "session_start",
        model=MODEL,
        llm_api=LLM_API,
        agents_md=AGENTS_MD,
        semantic_router_enabled=SEMANTIC_ROUTER_ENABLED,
        semantic_router_top_k=SEMANTIC_ROUTER_TOP_K,
        semantic_router_catalog_md=SEMANTIC_ROUTER_CATALOG_MD,
        semantic_router_catalog_jsonl=SEMANTIC_ROUTER_CATALOG_JSONL,
        skill_dynamic_fetch_enabled=SKILL_DYNAMIC_FETCH_ENABLED,
        skill_dynamic_fetch_catalog_jsonl=SKILL_DYNAMIC_FETCH_CATALOG_JSONL,
        skill_dynamic_fetch_root=str(SKILL_DYNAMIC_FETCH_ROOT),
        cli_create_on_miss=CLI_CREATE_ON_MISS,
    )
    skills, skills_xml, skill_names = _reload_skills_catalog()
    log_event("skills_catalog_loaded", skills_count=len(skills), skills=list(skill_names))

    while True:
        user = input("You> ").strip()
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            log_event("session_end", reason="user_exit")
            break
        log_event("user_input", text=user)

        # First routing decision
        skills, skills_xml, skill_names = _reload_skills_catalog(skills, skills_xml)
        current_goal = user
        decision = route_skill(user, skills, skills_xml, routing_goal=current_goal)
        # route_skill may rewrite AGENTS.md visible skills; refresh names before validation.
        skills, skills_xml, skill_names = _reload_skills_catalog(skills, skills_xml)
        if DEBUG:
            print(f"[debug] route_skill: {decision}")
        log_event("route_decision", step=1, decision=decision)
        action = decision.get("action")

        if action == "none" and CLI_CREATE_ON_MISS:
            available_skill_names_list = sorted(
                {
                    str(s.get("name") or "").strip()
                    for s in skills
                    if isinstance(s, dict) and str(s.get("name") or "").strip()
                }
            )
            created, created_skill_name, create_report = create_skill_on_miss(
                user,
                router_reason=str(decision.get("reason") or "").strip() or None,
                available_skill_names=available_skill_names_list,
            )
            log_event(
                "create_on_miss_attempt",
                user_text=user,
                router_reason=decision.get("reason"),
                created=created,
                created_skill=created_skill_name,
                report=create_report,
            )
            if created and isinstance(created_skill_name, str) and created_skill_name.strip():
                created_skill = created_skill_name.strip()
                skills, skills_xml, skill_names = _reload_skills_catalog(skills, skills_xml)
                skill_names = set(skill_names)
                skill_names.add(created_skill)
                decision = {
                    "action": "next_step",
                    "name": created_skill,
                    "user": user,
                    "reason": "create_on_miss",
                }
                action = "next_step"
                if DEBUG:
                    print(f"[debug] create_on_miss created skill={created_skill!r}; rerouted to next_step")
            elif DEBUG:
                print(f"[debug] create_on_miss skipped/failed: {create_report}")

        # Dynamic workflow - execute step by step
        if action == "next_step":
            name = decision.get("name")
            if not isinstance(name, str) or not name.strip():
                print("Assistant> Invalid decision: missing skill name\n")
                log_event("assistant_output", kind="error", text="Invalid decision: missing skill name")
                continue
            name = name.strip()
            if name not in skill_names and not has_local_skill_dir(name):
                ok, fetch_msg = ensure_skill_available(name)
                if ok:
                    skill_names = set(skill_names)
                    skill_names.add(name)
                    if DEBUG:
                        print(f"[debug] auto-fetched skill: {name!r} ({fetch_msg})")
                    log_event("skill_dynamic_fetch", skill=name, ok=True, detail=fetch_msg, step=1)
                else:
                    created = False
                    created_skill_name: str | None = None
                    create_report = ""
                    if CLI_CREATE_ON_MISS:
                        available_skill_names_list = sorted(
                            {
                                str(s.get("name") or "").strip()
                                for s in skills
                                if isinstance(s, dict) and str(s.get("name") or "").strip()
                            }
                        )
                        created, created_skill_name, create_report = create_skill_on_miss(
                            user,
                            router_reason=str(fetch_msg or "").strip() or None,
                            available_skill_names=available_skill_names_list,
                        )
                        log_event(
                            "create_on_miss_attempt",
                            user_text=user,
                            router_reason=fetch_msg,
                            created=created,
                            created_skill=created_skill_name,
                            report=create_report,
                            step=1,
                        )
                        if created and isinstance(created_skill_name, str) and created_skill_name.strip():
                            name = created_skill_name.strip()
                            skills, skills_xml, skill_names = _reload_skills_catalog(skills, skills_xml)
                            skill_names = set(skill_names)
                            skill_names.add(name)
                            if DEBUG:
                                print(
                                    f"[debug] create_on_miss created skill={name!r}; "
                                    "rerouted to next_step (step 1)"
                                )
                        elif DEBUG and create_report:
                            print(f"[debug] create_on_miss skipped/failed at step 1: {create_report}")
                    if not created:
                        print(f"Assistant> Unknown skill: {name!r}\n")
                        log_event(
                            "assistant_output",
                            kind="error",
                            text=f"Unknown skill: {name!r}",
                            detail=fetch_msg,
                            create_on_miss_report=create_report,
                        )
                        continue

            step_user = decision.get("user") or decision.get("user_text") or user
            if DEBUG:
                print(f"[debug] executing first step: skill={name!r}")
            log_event("skill_step_start", step=1, skill=name, instruction=step_user)
            try:
                result = run_one_skill_loop(step_user if isinstance(step_user, str) else user, name)
            except Exception as exc:
                result = f"ERR: {exc}"
            print(f"Assistant>\n{result}\n")
            log_event("skill_step_result", step=1, skill=name, result=result)
            log_event("assistant_output", kind="skill_result", step=1, skill=name, text=result)

            # Continue with dynamic workflow if there might be more steps.
            step_instruction = step_user if isinstance(step_user, str) else user
            summarized_result = summarize_step_output(
                question=user,
                step_skill=name,
                step_output=result,
            )
            execution_context = [
                f"[Step 1] Skill: {name}\nInstruction: {step_instruction}\nOutput:\n{summarized_result}"
            ]
            router_context = [
                build_router_step_note(
                    step_num=1,
                    step_skill=name,
                    step_instruction=step_instruction,
                    step_output=result,
                    original_goal=user,
                )
            ]
            current_goal = derive_semantic_goal(user, router_context)
            for step_num in range(2, MAX_WORKFLOW_STEPS + 1):
                skills, skills_xml, skill_names = _reload_skills_catalog(skills, skills_xml)
                next_decision = route_skill(
                    user,
                    skills,
                    skills_xml,
                    context=execution_context,
                    routing_goal=current_goal,
                )
                # route_skill may rewrite AGENTS.md visible skills; refresh names before validation.
                skills, skills_xml, skill_names = _reload_skills_catalog(skills, skills_xml)
                if DEBUG:
                    print(f"[debug] step {step_num} decision: {next_decision}")
                log_event("route_decision", step=step_num, decision=next_decision, routing_goal=current_goal)

                next_action = next_decision.get("action")
                if next_action == "done":
                    reason = next_decision.get("reason", "Task completed")
                    if DEBUG:
                        print(f"[debug] workflow complete: {reason}\n")
                    final_reason = str(reason or "").strip() or "Task completed"
                    print(f"Assistant> {final_reason}\n")
                    log_event("assistant_output", kind="workflow_done", step=step_num, text=final_reason)
                    log_event("workflow_done", step=step_num, reason=reason)
                    break
                if next_action == "none":
                    create_report = ""
                    if CLI_CREATE_ON_MISS:
                        available_skill_names_list = sorted(
                            {
                                str(s.get("name") or "").strip()
                                for s in skills
                                if isinstance(s, dict) and str(s.get("name") or "").strip()
                            }
                        )
                        created, created_skill_name, create_report = create_skill_on_miss(
                            user,
                            router_reason=str(next_decision.get("reason") or "").strip() or None,
                            available_skill_names=available_skill_names_list,
                        )
                        log_event(
                            "create_on_miss_attempt",
                            user_text=user,
                            router_reason=next_decision.get("reason"),
                            created=created,
                            created_skill=created_skill_name,
                            report=create_report,
                            step=step_num,
                        )
                        if created and isinstance(created_skill_name, str) and created_skill_name.strip():
                            created_skill = created_skill_name.strip()
                            skills, skills_xml, skill_names = _reload_skills_catalog(skills, skills_xml)
                            skill_names = set(skill_names)
                            skill_names.add(created_skill)
                            next_decision = {
                                "action": "next_step",
                                "name": created_skill,
                                "user": user,
                                "reason": "create_on_miss",
                            }
                            next_action = "next_step"
                            if DEBUG:
                                print(
                                    f"[debug] create_on_miss created skill={created_skill!r}; "
                                    f"rerouted to next_step (step {step_num})"
                                )
                        elif DEBUG and create_report:
                            print(
                                f"[debug] create_on_miss skipped/failed at step {step_num}: "
                                f"{create_report}"
                            )
                    if next_action == "none":
                        if DEBUG:
                            print("[debug] no more steps needed\n")
                        log_event(
                            "workflow_none",
                            step=step_num,
                            reason=next_decision.get("reason"),
                            create_on_miss_report=create_report,
                        )
                        break
                if next_action != "next_step":
                    log_event(
                        "workflow_unknown_action",
                        step=step_num,
                        action=next_action,
                        decision=next_decision,
                    )
                    break

                next_name = next_decision.get("name")
                if not isinstance(next_name, str) or not next_name.strip():
                    log_event("workflow_invalid_next_skill", step=step_num, decision=next_decision)
                    break
                next_name = next_name.strip()
                if next_name not in skill_names and not has_local_skill_dir(next_name):
                    ok, fetch_msg = ensure_skill_available(next_name)
                    if ok:
                        skill_names = set(skill_names)
                        skill_names.add(next_name)
                        if DEBUG:
                            print(f"[debug] auto-fetched skill: {next_name!r} ({fetch_msg})")
                        log_event(
                            "skill_dynamic_fetch",
                            skill=next_name,
                            ok=True,
                            detail=fetch_msg,
                            step=step_num,
                        )
                    else:
                        created = False
                        created_skill_name: str | None = None
                        create_report = ""
                        if CLI_CREATE_ON_MISS:
                            available_skill_names_list = sorted(
                                {
                                    str(s.get("name") or "").strip()
                                    for s in skills
                                    if isinstance(s, dict) and str(s.get("name") or "").strip()
                                }
                            )
                            created, created_skill_name, create_report = create_skill_on_miss(
                                user,
                                router_reason=str(fetch_msg or "").strip() or None,
                                available_skill_names=available_skill_names_list,
                            )
                            log_event(
                                "create_on_miss_attempt",
                                user_text=user,
                                router_reason=fetch_msg,
                                created=created,
                                created_skill=created_skill_name,
                                report=create_report,
                                step=step_num,
                            )
                            if created and isinstance(created_skill_name, str) and created_skill_name.strip():
                                next_name = created_skill_name.strip()
                                skills, skills_xml, skill_names = _reload_skills_catalog(skills, skills_xml)
                                skill_names = set(skill_names)
                                skill_names.add(next_name)
                                if DEBUG:
                                    print(
                                        f"[debug] create_on_miss created skill={next_name!r}; "
                                        f"rerouted to next_step (step {step_num})"
                                    )
                            elif DEBUG and create_report:
                                print(
                                    f"[debug] create_on_miss skipped/failed at step {step_num}: "
                                    f"{create_report}"
                                )
                        if not created:
                            print(f"Assistant> Unknown skill: {next_name!r}\n")
                            log_event(
                                "assistant_output",
                                kind="error",
                                text=f"Unknown skill: {next_name!r}",
                                detail=fetch_msg,
                                step=step_num,
                                create_on_miss_report=create_report,
                            )
                            break

                next_user = next_decision.get("user") or next_decision.get("user_text") or user
                next_user_str = next_user if isinstance(next_user, str) else user
                full_user = next_user_str.strip()
                full_user += "\n\n# Context (previous step outputs)\n" + "\n---\n".join(execution_context)

                if DEBUG:
                    print(f"[debug] executing step {step_num}: skill={next_name!r}")
                log_event(
                    "skill_step_start",
                    step=step_num,
                    skill=next_name,
                    instruction=next_user_str,
                    full_input=full_user,
                )
                try:
                    result = run_one_skill_loop(full_user, next_name)
                except Exception as exc:
                    result = f"ERR: {exc}"
                print(f"Assistant>\n{result}\n")
                log_event("skill_step_result", step=step_num, skill=next_name, result=result)
                log_event("assistant_output", kind="skill_result", step=step_num, skill=next_name, text=result)
                # Summarize long outputs before adding to context
                summarized_result = summarize_step_output(
                    question=user,
                    step_skill=next_name,
                    step_output=result,
                )
                execution_context.append(
                    f"[Step {step_num}] Skill: {next_name}\nInstruction: {next_user_str}\nOutput:\n{summarized_result}"
                )
                router_context.append(
                    build_router_step_note(
                        step_num=step_num,
                        step_skill=next_name,
                        step_instruction=next_user_str,
                        step_output=result,
                        original_goal=user,
                    )
                )
                current_goal = derive_semantic_goal(user, router_context)
            continue

        # No skill needed / no route match / direct-complete
        if action in ("none", "done"):
            reason = str(decision.get("reason", "") or "").strip().lower()
            # Align with TUI: if router says "no_skill_needed", do normal chat fallback.
            if action == "none" or reason in {"no_skill_needed", "no-skill-needed", "no skill needed"}:
                fallback = openrouter_messages(
                    CHAT_SYSTEM_PROMPT,
                    [{"role": "user", "content": user}],
                )
                print(f"Assistant> {fallback}\n")
                log_event("assistant_output", kind="chat_fallback", text=fallback, reason=reason)
            else:
                final_text = f"[Complete] {decision.get('reason', '')}"
                print(f"Assistant> {final_text}\n")
                log_event("assistant_output", kind="done", text=final_text, reason=decision.get("reason"))
            continue

        # Unknown action - fallback
        fallback = openrouter_messages(
            CHAT_SYSTEM_PROMPT,
            [{"role": "user", "content": user}],
        )
        print(f"Assistant> {fallback}\n")
        log_event("assistant_output", kind="chat_fallback_unknown_action", text=fallback, action=action)


if __name__ == "__main__":
    main()
