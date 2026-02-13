"""CLI-native workflow runner with no TUI dependency."""

from __future__ import annotations

import json
import time

from core.llm import openrouter_messages
import core.router as router_module
from core.skill_engine.skill_catalog import (
    build_router_step_note,
    derive_semantic_goal,
    load_available_skills_block,
    parse_available_skills,
)
from core.skill_engine.skill_resolver import (
    ensure_skill_available,
    has_local_skill_dir,
    install_or_update_skill,
    openskills_read,
)
from core.skill_engine.skill_runner import (
    _count_approx_tokens as count_approx_tokens,
    run_skill_once_with_plan,
    summarize_step_output,
)

_CONVERSATION_CONTEXT_MAX_TOKENS = 80000


def _is_likely_format_error(output: str) -> bool:
    s = (output or "").strip()
    if not s:
        return False
    needles = [
        "Invalid skill plan (refused)",
        "ERR: plan contained no ops",
        "ERR: exceeded max_rounds",
        "Invalid plan: received type=code",
        "Missing ops",
        "missing required key 'type'",
        "op missing required key 'type'",
        "KeyError: 'type'",
        "Unknown operation type",
        "unknown operation type",
        "operation type: None",
        "op_type is None",
        "unsupported operation",
        "invalid operation",
        "Error in None:",
        "TypeError: 'NoneType'",
    ]
    return any(n in s for n in needles)


def _is_skill_execution_error(output: str) -> bool:
    s = (output or "").strip()
    if not s:
        return False
    error_patterns = [
        "Unknown operation type",
        "unknown operation type",
        "Unsupported operation",
        "Invalid operation",
        "operation type: None",
        "Error in ",
        "missing required",
        "KeyError:",
        "TypeError:",
        "ValueError:",
        "AttributeError:",
        "FileNotFoundError:",
        "PermissionError:",
    ]
    return any(p in s for p in error_patterns)


def _should_optimize_skill(output: str) -> bool:
    return _is_likely_format_error(output) or _is_skill_execution_error(output)


def optimize_skill_with_creator(
    skill_name: str,
    *,
    user_text: str,
    step_user: str,
    error_output: str,
    last_plan: dict | None = None,
) -> tuple[bool, str]:
    """Use skill-creator to patch a failing skill and re-install it."""
    if not has_local_skill_dir(skill_name):
        return False, f"missing local skill dir for {skill_name!r}"

    try:
        skill_md = openskills_read(skill_name)
    except Exception:
        skill_md = ""

    plan_preview = json.dumps(last_plan or {}, ensure_ascii=False, indent=2)[:2000]
    prompt = (
        "Update the existing skill below to fix the failure. "
        "Keep the skill name unchanged, and keep the response format unchanged. "
        "Make minimal, precise edits.\n\n"
        "Goal: prevent ops/format errors (especially missing op.type) and ensure the skill reliably executes.\n"
        "- Ensure SKILL.md clearly specifies the JSON schema and that every op has a string key `type`.\n"
        "- Keep this as a SKILL.md-only skill.\n"
        "- Do NOT add task-specific hacks; fix the general mechanism.\n\n"
        f"Skill name to update (MUST match): {skill_name}\n\n"
        f"User request:\n{user_text}\n\n"
        f"Instruction given to skill (this step):\n{step_user}\n\n"
        f"Observed error output:\n{error_output}\n\n"
        f"Last plan JSON (truncated):\n{plan_preview}\n\n"
        f"Current SKILL.md:\n{skill_md}\n"
    )

    try:
        result, plan = run_skill_once_with_plan(prompt, "skill-creator", max_rounds=20)
    except Exception as exc:
        return False, f"skill-creator failed: {exc}"

    if not isinstance(plan, dict):
        return False, "skill-creator returned no plan"
    if (plan.get("action") or "").strip() != "update":
        return False, f"skill-creator unexpected action={plan.get('action')!r}"
    if (plan.get("skill_name") or "").strip() != skill_name:
        return False, f"skill-creator returned mismatched skill_name={plan.get('skill_name')!r}"

    ok, msg = install_or_update_skill(skill_name)
    if not ok:
        return False, f"install/update failed: {msg}"

    return True, result or "optimized"


class SkillWorkflowRunner:
    """Step-wise workflow runner for CLI."""

    def __init__(
        self,
        auto_sync: bool = False,
        *,
        optimize_on_error: bool = True,
        optimize_attempts: int = 1,
        debug: bool = False,
    ):
        self.sync_message = ""
        self.optimize_on_error = bool(optimize_on_error)
        self.optimize_attempts = max(0, int(optimize_attempts))
        self.debug = bool(debug)

        # auto_sync reserved for compatibility; no-op in CLI runner.
        _ = auto_sync

        try:
            self.skills_xml = load_available_skills_block()
            self.skills = parse_available_skills(self.skills_xml)
            self.skill_names = {s.get("name") for s in self.skills if isinstance(s, dict) and s.get("name")}
        except Exception as exc:
            print(f"Warning: Failed to load skills: {exc}")
            self.skills_xml = ""
            self.skills = []
            self.skill_names = set()

        self.context: list[str] = []
        self.conversation_history: list[dict] = []

    def _debug_print_router_skills(self, round_no: int) -> None:
        if not self.debug:
            return
        names = sorted(
            {
                str(s.get("name")).strip()
                for s in self.skills
                if isinstance(s, dict) and str(s.get("name") or "").strip()
            }
        )
        if not names:
            print(f"[debug] router round {round_no}: local_skills(0)")
            return
        preview_n = 40
        preview = names[:preview_n]
        more = len(names) - len(preview)
        suffix = f", ... (+{more} more)" if more > 0 else ""
        print(
            f"[debug] router round {round_no}: local_skills({len(names)}): "
            + ", ".join(preview)
            + suffix
        )

    def _debug_timing(self, label: str, started_at: float) -> None:
        if not self.debug:
            return
        elapsed = max(0.0, time.perf_counter() - float(started_at))
        print(f"[debug][timing] {label}: {elapsed:.3f}s")

    def reload_skills_metadata(self) -> None:
        self.skills_xml = load_available_skills_block()
        self.skills = parse_available_skills(self.skills_xml)
        self.skill_names = {s.get("name") for s in self.skills if isinstance(s, dict) and s.get("name")}

    def has_skills(self) -> bool:
        return bool(self.skills)

    def get_skill_names(self) -> list[str]:
        return sorted(
            {
                str(name).strip()
                for name in self.skill_names
                if isinstance(name, str) and str(name).strip()
            }
        )

    def reset_context(self) -> None:
        self.context = []
        self.conversation_history = []

    def set_conversation_history(self, messages: list[dict]) -> None:
        self.conversation_history = list(messages) if messages else []

    def _summarize_conversation(self, text: str, max_tokens: int = 500) -> str:
        try:
            prompt = f"""Summarize this conversation concisely, preserving:
- Key topics discussed
- Important decisions or conclusions
- Specific data, file paths, or code mentioned
- Any pending tasks or questions

Target length: ~{max_tokens} tokens.

Conversation:
{text}

Return ONLY the summary, no meta-commentary."""
            t_summary = time.perf_counter()
            summary = openrouter_messages(
                "You are a precise summarizer. Return only essential information.",
                [{"role": "user", "content": prompt}],
            )
            self._debug_timing("conversation_summary_llm", t_summary)
            text_out = str(summary or "").strip()
            if text_out:
                return text_out
        except Exception:
            pass

        max_chars = max_tokens * 4
        text = str(text or "")
        if len(text) > max_chars:
            return text[:max_chars] + "...[truncated]"
        return text

    def _build_conversation_context(self, max_tokens: int | None = None) -> str:
        t_ctx = time.perf_counter()
        if max_tokens is None:
            max_tokens = _CONVERSATION_CONTEXT_MAX_TOKENS
        if not self.conversation_history:
            self._debug_timing("build_conversation_context(empty)", t_ctx)
            return ""

        parts: list[str] = []
        for msg in self.conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(f"[{role}]: {content}")

        if not parts:
            self._debug_timing("build_conversation_context(no_parts)", t_ctx)
            return ""

        full_context = "\n\n".join(parts)
        if count_approx_tokens(full_context) <= max_tokens:
            self._debug_timing("build_conversation_context(full)", t_ctx)
            return f"\n\n=== CONVERSATION HISTORY ===\n{full_context}\n=== END CONVERSATION ===\n"

        recent_count = min(4, len(self.conversation_history))
        older_messages = self.conversation_history[:-recent_count] if recent_count < len(self.conversation_history) else []
        recent_messages = self.conversation_history[-recent_count:]

        recent_parts: list[str] = []
        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                recent_parts.append(f"[{role}]: {content}")
        recent_text = "\n\n".join(recent_parts)

        if not older_messages:
            self._debug_timing("build_conversation_context(recent_only)", t_ctx)
            return f"\n\n=== CONVERSATION HISTORY ===\n{recent_text}\n=== END CONVERSATION ===\n"

        older_parts: list[str] = []
        for msg in older_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                older_parts.append(f"[{role}]: {content}")
        older_text = "\n\n".join(older_parts)

        remaining_tokens = max(500, max_tokens - count_approx_tokens(recent_text) - 200)
        summary = self._summarize_conversation(older_text, max_tokens=remaining_tokens)
        out = (
            "\n\n=== CONVERSATION HISTORY ===\n"
            f"[Earlier conversation summary]: {summary}\n\n"
            f"[Recent messages]:\n{recent_text}\n"
            "=== END CONVERSATION ===\n"
        )
        self._debug_timing("build_conversation_context(summary)", t_ctx)
        return out

    def run_workflow_steps(self, user_text: str, max_steps: int = 20):
        """Yield step events as (step_info, result)."""
        t_workflow = time.perf_counter()
        try:
            t = time.perf_counter()
            self.reload_skills_metadata()
            self._debug_timing("reload_skills_metadata(start)", t)
        except Exception:
            pass

        if not self.skills:
            yield {"step_num": 0, "skill_name": None, "status": "no_skills", "is_final": True}, ""
            self._debug_timing("workflow_total", t_workflow)
            return

        t = time.perf_counter()
        conversation_context = self._build_conversation_context()
        self._debug_timing("build_conversation_context", t)
        enriched_user_text = user_text
        if conversation_context:
            enriched_user_text = f"{user_text}\n{conversation_context}"

        try:
            t = time.perf_counter()
            self.reload_skills_metadata()
            self._debug_timing("reload_skills_metadata(pre-route-1)", t)
        except Exception:
            pass

        current_goal = user_text
        self._debug_print_router_skills(1)
        t = time.perf_counter()
        decision = router_module.route_skill(
            enriched_user_text,
            self.skills,
            self.skills_xml,
            routing_goal=current_goal,
            debug=self.debug,
        )
        self._debug_timing("route_skill(round=1)", t)

        try:
            t = time.perf_counter()
            self.reload_skills_metadata()
            self._debug_timing("reload_skills_metadata(post-route-1)", t)
        except Exception:
            pass

        action = decision.get("action")
        if action == "none":
            yield {
                "step_num": 0,
                "skill_name": None,
                "status": "no_match",
                "reason": decision.get("reason", ""),
                "is_final": True,
            }, ""
            self._debug_timing("workflow_total", t_workflow)
            return

        if action == "done":
            yield {
                "step_num": 0,
                "skill_name": None,
                "status": "done",
                "reason": decision.get("reason", "Task complete"),
                "is_final": True,
            }, decision.get("reason", "Task complete")
            self._debug_timing("workflow_total", t_workflow)
            return

        if action not in ("next_step", "load_skill"):
            yield {
                "step_num": 0,
                "skill_name": None,
                "status": "unknown_action",
                "reason": decision.get("reason", ""),
                "is_final": True,
            }, ""
            self._debug_timing("workflow_total", t_workflow)
            return

        skill_name = str(decision.get("name", "") or "").strip()
        if not skill_name:
            yield {"step_num": 1, "skill_name": "", "status": "error", "is_final": True}, "Unknown skill: ''"
            self._debug_timing("workflow_total", t_workflow)
            return

        if skill_name not in self.skill_names and not has_local_skill_dir(skill_name):
            t = time.perf_counter()
            ok, fetch_msg = ensure_skill_available(skill_name)
            self._debug_timing(f"ensure_skill_available(step=1, skill={skill_name})", t)
            if not ok:
                yield {
                    "step_num": 1,
                    "skill_name": skill_name,
                    "status": "error",
                    "is_final": True,
                }, f"Unknown skill: {skill_name} ({fetch_msg})"
                self._debug_timing("workflow_total", t_workflow)
                return
            self.skill_names.add(skill_name)
            try:
                t = time.perf_counter()
                self.reload_skills_metadata()
                self._debug_timing("reload_skills_metadata(post-fetch-step-1)", t)
            except Exception:
                pass

        step_user = decision.get("user") or user_text
        yield {"step_num": 1, "skill_name": skill_name, "status": "running", "is_final": False}, None

        step_user_text = step_user if isinstance(step_user, str) else user_text
        if conversation_context:
            step_user_text = f"{step_user_text}\n{conversation_context}"

        t = time.perf_counter()
        result, last_plan = run_skill_once_with_plan(step_user_text, skill_name, max_rounds=50)
        self._debug_timing(f"run_skill_once(step=1, skill={skill_name})", t)

        if self.optimize_on_error and self.optimize_attempts > 0 and _should_optimize_skill(result):
            for attempt in range(1, self.optimize_attempts + 1):
                yield {
                    "step_num": 1,
                    "skill_name": skill_name,
                    "status": "optimizing",
                    "attempt": attempt,
                    "is_final": False,
                }, result
                t = time.perf_counter()
                ok, report = optimize_skill_with_creator(
                    skill_name,
                    user_text=user_text,
                    step_user=step_user_text,
                    error_output=result,
                    last_plan=last_plan,
                )
                self._debug_timing(
                    f"optimize_skill_with_creator(step=1, skill={skill_name}, attempt={attempt})",
                    t,
                )
                yield {
                    "step_num": 1,
                    "skill_name": skill_name,
                    "status": "optimized",
                    "attempt": attempt,
                    "ok": ok,
                    "is_final": False,
                }, report
                if not ok:
                    break
                try:
                    t = time.perf_counter()
                    self.reload_skills_metadata()
                    self._debug_timing("reload_skills_metadata(post-optimize-step-1)", t)
                except Exception:
                    pass
                t = time.perf_counter()
                result, last_plan = run_skill_once_with_plan(step_user_text, skill_name, max_rounds=50)
                self._debug_timing(
                    f"run_skill_once(step=1-retry, skill={skill_name}, attempt={attempt})",
                    t,
                )
                if not _should_optimize_skill(result):
                    break

        yield {"step_num": 1, "skill_name": skill_name, "status": "completed", "is_final": False}, result

        t = time.perf_counter()
        summarized = summarize_step_output(
            question=user_text,
            step_skill=skill_name,
            step_output=result,
        )
        self._debug_timing(f"summarize_step_output(step=1, skill={skill_name})", t)
        self.context = [f"[Step 1] Skill: {skill_name}\nInstruction: {step_user}\nOutput:\n{summarized}"]
        router_context = [
            build_router_step_note(
                step_num=1,
                step_skill=skill_name,
                step_instruction=str(step_user),
                step_output=result,
                original_goal=user_text,
            )
        ]
        t = time.perf_counter()
        current_goal = derive_semantic_goal(user_text, router_context)
        self._debug_timing("derive_semantic_goal(step=1)", t)

        for step_num in range(2, max_steps + 1):
            try:
                t = time.perf_counter()
                self.reload_skills_metadata()
                self._debug_timing(f"reload_skills_metadata(pre-route-step-{step_num})", t)
            except Exception:
                pass

            self._debug_print_router_skills(step_num)
            t = time.perf_counter()
            next_decision = router_module.route_skill(
                user_text,
                self.skills,
                self.skills_xml,
                context=self.context,
                routing_goal=current_goal,
                debug=self.debug,
            )
            self._debug_timing(f"route_skill(round={step_num})", t)

            try:
                t = time.perf_counter()
                self.reload_skills_metadata()
                self._debug_timing(f"reload_skills_metadata(post-route-step-{step_num})", t)
            except Exception:
                pass

            next_action = next_decision.get("action")
            if next_action == "done":
                reason = next_decision.get("reason", "Task completed")
                yield {
                    "step_num": step_num,
                    "skill_name": None,
                    "status": "done",
                    "reason": reason,
                    "is_final": True,
                }, result
                self._debug_timing("workflow_total", t_workflow)
                return

            if next_action != "next_step":
                yield {
                    "step_num": step_num,
                    "skill_name": None,
                    "status": "done",
                    "reason": next_decision.get("reason", "Task completed"),
                    "is_final": True,
                }, result
                self._debug_timing("workflow_total", t_workflow)
                return

            next_name = str(next_decision.get("name", "") or "").strip()
            if not next_name:
                yield {
                    "step_num": step_num,
                    "skill_name": next_name,
                    "status": "done",
                    "reason": "missing next skill name",
                    "is_final": True,
                }, result
                self._debug_timing("workflow_total", t_workflow)
                return

            if next_name not in self.skill_names and not has_local_skill_dir(next_name):
                t = time.perf_counter()
                ok, fetch_msg = ensure_skill_available(next_name)
                self._debug_timing(f"ensure_skill_available(step={step_num}, skill={next_name})", t)
                if not ok:
                    yield {
                        "step_num": step_num,
                        "skill_name": next_name,
                        "status": "error",
                        "is_final": True,
                    }, f"Unknown skill: {next_name} ({fetch_msg})"
                    self._debug_timing("workflow_total", t_workflow)
                    return
                self.skill_names.add(next_name)
                try:
                    t = time.perf_counter()
                    self.reload_skills_metadata()
                    self._debug_timing(f"reload_skills_metadata(post-fetch-step-{step_num})", t)
                except Exception:
                    pass

            next_user = next_decision.get("user") or user_text
            yield {
                "step_num": step_num,
                "skill_name": next_name,
                "status": "running",
                "is_final": False,
            }, None

            full_user = f"{next_user}\n\n# Context (previous step outputs)\n" + "\n---\n".join(self.context)
            if conversation_context:
                full_user = f"{full_user}\n{conversation_context}"

            t = time.perf_counter()
            result, last_plan = run_skill_once_with_plan(full_user, next_name, max_rounds=50)
            self._debug_timing(f"run_skill_once(step={step_num}, skill={next_name})", t)

            if self.optimize_on_error and self.optimize_attempts > 0 and _should_optimize_skill(result):
                for attempt in range(1, self.optimize_attempts + 1):
                    yield {
                        "step_num": step_num,
                        "skill_name": next_name,
                        "status": "optimizing",
                        "attempt": attempt,
                        "is_final": False,
                    }, result
                    t = time.perf_counter()
                    ok, report = optimize_skill_with_creator(
                        next_name,
                        user_text=user_text,
                        step_user=full_user,
                        error_output=result,
                        last_plan=last_plan,
                    )
                    self._debug_timing(
                        f"optimize_skill_with_creator(step={step_num}, skill={next_name}, attempt={attempt})",
                        t,
                    )
                    yield {
                        "step_num": step_num,
                        "skill_name": next_name,
                        "status": "optimized",
                        "attempt": attempt,
                        "ok": ok,
                        "is_final": False,
                    }, report
                    if not ok:
                        break
                    try:
                        t = time.perf_counter()
                        self.reload_skills_metadata()
                        self._debug_timing(f"reload_skills_metadata(post-optimize-step-{step_num})", t)
                    except Exception:
                        pass
                    t = time.perf_counter()
                    result, last_plan = run_skill_once_with_plan(full_user, next_name, max_rounds=50)
                    self._debug_timing(
                        f"run_skill_once(step={step_num}-retry, skill={next_name}, attempt={attempt})",
                        t,
                    )
                    if not _should_optimize_skill(result):
                        break

            yield {
                "step_num": step_num,
                "skill_name": next_name,
                "status": "completed",
                "is_final": False,
            }, result

            t = time.perf_counter()
            summarized = summarize_step_output(
                question=user_text,
                step_skill=next_name,
                step_output=result,
            )
            self._debug_timing(f"summarize_step_output(step={step_num}, skill={next_name})", t)
            self.context.append(f"[Step {step_num}] Skill: {next_name}\nInstruction: {next_user}\nOutput:\n{summarized}")
            router_context.append(
                build_router_step_note(
                    step_num=step_num,
                    step_skill=next_name,
                    step_instruction=str(next_user),
                    step_output=result,
                    original_goal=user_text,
                )
            )
            t = time.perf_counter()
            current_goal = derive_semantic_goal(user_text, router_context)
            self._debug_timing(f"derive_semantic_goal(step={step_num})", t)

        yield {
            "step_num": max_steps,
            "skill_name": None,
            "status": "max_steps",
            "is_final": True,
        }, result
        self._debug_timing("workflow_total", t_workflow)
