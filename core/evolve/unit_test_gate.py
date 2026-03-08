"""Optimize-stage unit test gate — verify a skill works after optimization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .judge import judge_answer_soft_score_0_to_10
from .tracker import DEFAULT_JUDGE_MODEL
from .skill_helpers import read_skill_md_text as _read_skill_md_text
from .runner import execute_task_once
from .text_utils import json_fragment, normalize_skill_name, normalize_space
from .tracker import append_jsonl, error_judge_payload as _error_judge_payload


# ---------------------------------------------------------------------------
# Test-case generation categories
# ---------------------------------------------------------------------------

_CASE_CATEGORIES: list[tuple[str, str]] = [
    (
        "core_functionality",
        "Test the PRIMARY capability described in the skill specification. "
        "Pick the most important function the skill provides and write a concrete question for it.",
    ),
    (
        "edge_case",
        "Test a BOUNDARY CONDITION or unusual input for the skill. "
        "Think about edge cases: empty input, very large values, special characters, "
        "or uncommon but valid scenarios the skill should handle.",
    ),
    (
        "input_variation",
        "Test the skill with a DIFFERENT INPUT FORMAT or style than typical usage. "
        "For example, if the skill normally receives formal text, try informal phrasing; "
        "if it handles numbers, try a different numeric format.",
    ),
    (
        "error_handling",
        "Test how the skill handles AMBIGUOUS or TRICKY input. "
        "Create a question that requires the skill to deal with something slightly "
        "misleading or that needs careful interpretation.",
    ),
    (
        "complex_scenario",
        "Test the skill with a COMPLEX, multi-part question that exercises multiple "
        "aspects of the skill specification. Combine two or more capabilities if possible.",
    ),
]


# ---------------------------------------------------------------------------
# Case builders (kept from original)
# ---------------------------------------------------------------------------

async def _build_flexible_runtime_skill_unit_case(
    *,
    agent: Any,
    target_skill: str,
    target_skill_dir: Path,
    task: dict[str, Any],
    skill_manager: Any = None,
    category_instruction: str = "",
    previous_questions: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a unit test case using the agent."""
    skill_md = _read_skill_md_text(target_skill_dir, max_chars=6000)
    if not skill_md:
        return {
            "status": "case_generation_failed",
            "reason": "missing_skill_md",
        }

    task_q = normalize_space(str(task.get("question") or ""))
    task_a = normalize_space(str(task.get("answer") or ""))
    attachments = task.get("attachments")
    attachments_text = (
        json.dumps(attachments, ensure_ascii=False, indent=2)
        if attachments is not None
        else "[]"
    )

    category_block = ""
    if category_instruction:
        category_block = f"\nTEST CATEGORY INSTRUCTION:\n{category_instruction}\n"

    dedup_block = ""
    if previous_questions:
        numbered = "\n".join(
            f"  {i}. {q}" for i, q in enumerate(previous_questions, 1)
        )
        dedup_block = (
            f"\nDo NOT duplicate any of these previously generated questions:\n{numbered}\n"
        )

    gen_prompt = (
        "You are creating a unit test for the skill described below.\n"
        "Generate ONE test case that verifies the skill's capability.\n\n"
        "CRITICAL: Return ONLY a raw JSON object (no markdown, no ```json```, no explanation):\n"
        '{"question": "...", "expected_answer": "..."}\n\n'
        "Requirements:\n"
        "- The question MUST test a real capability described in the skill specification\n"
        "- The question should be a concrete, domain-specific question that requires the skill's expertise\n"
        "- expected_answer must be a concise, deterministic, verifiable answer\n"
        "- Do NOT create trivial or meta questions (like 'can you invoke X skill')\n"
        "- Create questions at the difficulty level of a professional exam in the domain\n"
        f"{category_block}"
        f"{dedup_block}\n"
        f"Skill specification:\n{skill_md}\n\n"
        "Reference domain context (for difficulty calibration only):\n"
        f"Question: {task_q[:300]}\n"
        f"Answer: {task_a[:200]}\n"
    )

    generation_attempts: list[dict[str, Any]] = []
    generated_question = ""
    generated_expected = ""
    generation_reason = ""

    for attempt in range(1, 3):
        if attempt > 1:
            prev = generation_attempts[-1] if generation_attempts else {}
            prev_out = str(prev.get("raw_output") or "").strip()
            prev_reason = str(prev.get("reason") or "non_json_or_missing_fields").strip()
            gen_prompt = (
                "Your previous output was not valid JSON. Return ONLY this format:\n"
                '{"question": "your question here", "expected_answer": "the answer"}\n'
                "No markdown code fences, no extra text.\n\n"
                f"Previous failure reason: {prev_reason}\n"
                f"Previous output:\n{prev_out}\n\n"
                f"Skill specification:\n{skill_md}\n"
            )
        try:
            llm_response = await agent.llm.chat(
                messages=[{"role": "user", "content": gen_prompt}],
            )
            raw_output = str(llm_response.content or "").strip()
            parsed = json_fragment(raw_output)
            if isinstance(parsed, dict):
                q = normalize_space(str(parsed.get("question") or ""))
                a = normalize_space(str(parsed.get("expected_answer") or ""))
                if q and a:
                    generated_question = q
                    generated_expected = a
                    generation_attempts.append(
                        {
                            "attempt": attempt,
                            "status": "ok",
                            "question_preview": q[:280],
                            "expected_answer_preview": a[:180],
                        }
                    )
                    break
            generation_reason = "non_json_or_missing_fields"
            generation_attempts.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "reason": generation_reason,
                    "raw_output": raw_output[:1200],
                }
            )
        except Exception as exc:
            generation_reason = f"generation_error:{type(exc).__name__}: {exc}"
            generation_attempts.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "reason": generation_reason,
                }
            )

    if not generated_question or not generated_expected:
        # Fallback: contract smoke test.
        return _build_contract_fallback_skill_unit_case(
            target_skill=target_skill,
            generation_attempts=generation_attempts,
        )

    prompt = (
        f"This is an optimize-stage unit gate task for skill `{target_skill}`.\n"
        f"You MUST invoke skill `{target_skill}` at least once.\n"
        "Solve the generated unit requirement below using the agent workflow and available skills.\n"
        "Return only the final answer string with no extra words.\n\n"
        "Generated unit requirement:\n"
        f"{generated_question}\n\n"
        "Task attachments metadata (if any):\n"
        f"{attachments_text}\n\n"
        "Target skill specification excerpt:\n"
        f"{skill_md}\n"
    )
    return {
        "status": "ok",
        "question": prompt,
        "expected_answer": generated_expected,
        "evidence": {
            "mode": "agent_framework_generated_case",
            "source_task_id": task.get("id"),
            "source_answer_type": task.get("answer_type"),
            "source_task_question": task_q,
            "generated_question": generated_question,
            "generation_attempts": generation_attempts,
        },
    }


def _build_contract_fallback_skill_unit_case(
    *,
    target_skill: str,
    generation_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for item in generation_attempts:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").strip()
        if status == "ok":
            continue
        failures.append(
            {"status": status, "reason": str(item.get("reason") or "").strip()}
        )

    prompt = (
        f"This is a fallback micro unit test for skill `{target_skill}`.\n"
        f"You MUST invoke skill `{target_skill}` at least once.\n"
        "Run one minimal smoke-test call that follows the skill interface.\n"
        f"After the call, return exactly this token and nothing else:\n{target_skill}"
    )
    return {
        "status": "ok",
        "question": prompt,
        "expected_answer": target_skill,
        "evidence": {
            "mode": "fallback_contract_smoke_test",
            "generation_failures": failures,
        },
    }


# ---------------------------------------------------------------------------
# Single-case execution + weighted scoring
# ---------------------------------------------------------------------------

async def _execute_and_judge_single_case(
    *,
    agent: Any,
    target_skill: str,
    case: dict[str, Any],
    judge_model: str,
    judge_retries: int,
    judge_pass_score_0_to_10: float,
    skill_manager: Any = None,
    correctness_weight: float = 0.7,
    skill_usage_weight: float = 0.2,
    trace_done_weight: float = 0.1,
) -> dict[str, Any]:
    """Execute one test case and return a scored result dict."""
    case_question = str(case.get("question") or "").strip()
    expected_answer = str(case.get("expected_answer") or "").strip()

    if not case_question or not expected_answer:
        return {
            "passed": False,
            "weighted_score": 0.0,
            "case": case,
            "reason": "empty_case_payload",
            "checks": {
                "answer_correct": False,
                "used_target_skill": False,
                "trace_status_done": False,
            },
        }

    try:
        result = await execute_task_once(
            agent,
            user_text=case_question,
            create_on_miss_enabled=False,
            skill_manager=skill_manager,
        )
        judge = await judge_answer_soft_score_0_to_10(
            question=case_question,
            model_answer=result.answer,
            expected_answer=expected_answer,
            judge_model=str(judge_model).strip() or DEFAULT_JUDGE_MODEL,
            retries=max(1, int(judge_retries)),
            pass_threshold=max(0.0, min(10.0, float(judge_pass_score_0_to_10))),
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        return {
            "passed": False,
            "weighted_score": 0.0,
            "question": case_question,
            "expected_answer": expected_answer,
            "predicted_answer": f"ERROR: {err}",
            "evidence": case.get("evidence"),
            "execution_error": err,
            "judge": _error_judge_payload(
                err, str(judge_model).strip() or DEFAULT_JUDGE_MODEL
            ),
            "trace_status": "error_gate_execution",
            "used_skills": [],
            "checks": {
                "answer_correct": False,
                "used_target_skill": False,
                "trace_status_done": False,
            },
        }

    target_norm = normalize_skill_name(target_skill)
    used_norms = {normalize_skill_name(x) for x in (result.used_skills or [])}
    used_target_skill = target_norm in used_norms
    trace_done = str(result.status or "").strip().lower() == "done"
    answer_correct = bool(judge.get("is_correct"))
    try:
        answer_score_0_to_10 = float(judge.get("score_0_to_10"))
    except Exception:
        answer_score_0_to_10 = float(judge.get("score") or 0.0) * 10.0
    answer_score_0_to_10 = max(0.0, min(10.0, answer_score_0_to_10))

    weighted_score = (
        correctness_weight * (answer_score_0_to_10 / 10.0)
        + skill_usage_weight * (1.0 if used_target_skill else 0.0)
        + trace_done_weight * (1.0 if trace_done else 0.0)
    )
    passed = bool(answer_correct and used_target_skill and trace_done)

    return {
        "passed": passed,
        "weighted_score": round(weighted_score, 4),
        "question": case_question,
        "expected_answer": expected_answer,
        "predicted_answer": result.answer,
        "evidence": case.get("evidence"),
        "judge": judge,
        "trace_status": result.status,
        "used_skills": result.used_skills,
        "checks": {
            "answer_correct": answer_correct,
            "answer_score_0_to_10": answer_score_0_to_10,
            "used_target_skill": used_target_skill,
            "trace_status_done": trace_done,
            "weighted_score": round(weighted_score, 4),
        },
    }


# ---------------------------------------------------------------------------
# Multi-case diverse generation
# ---------------------------------------------------------------------------

async def _build_diverse_test_cases(
    *,
    agent: Any,
    target_skill: str,
    target_skill_dir: Path,
    task: dict[str, Any],
    skill_manager: Any = None,
    num_cases: int = 3,
) -> list[dict[str, Any]]:
    """Generate diverse test cases via separate LLM calls, one per category."""
    num_cases = max(1, min(num_cases, len(_CASE_CATEGORIES)))
    categories = _CASE_CATEGORIES[:num_cases]
    cases: list[dict[str, Any]] = []
    previous_questions: list[str] = []

    for cat_name, cat_instruction in categories:
        try:
            case = await _build_flexible_runtime_skill_unit_case(
                agent=agent,
                target_skill=target_skill,
                target_skill_dir=target_skill_dir,
                task=task,
                skill_manager=skill_manager,
                category_instruction=cat_instruction,
                previous_questions=previous_questions if previous_questions else None,
            )
            if str(case.get("status") or "") == "ok":
                case["category"] = cat_name
                case["source"] = "generated"
                cases.append(case)
                # Track generated question for dedup
                gen_q = str(
                    (case.get("evidence") or {}).get("generated_question") or ""
                ).strip()
                if gen_q:
                    previous_questions.append(gen_q)
            # If generation failed for this category, skip it
        except Exception:
            # Skip failed category, don't fail the whole batch
            continue

    if not cases:
        # All categories failed — fall back to contract smoke test
        fallback = _build_contract_fallback_skill_unit_case(
            target_skill=target_skill,
            generation_attempts=[],
        )
        if str(fallback.get("status") or "") == "ok":
            fallback["category"] = "fallback_smoke_test"
            fallback["source"] = "generated"
            cases.append(fallback)

    return cases


# ---------------------------------------------------------------------------
# Regression case persistence
# ---------------------------------------------------------------------------

def _load_regression_cases(
    regression_dir: Path,
    max_cases: int = 2,
) -> list[dict[str, Any]]:
    """Load the most recent regression cases from JSONL."""
    regression_path = regression_dir / "regression_cases.jsonl"
    if not regression_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in regression_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict) and rec.get("question") and rec.get("expected_answer"):
            entries.append(rec)
    # Return most recent max_cases entries
    if max_cases > 0 and len(entries) > max_cases:
        entries = entries[-max_cases:]
    for entry in entries:
        entry["source"] = "regression_cache"
    return entries


def _save_passed_case_to_regression(
    regression_dir: Path,
    case: dict[str, Any],
    judge_result: dict[str, Any],
) -> None:
    """Append a passed case to the regression JSONL cache."""
    question = str(case.get("question") or "").strip()
    expected_answer = str(case.get("expected_answer") or "").strip()
    if not question or not expected_answer:
        return

    try:
        judge_score = float(judge_result.get("score_0_to_10", 0))
    except Exception:
        try:
            judge_score = float(judge_result.get("score", 0)) * 10.0
        except Exception:
            judge_score = 0.0

    entry = {
        "question": question,
        "expected_answer": expected_answer,
        "source": "regression_cache",
        "passed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge_score_0_to_10": round(judge_score, 2),
        "originating_task_key": str(
            (case.get("evidence") or {}).get("source_task_id") or ""
        ),
        "category": str(case.get("category") or ""),
    }
    append_jsonl(regression_dir / "regression_cases.jsonl", entry)


# ---------------------------------------------------------------------------
# Main gate function
# ---------------------------------------------------------------------------

async def run_optimize_stage_skill_unit_gate(
    *,
    agent: Any,
    target_skill: str,
    target_skill_dir: Path,
    task: dict[str, Any],
    judge_model: str,
    judge_retries: int,
    judge_pass_score_0_to_10: float,
    skill_manager: Any = None,
    # NEW params — all have defaults for backward compat
    num_generated_cases: int = 3,
    max_regression_cases: int = 2,
    regression_dir: Path | None = None,
    regression_hard_gate: bool = True,
    generated_pass_ratio: float = 0.5,
    correctness_weight: float = 0.7,
    skill_usage_weight: float = 0.2,
    trace_done_weight: float = 0.1,
) -> dict[str, Any]:
    """Run diverse unit tests after skill optimization to verify the skill works.

    Generates multiple test cases, executes them, optionally loads regression
    cases, and applies weighted scoring with aggregate pass/fail logic.
    """

    # --- 1. Load regression cases ---
    regression_cases: list[dict[str, Any]] = []
    if regression_dir is not None and max_regression_cases > 0:
        regression_cases = _load_regression_cases(
            regression_dir, max_cases=max_regression_cases
        )

    # --- 2. Generate diverse test cases ---
    generated_cases = await _build_diverse_test_cases(
        agent=agent,
        target_skill=target_skill,
        target_skill_dir=target_skill_dir,
        task=task,
        skill_manager=skill_manager,
        num_cases=max(1, int(num_generated_cases)),
    )

    # --- 3. Execute all cases sequentially ---
    all_case_results: list[dict[str, Any]] = []
    regression_results: list[dict[str, Any]] = []
    generated_results: list[dict[str, Any]] = []
    early_termination = False

    # 3a. Regression cases first
    for reg_case in regression_cases:
        case_result = await _execute_and_judge_single_case(
            agent=agent,
            target_skill=target_skill,
            case=reg_case,
            judge_model=judge_model,
            judge_retries=judge_retries,
            judge_pass_score_0_to_10=judge_pass_score_0_to_10,
            skill_manager=skill_manager,
            correctness_weight=correctness_weight,
            skill_usage_weight=skill_usage_weight,
            trace_done_weight=trace_done_weight,
        )
        case_result["case_source"] = "regression"
        case_result["case_category"] = str(reg_case.get("category") or "regression")
        regression_results.append(case_result)
        all_case_results.append(case_result)

        # Early termination on regression failure
        if regression_hard_gate and not case_result.get("passed"):
            early_termination = True
            break

    # 3b. Generated cases (skip if early termination)
    if not early_termination:
        for gen_case in generated_cases:
            case_result = await _execute_and_judge_single_case(
                agent=agent,
                target_skill=target_skill,
                case=gen_case,
                judge_model=judge_model,
                judge_retries=judge_retries,
                judge_pass_score_0_to_10=judge_pass_score_0_to_10,
                skill_manager=skill_manager,
                correctness_weight=correctness_weight,
                skill_usage_weight=skill_usage_weight,
                trace_done_weight=trace_done_weight,
            )
            case_result["case_source"] = "generated"
            case_result["case_category"] = str(gen_case.get("category") or "")
            generated_results.append(case_result)
            all_case_results.append(case_result)

    # --- 4. Compute aggregate result ---
    regression_all_passed = all(
        bool(r.get("passed")) for r in regression_results
    ) if regression_results else True

    generated_pass_count = sum(1 for r in generated_results if r.get("passed"))
    generated_total = len(generated_results)
    generated_ratio_met = (
        (generated_pass_count / generated_total) >= generated_pass_ratio
        if generated_total > 0
        else True  # No generated cases = vacuously true (shouldn't happen)
    )

    gate_passed = bool(regression_all_passed and generated_ratio_met)

    # --- 5. Save passed generated cases to regression cache ---
    if regression_dir is not None:
        regression_dir.mkdir(parents=True, exist_ok=True)
        for gen_result, gen_case in zip(generated_results, generated_cases):
            if gen_result.get("passed"):
                _save_passed_case_to_regression(
                    regression_dir=regression_dir,
                    case=gen_case,
                    judge_result=gen_result.get("judge") or {},
                )

    # --- 6. Build aggregate return dict ---
    # Compute aggregate weighted score
    all_scores = [r.get("weighted_score", 0.0) for r in all_case_results]
    avg_weighted_score = (
        sum(all_scores) / len(all_scores) if all_scores else 0.0
    )

    # For backward-compat: pick the first generated result's fields as "primary"
    primary = generated_results[0] if generated_results else (
        regression_results[0] if regression_results else {}
    )

    return {
        "status": "executed",
        "skill": target_skill,
        "passed": gate_passed,
        # Primary case fields (backward compat)
        "question": primary.get("question", ""),
        "expected_answer": primary.get("expected_answer", ""),
        "predicted_answer": primary.get("predicted_answer", ""),
        "evidence": primary.get("evidence"),
        "judge": primary.get("judge"),
        "trace_status": primary.get("trace_status"),
        "used_skills": primary.get("used_skills", []),
        "checks": primary.get("checks", {}),
        # Aggregate info
        "aggregate": {
            "gate_passed": gate_passed,
            "avg_weighted_score": round(avg_weighted_score, 4),
            "regression_count": len(regression_results),
            "regression_all_passed": regression_all_passed,
            "generated_count": generated_total,
            "generated_pass_count": generated_pass_count,
            "generated_pass_ratio": (
                round(generated_pass_count / generated_total, 4)
                if generated_total > 0 else 0.0
            ),
            "generated_pass_ratio_threshold": generated_pass_ratio,
            "early_termination": early_termination,
            "total_cases_executed": len(all_case_results),
        },
        # Detailed per-case results
        "case_results": all_case_results,
    }
