"""CLI entry point for the evolve loop — ``memento-evolve``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root (opc_memento_s-main/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.evolve.dataset import (
    build_user_text,
    iter_tasks,
    load_hle_rows,
    load_learning_tips,
    prepare_task_attachments,
    task_key,
)
from core.evolve.feedback import (
    build_skill_judgement_after_answer_failure,
    collect_used_skills_for_feedback,
    select_target_skill_for_feedback,
    summarize_tips,
    write_tip_for_failure,
)
from core.evolve.judge import judge_answer
from core.evolve.tracker import DEFAULT_JUDGE_MODEL
from core.evolve.optimizer import (
    discover_alternative_skill,
    optimize_skill_with_trajectory,
    restore_skill_folder_from_backup,
    skill_names_from_dir,
)
from core.evolve.runner import execute_task_once
from core.evolve.text_utils import normalize_skill_name
from core.evolve.tracker import (
    UtilityTracker,
    append_jsonl,
    append_result_record,
    empty_totals,
    error_judge_payload,
    feedback_round_accuracy_curve,
    load_result_records,
    load_resume_state,
    resolve_experiment_profile,
    summary_metrics,
)
from core.evolve.unit_test_gate import run_optimize_stage_skill_unit_gate


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_RUNTIME_FIXED_SKILL_NORMS: set[str] = set()


def _run_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / f"hle_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _latest_resumable_run(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [
        p
        for p in root.glob("hle_*")
        if p.is_dir() and ((p / "results.jsonl").exists() or (p / "results.json").exists())
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _is_llm_budget_exceeded_error(exc: BaseException) -> bool:
    needle = "llm call budget exceeded for current turn"
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if needle in str(cur).lower():
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _should_skip_task_exception(exc: BaseException, args: argparse.Namespace) -> bool:
    if _is_llm_budget_exceeded_error(exc):
        return True
    return bool(getattr(args, "skip_task_exceptions", False))


def _format_exception_message(exc: BaseException) -> str:
    msg = str(exc).strip()
    if msg:
        return f"{type(exc).__name__}: {msg}"
    return type(exc).__name__


def _exception_status(exc: BaseException, *, stage: str = "") -> str:
    suffix = "llm_budget_exceeded" if _is_llm_budget_exceeded_error(exc) else "exception"
    if stage:
        return f"error_{stage}_{suffix}"
    return f"error_{suffix}"


def _resolve_skills_root(raw: str | Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> None:
    # --- Paths ---
    data_path = Path(args.data).expanduser().resolve()
    run_root = Path(args.run_dir).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    base_skills_root = _resolve_skills_root(args.local_skills_dir)
    if not base_skills_root.exists():
        raise FileNotFoundError(f"Local skills directory not found: {base_skills_root}")

    skill_extra_root = _resolve_skills_root(args.skill_extra_dir)
    skill_extra_root.mkdir(parents=True, exist_ok=True)

    tip_file = Path(args.tip_file).expanduser()
    if not tip_file.is_absolute():
        tip_file = (PROJECT_ROOT / tip_file).resolve()
    else:
        tip_file = tip_file.resolve()
    tip_write_enabled = not bool(args.disable_tip_write)

    # --- Experiment profile ---
    profile = resolve_experiment_profile(args.experiment)
    create_on_miss_enabled = bool(profile.create_on_miss)
    optimize_on_error_enabled = bool(profile.optimize_on_error)
    task_feedback_max_rounds = int(args.optimize_attempts)
    if task_feedback_max_rounds == -1:
        task_feedback_max_rounds = 0
    if not optimize_on_error_enabled:
        task_feedback_max_rounds = 0
    feedback_retry_enabled = bool(profile.feedback_retry)
    if bool(args.disable_feedback_retry):
        feedback_retry_enabled = False

    print(
        "[experiment] "
        f"{profile.name} | "
        f"create_on_miss={create_on_miss_enabled} "
        f"optimize_on_error={optimize_on_error_enabled} "
        f"task_feedback_rounds={('unlimited' if task_feedback_max_rounds <= 0 else task_feedback_max_rounds)} "
        f"feedback_retry={feedback_retry_enabled}"
    )
    print(f"[base-skills] {base_skills_root}")
    print(f"[skill-extra] {skill_extra_root}")
    print(f"[judge] model={str(args.judge_model).strip() or DEFAULT_JUDGE_MODEL}")
    print(f"[tips] file={tip_file} enabled={tip_write_enabled}")

    # --- Resume ---
    if args.resume and args.resume_auto:
        raise ValueError("Use either --resume or --resume-auto, not both.")

    resumed = False
    if args.resume:
        run_dir = Path(args.resume).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        resumed = (run_dir / "results.jsonl").exists() or (run_dir / "results.json").exists()
        print(f"[resume] using run dir: {run_dir}")
    elif args.resume_auto:
        latest = _latest_resumable_run(run_root)
        if latest is not None:
            run_dir = latest
            resumed = True
            print(f"[resume] auto from latest run: {run_dir}")
        else:
            run_dir = _run_dir(run_root)
            print(f"[resume] no resumable run found, starting fresh: {run_dir}")
    else:
        run_dir = _run_dir(run_root)

    optimize_unit_test_gate_enabled = bool(args.optimize_unit_test_gate)
    optimize_unit_test_pass_score = max(0.0, min(10.0, float(args.optimize_unit_test_pass_score)))

    results_path = run_dir / "results.jsonl"
    results_json_path = run_dir / "results.json"
    summary_path = run_dir / "summary.json"
    unit_test_gate_log_path = run_dir / "optimize_stage_unit_tests.jsonl"
    optimize_round_io_log_path = run_dir / "optimize_round_io.jsonl"
    utility_path = (
        Path(args.utility_table_path).expanduser().resolve()
        if args.utility_table_path
        else (run_dir / "utility_table.json")
    )

    if optimize_unit_test_gate_enabled:
        print(
            "[optimize-unit-test-gate] "
            f"enabled=True "
            f"pass_score_0_to_10>{optimize_unit_test_pass_score:g}"
        )
    else:
        print("[optimize-unit-test-gate] enabled=False")

    # --- Utility tracker ---
    utility = UtilityTracker(
        threshold=float(args.utility_threshold),
        min_samples=int(args.utility_min_samples),
    )
    utility.load(utility_path)

    # --- Load dataset ---
    tasks = load_hle_rows(data_path)
    if not tasks:
        raise RuntimeError(f"No valid HLE rows found in dataset: {data_path}")
    print(f"[dataset] loaded {len(tasks)} tasks from {data_path}")

    # --- Learning tips ---
    learning_tips_path: Path | None = None
    if args.learning_tips:
        learning_tips_path = Path(args.learning_tips).expanduser().resolve()
        if not learning_tips_path.exists():
            raise FileNotFoundError(f"Learning tips file not found: {learning_tips_path}")
    learning_tips = load_learning_tips(learning_tips_path, max_chars=int(args.learning_tips_max_chars))
    if learning_tips:
        print(f"[learning-tips] loaded from {learning_tips_path} (chars={len(learning_tips)})")

    # --- Build MementoSAgent (target project interface) ---
    from core.agent import MementoSAgent
    from core.config import g_settings
    from core.llm import LLM

    workspace = g_settings.workspace_path
    workspace.mkdir(parents=True, exist_ok=True)

    llm = LLM()
    agent = MementoSAgent(workspace=workspace, llm=llm)
    print(f"[agent] MementoSAgent started (model={llm.default_model}, workspace={workspace})")

    # Get skill_manager from agent for routing
    skill_manager = agent.skill_manager

    # --- Fixed skills ---
    fixed_dir = (
        Path(args.fixed_skills_dir).expanduser().resolve()
        if args.fixed_skills_dir
        else base_skills_root
    )
    fixed_names = skill_names_from_dir(fixed_dir)
    global _RUNTIME_FIXED_SKILL_NORMS
    _RUNTIME_FIXED_SKILL_NORMS = {normalize_skill_name(name) for name in fixed_names}
    print(f"[fixed-skills] scanned={len(fixed_names)} dir={fixed_dir}")

    # --- Resume records ---
    result_records_cache = load_result_records(
        results_jsonl_path=results_path,
        results_json_path=results_json_path,
    )
    if result_records_cache and not results_json_path.exists():
        results_json_path.write_text(
            json.dumps(result_records_cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    totals = empty_totals()
    completed_keys: set[str] = set()
    if resumed:
        completed_keys, resumed_totals = load_resume_state(result_records_cache)
        totals.update(resumed_totals)
        current_acc = (totals["correct"] / totals["total"]) if totals["total"] else 0.0
        print(f"[resume] loaded {len(completed_keys)} completed task(s), accuracy={current_acc:.3f}")

    # =======================================================================
    # Main loop
    # =======================================================================
    try:
        for idx, task in iter_tasks(tasks, start=args.start, end=args.end, limit=args.max_tasks):
            key = task_key(idx, task)
            if key in completed_keys:
                print(f"[resume-skip] task {idx} key={key} already completed")
                continue

            totals["total"] += 1
            print(f"\n[task {idx}] key={key} question={str(task.get('question') or '')[:120]}")

            attachment_bundle = prepare_task_attachments(
                task=task,
                data_path=data_path,
                task_key=key,
                run_dir=run_dir,
                project_root=PROJECT_ROOT,
            )
            prepared_refs = list(attachment_bundle.get("attachment_refs") or [])
            prepared_image_ref = str(attachment_bundle.get("image_ref") or "")

            user_text = build_user_text(
                task,
                data_path=data_path,
                project_root=PROJECT_ROOT,
                learning_tips=learning_tips,
                read_only_mode=(profile.name == "read-only"),
                attachment_refs_override=prepared_refs,
                image_ref_override=prepared_image_ref,
            )

            # --- Snapshot extra-only skills before execution ---
            _evolve_skills_before = {
                d.name for d in skill_extra_root.iterdir()
                if d.is_dir() and (d / "SKILL.md").exists()
                and not (base_skills_root / d.name / "SKILL.md").exists()
            } if skill_extra_root.exists() else set()

            # --- First attempt ---
            try:
                first = await execute_task_once(
                    agent,
                    user_text=user_text,

                    create_on_miss_enabled=create_on_miss_enabled,
                    skill_manager=skill_manager,
                )
            except Exception as exc:
                if not _should_skip_task_exception(exc, args):
                    raise
                err_msg = _format_exception_message(exc)
                from core.evolve.tracker import TaskRunResult
                first = TaskRunResult(
                    answer=f"ERROR: {err_msg}",
                    used_skills=[],
                    status=_exception_status(exc),
                    trace=[{"step_num": 0, "status": "error", "skill_name": "", "result_preview": err_msg}],
                )
                j = error_judge_payload(err_msg, str(args.judge_model))
                is_correct = False
                totals["incorrect"] += 1
                totals["skipped_exception"] += 1
            else:
                try:
                    j = await judge_answer(
                        question=str(task.get("question") or ""),
                        answer_type=str(task.get("answer_type") or ""),
                        model_answer=str(first.answer or "").strip(),
                        expected_answer=str(task.get("answer") or ""),
                        judge_model=str(args.judge_model),
                        retries=max(1, int(args.judge_retries)),
                    )
                except Exception as exc:
                    if not _should_skip_task_exception(exc, args):
                        raise
                    err_msg = _format_exception_message(exc)
                    j = error_judge_payload(err_msg, str(args.judge_model))
                    is_correct = False
                    totals["incorrect"] += 1
                    totals["skipped_exception"] += 1
                else:
                    is_correct = bool(j.get("is_correct"))
                    if is_correct:
                        totals["correct"] += 1
                    else:
                        totals["incorrect"] += 1

            # --- Utility tracking (evolve-created skills only) ---
            # Only track skills that exist in skill_extra_root but NOT in
            # base_skills_root — i.e. skills the evolve loop itself created
            # or fetched.  Pre-installed / builtin skills are excluded.
            #
            # Snapshot diff catches skills created during this task via
            # create_on_miss (the agent won't read_skill them so they
            # don't appear in used_skills).
            _evolve_skills_after = {
                d.name for d in skill_extra_root.iterdir()
                if d.is_dir() and (d / "SKILL.md").exists()
                and not (base_skills_root / d.name / "SKILL.md").exists()
            } if skill_extra_root.exists() else set()
            _newly_created_skills = _evolve_skills_after - _evolve_skills_before

            utility_updates: dict[str, dict[str, float | int]] = {}
            _extra_used_in_first = set()
            for sk in first.used_skills:
                if (skill_extra_root / sk / "SKILL.md").exists() \
                        and not (base_skills_root / sk / "SKILL.md").exists():
                    _extra_used_in_first.add(sk)
            _extra_used_in_first |= _newly_created_skills

            for sk in _extra_used_in_first:
                old_u, old_n = utility.get(sk)
                new_u = utility.update(sk, is_correct)
                _, new_n = utility.get(sk)
                utility_updates[sk] = {
                    "utility_before": old_u, "utility_after": new_u,
                    "n_before": old_n, "n_after": new_n,
                }

            record: dict[str, Any] = {
                "task_key": key,
                "task_id": task.get("id"),
                "idx": idx,
                "level": task.get("level"),
                "category": task.get("category"),
                "subject": task.get("raw_subject"),
                "answer_type": task.get("answer_type"),
                "question": task.get("question"),
                "gold_answer": task.get("answer"),
                "predicted_answer": first.answer,
                "used_skills": first.used_skills,
                "trace_status": first.status,
                "trace": first.trace,
                "judge": j,
                "first_try_correct": is_correct,
                "utility_updates": utility_updates,
            }

            acc = (totals["correct"] / totals["total"]) if totals["total"] else 0.0
            print(
                f"[task {idx}] correct={is_correct} skills={first.used_skills} "
                f"status={first.status} acc={acc:.3f} ({totals['correct']}/{totals['total']})"
            )

            # --- Feedback loop ---
            feedback_rounds: list[dict[str, Any]] = []
            current_result = first
            current_judge = j
            current_is_correct = is_correct
            feedback_round = 0

            while True:
                should_run_feedback = (
                    feedback_retry_enabled
                    and (not current_is_correct)
                    and (
                        task_feedback_max_rounds <= 0
                        or feedback_round < task_feedback_max_rounds
                    )
                )
                if not should_run_feedback:
                    break

                feedback_round += 1
                feedback_result: dict[str, Any] | None = None
                feedback_type: str | None = None
                target_skill_resolved = ""
                target_skill: str | None = None
                target_selector: dict[str, Any] = {"status": "no_skills_in_trace"}
                target_in_extra = False
                is_fixed = False
                skill_judgement: dict[str, Any] = {
                    "status": "no_skill_used_in_trace",
                    "reason": "cannot_judge_skill_without_used_skills",
                }

                if current_result.used_skills:
                    resolved_used, resolved_used_norms, extra_used = collect_used_skills_for_feedback(
                        used_skills=current_result.used_skills,
                        base_skills_root=base_skills_root,
                        extra_skills_root=skill_extra_root,
                    )

                    if extra_used:
                        target_skill, target_selector = await select_target_skill_for_feedback(
                            used_skills=extra_used,
                            trace=current_result.trace,
                            judge=current_judge,
                            base_skills_root=base_skills_root,
                            extra_skills_root=skill_extra_root,
                        )
                        target_selector = {
                            **dict(target_selector or {}),
                            "selection_scope": "extra_only",
                            "extra_used_skills": extra_used,
                            "used_skills": resolved_used,
                        }

                        if target_skill:
                            target_skill_resolved = target_skill
                            target_norm = normalize_skill_name(target_skill_resolved)
                            if target_norm in resolved_used_norms:
                                target_in_extra = (skill_extra_root / target_skill_resolved / "SKILL.md").exists()
                                is_fixed = target_norm in _RUNTIME_FIXED_SKILL_NORMS
                            else:
                                target_skill = None
                                target_skill_resolved = ""
                                target_in_extra = False

                    skill_judgement = build_skill_judgement_after_answer_failure(
                        trace=current_result.trace,
                        judge=current_judge,
                        target_skill=target_skill_resolved or target_skill,
                        target_selector=target_selector,
                    )

                # Decide optimization strategy
                if target_skill_resolved and target_in_extra and (not is_fixed):
                    tip_result = await write_tip_for_failure(
                        tip_write_enabled=tip_write_enabled,
                        tip_path=tip_file,
                        task=task,
                        judge=current_judge,
                        trace=current_result.trace,
                        no_update_reason=f"edit_skill_extra:{target_skill_resolved}",
                    )
                    if utility.should_discover(target_skill_resolved):
                        feedback_type = "discover"
                        utility_val, _ = utility.get(target_skill_resolved)
                        feedback_result = await discover_alternative_skill(
                            skill_name=target_skill_resolved,
                            base_skills_root=base_skills_root,
                            extra_skills_root=skill_extra_root,
                            utility_value=utility_val,
                            task=task,
                            user_text=user_text,
                            model_answer=current_result.answer,
                            trace=current_result.trace,
                            skill_judgement=skill_judgement,
                            selection_context=target_selector,
                            include_judge_rationale=bool(args.evolve_include_judge_rationale),
                            agent=agent,
                        )
                        # Handle agent-based discovery
                        if feedback_result.get("status") == "discover_via_agent":
                            discover_prompt = feedback_result.get("prompt", "")
                            try:
                                discover_run = await execute_task_once(
                                    agent,
                                    user_text=discover_prompt,
                                    create_on_miss_enabled=True,
                                    skill_manager=skill_manager,
                
                                )
                                planned_skill = target_skill_resolved
                                if (skill_extra_root / planned_skill / "SKILL.md").exists():
                                    feedback_result = {
                                        "status": "ok",
                                        "skill": planned_skill,
                                        "mode": "agent_discover",
                                        "result_preview": str(discover_run.answer or "")[:500],
                                    }
                                else:
                                    feedback_result = {
                                        "status": "discover_failed",
                                        "skill": planned_skill,
                                        "error": "agent did not produce SKILL.md",
                                    }
                            except Exception as exc:
                                feedback_result = {
                                    "status": "discover_failed",
                                    "skill": target_skill_resolved,
                                    "error": str(exc),
                                }
                        if feedback_result.get("status") == "ok":
                            totals["discovered_alternative"] += 1
                            utility.reset(target_skill_resolved)
                    else:
                        feedback_type = "optimize"
                        feedback_result = await optimize_skill_with_trajectory(
                            skill_name=target_skill_resolved,
                            base_skills_root=base_skills_root,
                            extra_skills_root=skill_extra_root,
                            task=task,
                            user_text=user_text,
                            model_answer=current_result.answer,
                            trace=current_result.trace,
                            skill_judgement=skill_judgement,
                            selection_context=target_selector,
                            include_judge_rationale=bool(args.evolve_include_judge_rationale),
                        )
                        if feedback_result.get("status") == "ok":
                            totals["optimized"] += 1
                            utility.reset(target_skill_resolved)
                    if isinstance(feedback_result, dict):
                        feedback_result.setdefault("tip", tip_result)
                else:
                    feedback_type = "tips"
                    tip_reason = "tip_only_no_skill_used"
                    if target_skill_resolved:
                        tip_reason = "tip_only_after_atomic_failure" if is_fixed else "tip_only_non_extra_skill"
                    elif current_result.used_skills:
                        tip_reason = "tip_only_no_clear_extra_target"
                    tip_result = await write_tip_for_failure(
                        tip_write_enabled=tip_write_enabled,
                        tip_path=tip_file,
                        task=task,
                        judge=current_judge,
                        trace=current_result.trace,
                        no_update_reason=tip_reason,
                    )
                    feedback_result = {"status": "tip_only", "reason": tip_reason, "tip": tip_result}

                # --- Unit test gate ---
                if (
                    feedback_result
                    and str(feedback_result.get("status") or "").strip().lower() == "ok"
                    and optimize_unit_test_gate_enabled
                    and target_skill_resolved
                    and target_in_extra
                    and (not is_fixed)
                ):
                    gate_skill = str(feedback_result.get("skill") or target_skill_resolved).strip()
                    gate_skill_dir = skill_extra_root / gate_skill
                    if not (gate_skill_dir / "SKILL.md").exists():
                        gate_skill = target_skill_resolved
                        gate_skill_dir = skill_extra_root / gate_skill
                    gate_result = await run_optimize_stage_skill_unit_gate(
                        agent=agent,
                        target_skill=gate_skill,
                        target_skill_dir=gate_skill_dir,
                        task=task,
                        judge_model=str(args.judge_model),
                        judge_retries=max(1, int(args.judge_retries)),
                        judge_pass_score_0_to_10=optimize_unit_test_pass_score,
                        skill_manager=skill_manager,
                        num_generated_cases=int(args.unit_test_num_generated),
                        max_regression_cases=int(args.unit_test_max_regression),
                        regression_dir=skill_extra_root / gate_skill / ".evals",
                        regression_hard_gate=bool(args.unit_test_regression_hard_gate),
                        generated_pass_ratio=float(args.unit_test_generated_pass_ratio),
                        correctness_weight=float(args.unit_test_correctness_weight),
                    )
                    feedback_result = dict(feedback_result)
                    feedback_result["unit_test_gate"] = gate_result
                    append_jsonl(
                        unit_test_gate_log_path,
                        {
                            "time": datetime.now().isoformat(timespec="seconds"),
                            "idx": idx, "task_key": key,
                            "feedback_round": feedback_round,
                            "feedback_type": feedback_type,
                            "skill": gate_skill,
                            "gate": gate_result,
                        },
                    )
                    gate_passed = bool(gate_result.get("passed"))
                    if not gate_passed:
                        backup_raw = str(feedback_result.get("backup") or "").strip()
                        rollback_result: dict[str, Any] = {"status": "skipped_no_backup"}
                        if backup_raw:
                            rollback_result = restore_skill_folder_from_backup(
                                skill_dir=gate_skill_dir,
                                backup_path=Path(backup_raw).expanduser().resolve(),
                            )
                        feedback_result["status"] = "optimize_rejected_by_unit_test"
                        feedback_result["rollback"] = rollback_result

                # --- Retry after feedback ---
                feedback_status = (
                    str(feedback_result.get("status") or "").strip().lower()
                    if isinstance(feedback_result, dict)
                    else ""
                )
                feedback_kind = feedback_type or "optimize"
                should_retry = bool(
                    isinstance(feedback_result, dict)
                    and feedback_status in {"ok", "tip_only"}
                )

                if should_retry:
                    try:
                        retry = await execute_task_once(
                            agent,
                            user_text=user_text,
        
                            create_on_miss_enabled=create_on_miss_enabled,
                            skill_manager=skill_manager,
                        )
                        retry_judge = await judge_answer(
                            question=str(task.get("question") or ""),
                            answer_type=str(task.get("answer_type") or ""),
                            model_answer=str(retry.answer or "").strip(),
                            expected_answer=str(task.get("answer") or ""),
                            judge_model=str(args.judge_model),
                            retries=max(1, int(args.judge_retries)),
                        )
                        fixed = bool(retry_judge.get("is_correct"))
                        if fixed:
                            totals["fixed_after_feedback"] += 1
                        # Update utility only for evolve-created skills
                        if (
                            target_skill_resolved
                            and (skill_extra_root / target_skill_resolved / "SKILL.md").exists()
                            and not (base_skills_root / target_skill_resolved / "SKILL.md").exists()
                        ):
                            utility.update(target_skill_resolved, fixed)
                        feedback_payload = {
                            "round": feedback_round,
                            "type": feedback_kind,
                            "result": feedback_result,
                            "retry_answer": retry.answer,
                            "retry_judge": retry_judge,
                            "retry_used_skills": retry.used_skills,
                            "retry_trace_status": retry.status,
                            "fixed": fixed,
                        }
                        feedback_rounds.append(feedback_payload)
                        current_result = retry
                        current_judge = retry_judge
                        current_is_correct = fixed
                        print(
                            f"[task {idx}] feedback_round={feedback_round} "
                            f"type={feedback_kind} status={feedback_status} fixed={fixed}"
                        )
                    except Exception as exc:
                        if not _should_skip_task_exception(exc, args):
                            raise
                        err_msg = _format_exception_message(exc)
                        totals["skipped_exception"] += 1
                        feedback_payload = {
                            "round": feedback_round,
                            "type": feedback_kind,
                            "result": feedback_result,
                            "retry_answer": f"ERROR: {err_msg}",
                            "retry_judge": error_judge_payload(err_msg, str(args.judge_model)),
                            "retry_trace_status": _exception_status(exc, stage="retry"),
                            "fixed": False,
                        }
                        feedback_rounds.append(feedback_payload)
                        print(
                            f"[task {idx}] feedback_round={feedback_round} "
                            f"type={feedback_kind} status=retry_exception fixed=False"
                        )
                        break
                else:
                    feedback_payload = {
                        "round": feedback_round,
                        "type": feedback_kind,
                        "result": feedback_result,
                        "fixed": False,
                    }
                    feedback_rounds.append(feedback_payload)
                    append_jsonl(
                        optimize_round_io_log_path,
                        {
                            "time": datetime.now().isoformat(timespec="seconds"),
                            "idx": idx, "task_key": key,
                            "feedback_round": feedback_round,
                            "feedback_type": feedback_kind,
                            "feedback_result": feedback_result,
                        },
                    )
                    break

            # --- Record result ---
            if feedback_rounds:
                record["feedback_rounds"] = feedback_rounds
                record["feedback"] = feedback_rounds[-1]

            append_result_record(
                record=record,
                results_jsonl_path=results_path,
                results_json_path=results_json_path,
                json_records_cache=result_records_cache,
            )
            utility.save(utility_path)

            # --- Periodic tip summarization ---
            tip_interval = int(args.tip_summarize_interval)
            if tip_write_enabled and tip_interval > 0 and totals["total"] % tip_interval == 0:
                tip_summary = await summarize_tips(tip_path=tip_file)
                if tip_summary.get("status") == "summarized":
                    print(
                        f"[tips] summarized: {tip_summary['chars_before']}"
                        f" -> {tip_summary['chars_after']} chars"
                    )

    finally:
        # --- Summary ---
        metrics = summary_metrics(totals)
        curve = feedback_round_accuracy_curve(
            records=result_records_cache,
            task_feedback_max_rounds=task_feedback_max_rounds,
        )
        summary = {
            "experiment": profile.name,
            "totals": totals,
            "metrics": metrics,
            "feedback_curve": curve,
            "data_path": str(data_path),
            "run_dir": str(run_dir),
            "base_skills_root": str(base_skills_root),
            "skill_extra_root": str(skill_extra_root),
            "judge_model": str(args.judge_model).strip() or DEFAULT_JUDGE_MODEL,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[summary] {json.dumps(metrics, indent=2)}")
        print(f"[summary] written to {summary_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Memento-S Evolve Loop — self-evolution benchmark runner"
    )
    parser.add_argument(
        "--experiment",
        default="read-only",
        help="Experiment profile: read-only, read-write, read-write-optimize",
    )
    parser.add_argument("--data", required=True, help="Dataset path (.json/.jsonl/.parquet)")
    parser.add_argument("--run-dir", default="workspace/evolve/runs", help="Output directory root")
    parser.add_argument("--resume", default=None, help="Resume from a specific run directory")
    parser.add_argument("--resume-auto", action="store_true", help="Auto-resume from latest run")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks to process")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index (inclusive)")

    parser.add_argument("--optimize-attempts", type=int, default=0, help="Feedback rounds (0=unlimited)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="LLM model for judge")
    parser.add_argument("--judge-retries", type=int, default=3, help="Judge retry attempts")
    parser.add_argument(
        "--local-skills-dir", default="builtin/skills", help="Base local skills directory"
    )
    parser.add_argument(
        "--skill-extra-dir", default="workspace/skills", help="Extra/evolved skills directory (matches agent skill save path)"
    )
    parser.add_argument(
        "--fixed-skills-dir", default=None, help="Immutable skills dir (never optimized)"
    )
    parser.add_argument("--utility-threshold", type=float, default=0.2, help="Discover threshold")
    parser.add_argument("--utility-min-samples", type=int, default=3, help="Min samples for utility")
    parser.add_argument("--utility-table-path", default=None, help="Explicit utility table path")
    parser.add_argument("--learning-tips", default=None, help="Path to TIP.md for injection")
    parser.add_argument(
        "--learning-tips-max-chars", type=int, default=4000, help="Max chars for learning tips"
    )
    parser.add_argument("--tip-file", default="workspace/evolve/TIP.md", help="Path to write generic tips")
    parser.add_argument("--disable-tip-write", action="store_true", help="Disable tip writing")
    parser.add_argument(
        "--tip-summarize-interval", type=int, default=10,
        help="Summarize TIP.md every N tasks (0=disable)",
    )
    parser.add_argument("--disable-feedback-retry", action="store_true", help="Disable feedback retry")
    parser.add_argument(
        "--evolve-include-judge-rationale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include judge rationale in optimization feedback",
    )
    parser.add_argument(
        "--optimize-unit-test-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable post-optimize unit test gate",
    )
    parser.add_argument(
        "--optimize-unit-test-pass-score", type=float, default=5.0, help="Gate pass score (0-10)"
    )
    parser.add_argument(
        "--unit-test-num-generated", type=int, default=3,
        help="Number of diverse generated test cases for unit test gate",
    )
    parser.add_argument(
        "--unit-test-max-regression", type=int, default=2,
        help="Max regression cases to load per skill gate",
    )
    parser.add_argument(
        "--unit-test-regression-hard-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail gate immediately if any regression case fails",
    )
    parser.add_argument(
        "--unit-test-generated-pass-ratio", type=float, default=0.5,
        help="Min ratio of generated cases that must pass (0.0-1.0)",
    )
    parser.add_argument(
        "--unit-test-correctness-weight", type=float, default=0.7,
        help="Correctness weight in weighted scoring (0.0-1.0)",
    )
    parser.add_argument(
        "--skip-task-exceptions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip task exceptions instead of aborting",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"\nFatal error: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
