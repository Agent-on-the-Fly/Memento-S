"""UtilityTracker, ExperimentProfile, result recording, and summary metrics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ExperimentProfile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    create_on_miss: bool
    optimize_on_error: bool
    optimize_attempts: int
    feedback_retry: bool


EXPERIMENT_PROFILES: dict[str, ExperimentProfile] = {
    "read-only": ExperimentProfile(
        name="read-only",
        create_on_miss=False,
        optimize_on_error=False,
        optimize_attempts=0,
        feedback_retry=False,
    ),
    "read-write": ExperimentProfile(
        name="read-write",
        create_on_miss=True,
        optimize_on_error=False,
        optimize_attempts=0,
        feedback_retry=False,
    ),
    "read-write-optimize": ExperimentProfile(
        name="read-write-optimize",
        create_on_miss=True,
        optimize_on_error=True,
        optimize_attempts=0,
        feedback_retry=True,
    ),
}


def resolve_experiment_profile(raw: str) -> ExperimentProfile:
    key = str(raw or "").strip().lower()
    profile = EXPERIMENT_PROFILES.get(key)
    if profile is None:
        available = ", ".join(sorted(EXPERIMENT_PROFILES))
        raise ValueError(f"Unknown experiment '{raw}'. Available: {available}")
    return profile


# ---------------------------------------------------------------------------
# TaskRunResult
# ---------------------------------------------------------------------------

@dataclass
class TaskRunResult:
    answer: str
    used_skills: list[str]
    status: str  # "done" | "max_steps" | "error" | "fallback_*" | "partial_completed"
    trace: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# DEFAULT_JUDGE_MODEL — moved from llm_utils, read from env
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_MODEL: str = (os.getenv("HLE_JUDGE_MODEL") or "openai/o3-mini").strip()


# ---------------------------------------------------------------------------
# UtilityTracker
# ---------------------------------------------------------------------------

@dataclass
class UtilityTracker:
    threshold: float = 0.2
    min_samples: int = 3
    table: dict[str, dict[str, int]] = field(default_factory=dict)

    def get(self, skill_name: str) -> tuple[float, int]:
        entry = self.table.get(skill_name, {"success": 0, "failure": 0})
        total = int(entry.get("success", 0)) + int(entry.get("failure", 0))
        if total <= 0:
            return 0.5, 0
        return float(entry.get("success", 0)) / total, total

    def update(self, skill_name: str, success: bool) -> float:
        if skill_name not in self.table:
            self.table[skill_name] = {"success": 0, "failure": 0}
        key = "success" if success else "failure"
        self.table[skill_name][key] += 1
        utility, _ = self.get(skill_name)
        return utility

    def reset(self, skill_name: str) -> None:
        """Reset utility counters for a skill (e.g. after optimization)."""
        self.table[skill_name] = {"success": 0, "failure": 0}

    def should_discover(self, skill_name: str) -> bool:
        utility, n = self.get(skill_name)
        if n < self.min_samples:
            return False
        return utility < self.threshold

    def load(self, path: Path) -> None:
        if not path.exists():
            self.table = {}
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            self.table = {
                str(k): {
                    "success": int(v.get("success", 0)),
                    "failure": int(v.get("failure", 0)),
                }
                for k, v in data.items()
                if isinstance(v, dict)
            }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.table, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Summary / totals helpers
# ---------------------------------------------------------------------------

def empty_totals() -> dict[str, int]:
    return {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "optimized": 0,
        "discovered_alternative": 0,
        "fixed_after_feedback": 0,
        "skipped_exception": 0,
    }


def summary_metrics(totals: dict[str, int]) -> dict[str, float | int]:
    total = int(totals.get("total", 0))
    first_try_correct = int(totals.get("correct", 0))
    fixed_after_feedback = int(totals.get("fixed_after_feedback", 0))
    final_correct = min(total, first_try_correct + fixed_after_feedback)
    return {
        "first_try_correct": first_try_correct,
        "final_correct_after_feedback": final_correct,
        "accuracy_first_try": (first_try_correct / total) if total else 0.0,
        "accuracy_after_feedback": (final_correct / total) if total else 0.0,
    }


def error_judge_payload(err_msg: str, judge_model: str) -> dict[str, Any]:
    return {
        "is_correct": False,
        "score": 0.0,
        "rationale": err_msg,
        "extracted_final_answer": "",
        "confidence": "0",
        "judge_model": str(judge_model).strip() or DEFAULT_JUDGE_MODEL,
    }


# ---------------------------------------------------------------------------
# Result JSONL/JSON persistence
# ---------------------------------------------------------------------------

def load_result_records_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def load_result_records_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def load_result_records(
    *, results_jsonl_path: Path, results_json_path: Path
) -> list[dict[str, Any]]:
    records_from_json = load_result_records_json(results_json_path)
    if records_from_json:
        return records_from_json
    return load_result_records_jsonl(results_jsonl_path)


def append_result_record(
    *,
    record: dict[str, Any],
    results_jsonl_path: Path,
    results_json_path: Path,
    json_records_cache: list[dict[str, Any]],
) -> None:
    with results_jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    json_records_cache.append(record)
    results_json_path.write_text(
        json.dumps(json_records_cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Resume state loader
# ---------------------------------------------------------------------------

def _feedback_payloads_from_record(rec: dict[str, Any]) -> list[dict[str, Any]]:
    rounds = rec.get("feedback_rounds")
    out: list[dict[str, Any]] = []
    if isinstance(rounds, list):
        for item in rounds:
            if isinstance(item, dict):
                out.append(item)
    if out:
        return out
    feedback = rec.get("feedback")
    if isinstance(feedback, dict):
        return [feedback]
    return []


def load_resume_state(
    records: list[dict[str, Any]],
) -> tuple[set[str], dict[str, int]]:
    completed: set[str] = set()
    totals = empty_totals()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        key = str(rec.get("task_key") or "").strip()
        if key:
            completed.add(key)
        totals["total"] += 1
        is_correct = (
            bool((rec.get("judge") or {}).get("is_correct"))
            if isinstance(rec.get("judge"), dict)
            else False
        )
        if is_correct:
            totals["correct"] += 1
        else:
            totals["incorrect"] += 1

        trace_status = str(rec.get("trace_status") or "").strip().lower()
        feedback_payloads = _feedback_payloads_from_record(rec)
        retry_trace_statuses = [
            str(payload.get("retry_trace_status") or "").strip().lower()
            for payload in feedback_payloads
            if isinstance(payload, dict)
        ]
        if trace_status.startswith("error_") or any(
            x.startswith("error_") for x in retry_trace_statuses
        ):
            totals["skipped_exception"] += 1

        task_fixed = False
        for feedback in feedback_payloads:
            result = feedback.get("result")
            status_ok = isinstance(result, dict) and result.get("status") == "ok"
            kind = str(feedback.get("type") or "").strip()
            if kind == "optimize" and status_ok:
                totals["optimized"] += 1
            if kind == "discover" and status_ok:
                totals["discovered_alternative"] += 1
            if bool(feedback.get("fixed")):
                task_fixed = True
        if task_fixed:
            totals["fixed_after_feedback"] += 1
    return completed, totals


# ---------------------------------------------------------------------------
# Feedback round accuracy curve
# ---------------------------------------------------------------------------

def feedback_round_accuracy_curve(
    *, records: list[dict[str, Any]], task_feedback_max_rounds: int
) -> dict[str, Any]:
    valid_records = [rec for rec in records if isinstance(rec, dict)]
    total = len(valid_records)
    observed_max_round = 0
    task_round_state: list[tuple[bool, int | None, int]] = []

    def _to_int_or(default: int, raw: Any) -> int:
        try:
            return int(raw)
        except Exception:
            return int(default)

    for rec in valid_records:
        first_try_correct = bool(rec.get("first_try_correct"))
        payloads = _feedback_payloads_from_record(rec)
        max_round_for_task = 0
        fixed_round: int | None = 0 if first_try_correct else None
        for i, payload in enumerate(payloads, start=1):
            if not isinstance(payload, dict):
                continue
            round_num = _to_int_or(i, payload.get("round"))
            if round_num <= 0:
                round_num = i
            if round_num > max_round_for_task:
                max_round_for_task = round_num
            if bool(payload.get("fixed")):
                if fixed_round is None or round_num < fixed_round:
                    fixed_round = round_num
        observed_max_round = max(observed_max_round, max_round_for_task)
        task_round_state.append((first_try_correct, fixed_round, max_round_for_task))

    max_round_evaluated = (
        int(task_feedback_max_rounds)
        if int(task_feedback_max_rounds) > 0
        else int(observed_max_round)
    )
    rows: list[dict[str, Any]] = []
    if total <= 0 or max_round_evaluated <= 0:
        return {
            "total_tasks": total,
            "observed_max_round": int(observed_max_round),
            "max_round_evaluated": int(max_round_evaluated),
            "rows": rows,
        }

    for n in range(1, max_round_evaluated + 1):
        correct_after_n = 0
        fixed_new_at_n = 0
        tasks_reached_n = 0
        for first_try_correct, fixed_round, max_round_for_task in task_round_state:
            if max_round_for_task >= n:
                tasks_reached_n += 1
            if first_try_correct:
                correct_after_n += 1
                continue
            if fixed_round is not None and fixed_round <= n:
                correct_after_n += 1
                if fixed_round == n:
                    fixed_new_at_n += 1
        rows.append(
            {
                "round": n,
                "tasks_reached_round": tasks_reached_n,
                "correct_after_round": correct_after_n,
                "accuracy_after_round": (correct_after_n / total) if total else 0.0,
                "newly_fixed_at_round": fixed_new_at_n,
            }
        )
    return {
        "total_tasks": total,
        "observed_max_round": int(observed_max_round),
        "max_round_evaluated": int(max_round_evaluated),
        "rows": rows,
    }
