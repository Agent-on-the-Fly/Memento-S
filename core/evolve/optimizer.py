"""Skill optimization — rewrite, apply, restore, discover alternative."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .feedback import (
    append_target_selection_feedback,
    build_failure_feedback_blob,
)
from .skill_helpers import read_skill_md_text, resolve_local_skill_name
from .text_utils import json_fragment, normalize_skill_name, strip_markdown_fence


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKILL_API_POLICY = (
    "Skill API policy:\n"
    "- If this skill needs external LLM/API access, use generic LLM env settings.\n"
    "- Read API key from LLM_API_KEY, model from LLM_MODEL, base URL from LLM_BASE_URL.\n"
    "- Never hardcode API keys or secrets into SKILL.md/scripts."
)


# ---------------------------------------------------------------------------
# Skill folder helpers
# ---------------------------------------------------------------------------

def _normalize_relative_update_path(raw_path: Any) -> str | None:
    path_text = str(raw_path or "").strip().replace("\\", "/")
    if not path_text:
        return None
    if path_text.startswith("/"):
        return None
    parts = [part for part in path_text.split("/") if part]
    if not parts:
        return None
    if any(part in {".", ".."} for part in parts):
        return None
    return "/".join(parts)


def _collect_skill_folder_text_files(
    *,
    skill_dir: Path,
    max_files: int = 40,
    max_chars_per_file: int = 12000,
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    if not skill_dir.is_dir():
        return files
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(skill_dir).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            continue
        truncated = False
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file] + "\n...[truncated]..."
            truncated = True
        files.append({"path": rel, "content": content, "truncated": truncated})
        if len(files) >= max_files:
            break
    return files


def skill_names_from_dir(skills_dir: Path) -> list[str]:
    if not skills_dir.is_dir():
        return []
    names: list[str] = []
    for child in skills_dir.iterdir():
        if child.is_dir() and (child / "SKILL.md").exists() and not child.name.startswith("."):
            names.append(child.name)
    return sorted(set(names))


# ---------------------------------------------------------------------------
# Rewrite skill folder with LLM feedback
# ---------------------------------------------------------------------------

async def rewrite_skill_folder_with_feedback(
    *,
    skill_name: str,
    skill_dir: Path,
    feedback: str,
) -> tuple[list[dict[str, str]] | None, str, dict[str, Any]]:
    """Ask LLM to rewrite skill files based on failure feedback.

    Returns ``(updates_list | None, status_str, artifact_dict)``.
    """
    from core.evolve import get_evolve_llm

    file_context = _collect_skill_folder_text_files(skill_dir=skill_dir)
    artifact: dict[str, Any] = {
        "skill": skill_name,
        "skill_dir": str(skill_dir),
        "failure_feedback": feedback,
    }
    if not file_context:
        artifact["status"] = "failed"
        artifact["reason"] = "no_text_files_found_in_skill_dir"
        return None, "no_text_files_found_in_skill_dir", artifact

    artifact["skill_files_snapshot"] = file_context

    system_prompt = (
        "You are maintaining a reusable skill folder. "
        "Given failure feedback, propose concrete file updates for this one skill folder. "
        "You may update SKILL.md and any helper files under this skill directory. "
        'Return JSON only with this schema:\n'
        "{\n"
        '  "updates": [\n'
        '    {"path": "relative/path.ext", "content": "full file content"}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Use relative paths only; never absolute paths.\n"
        "- Do not use .. path segments.\n"
        "- Include full content for every file you update.\n"
        "- Keep updates deterministic and generally reusable.\n"
        "- Avoid task-specific hardcoding."
    )
    user_prompt = (
        f"Skill name: {skill_name}\n\n"
        "Current skill folder files (JSON array):\n"
        "<skill_files>\n"
        f"{json.dumps(file_context, ensure_ascii=False, indent=2)}\n"
        "</skill_files>\n\n"
        "Failure feedback (complete trajectory context):\n"
        "<failure_feedback>\n"
        f"{feedback}\n"
        "</failure_feedback>\n\n"
        "Update requirements:\n"
        "- Keep behavior focused and deterministic.\n"
        "- Add concrete guardrails/checklists for observed failure modes.\n"
        "- Keep content concise and reusable.\n"
        "- Return JSON only."
    )
    artifact["optimizer_llm_input"] = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    try:
        llm = get_evolve_llm()
        resp = await llm.chat(
            [{"role": "user", "content": user_prompt}],
            system=system_prompt,
        )
        raw = resp.content or ""
    except Exception as exc:
        artifact["status"] = "failed"
        artifact["reason"] = "optimizer_llm_call_failed"
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        return None, f"optimizer LLM call failed: {type(exc).__name__}: {exc}", artifact

    artifact["optimizer_llm_output_raw"] = str(raw or "")
    parsed = json_fragment(raw)
    if not isinstance(parsed, dict):
        artifact["status"] = "failed"
        artifact["reason"] = "optimizer_returned_non_json"
        return None, "optimizer returned non-json", artifact
    artifact["optimizer_llm_output_parsed"] = parsed
    updates_raw = parsed.get("updates")
    if isinstance(updates_raw, dict):
        updates_raw = [updates_raw]
    if not isinstance(updates_raw, list):
        artifact["status"] = "failed"
        artifact["reason"] = "optimizer_json_missing_updates_list"
        return None, "optimizer json missing updates list", artifact

    updates_by_path: dict[str, str] = {}
    skill_root_resolved = skill_dir.resolve()
    for item in updates_raw:
        if not isinstance(item, dict):
            continue
        rel_path = _normalize_relative_update_path(item.get("path"))
        if not rel_path:
            continue
        content = strip_markdown_fence(str(item.get("content") or ""))
        if not content.strip():
            continue
        abs_target = (skill_dir / rel_path).resolve()
        try:
            abs_target.relative_to(skill_root_resolved)
        except Exception:
            continue
        updates_by_path[rel_path] = content.rstrip() + "\n"

    if not updates_by_path:
        artifact["status"] = "failed"
        artifact["reason"] = "optimizer_returned_no_valid_file_updates"
        return None, "optimizer returned no valid file updates", artifact

    updates = [{"path": p, "content": c} for p, c in sorted(updates_by_path.items())]
    artifact["status"] = "ok"
    artifact["suggested_updates"] = updates
    return updates, "ok", artifact


# ---------------------------------------------------------------------------
# Apply / restore
# ---------------------------------------------------------------------------

def apply_skill_folder_updates(
    *,
    skill_dir: Path,
    updates: list[dict[str, str]],
    backups_root: Path,
) -> dict[str, Any]:
    """Apply file updates to a skill folder with full backup."""
    skill_name = skill_dir.name
    backup_dir = backups_root / skill_name
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = backup_dir / f"bundle.{stamp}"
    shutil.copytree(skill_dir, backup_path)

    updated_files: list[str] = []
    created_files: list[str] = []
    for update in updates:
        rel = str(update.get("path") or "").strip()
        if not rel:
            continue
        new_content = str(update.get("content") or "")
        target_path = (skill_dir / rel).resolve()
        try:
            target_path.relative_to(skill_dir.resolve())
        except Exception:
            continue
        existed = target_path.exists()
        prev_content = target_path.read_text(encoding="utf-8") if existed else None
        if existed and prev_content == new_content:
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(new_content, encoding="utf-8")
        if existed:
            updated_files.append(rel)
        else:
            created_files.append(rel)

    if not updated_files and not created_files:
        return {"status": "failed", "reason": "optimizer returned unchanged content"}
    return {
        "status": "ok",
        "backup": str(backup_path),
        "updated_files": updated_files,
        "created_files": created_files,
    }


def restore_skill_folder_from_backup(
    *, skill_dir: Path, backup_path: Path
) -> dict[str, Any]:
    """Restore a skill folder from a backup snapshot."""
    if not backup_path.exists() or not backup_path.is_dir():
        return {"status": "failed", "reason": f"backup_not_found:{backup_path}"}
    if not skill_dir.exists() or not skill_dir.is_dir():
        return {"status": "failed", "reason": f"skill_dir_not_found:{skill_dir}"}

    tmp_name = f".restore_tmp_{skill_dir.name}_{datetime.now().strftime('%Y%m%dT%H%M%S%fZ')}"
    tmp_path = skill_dir.parent / tmp_name
    try:
        shutil.move(str(skill_dir), str(tmp_path))
        shutil.copytree(backup_path, skill_dir)
    except Exception as exc:
        if not skill_dir.exists() and tmp_path.exists():
            try:
                shutil.move(str(tmp_path), str(skill_dir))
            except Exception:
                pass
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
    return {"status": "ok", "restored_from": str(backup_path)}


# ---------------------------------------------------------------------------
# High-level optimize / discover
# ---------------------------------------------------------------------------

async def optimize_skill_with_trajectory(
    *,
    skill_name: str,
    base_skills_root: Path,
    extra_skills_root: Path,
    task: dict[str, Any],
    user_text: str,
    model_answer: str,
    trace: list[dict[str, Any]],
    skill_judgement: dict[str, Any] | None = None,
    selection_context: dict[str, Any] | None = None,
    include_judge_rationale: bool = True,
) -> dict[str, Any]:
    target = resolve_local_skill_name(
        skill_name,
        base_skills_root=base_skills_root,
        extra_skills_root=extra_skills_root,
    )
    if not target:
        return {"status": "skill_not_found", "skill": skill_name}
    skill_dir = extra_skills_root / target
    if not (skill_dir / "SKILL.md").exists():
        return {"status": "skill_not_in_skill_extra", "skill": target}

    feedback = build_failure_feedback_blob(
        task=task,
        user_text=user_text,
        model_answer=model_answer,
        judge=None,
        trace=trace,
        skill_judgement=skill_judgement,
        include_judge_rationale=include_judge_rationale,
    )
    feedback = append_target_selection_feedback(
        feedback=feedback,
        selection_context=selection_context,
    )
    updates, detail, suggestion = await rewrite_skill_folder_with_feedback(
        skill_name=target,
        skill_dir=skill_dir,
        feedback=feedback,
    )
    if not updates:
        return {
            "status": "optimize_failed",
            "skill": target,
            "error": detail,
            "llm_suggestion": suggestion,
        }

    applied = apply_skill_folder_updates(
        skill_dir=skill_dir,
        updates=updates,
        backups_root=extra_skills_root.parent / ".skill_backups",
    )
    if str(applied.get("status") or "") != "ok":
        return {
            "status": "optimize_failed",
            "skill": target,
            "error": str(applied.get("reason") or "failed applying folder updates"),
            "llm_suggestion": suggestion,
        }
    return {
        "status": "ok",
        "skill": target,
        "mode": "folder_update",
        "backup": applied.get("backup"),
        "updated_files": applied.get("updated_files") or [],
        "created_files": applied.get("created_files") or [],
        "llm_suggestion": suggestion,
    }


async def discover_alternative_skill(
    *,
    skill_name: str,
    base_skills_root: Path,
    extra_skills_root: Path,
    utility_value: float,
    task: dict[str, Any],
    user_text: str,
    model_answer: str,
    trace: list[dict[str, Any]],
    skill_judgement: dict[str, Any] | None = None,
    selection_context: dict[str, Any] | None = None,
    include_judge_rationale: bool = True,
    agent: Any = None,
) -> dict[str, Any]:
    """Discover an alternative implementation for a low-utility skill.

    If the skill exists in extra, rewrites it.  Otherwise, if *agent* is
    provided, instructs the agent to create a new skill via ``skill-creator``.
    """
    target = (
        resolve_local_skill_name(
            skill_name,
            base_skills_root=base_skills_root,
            extra_skills_root=extra_skills_root,
        )
        or str(skill_name or "").strip()
    )
    if not target:
        return {"status": "skill_not_found", "skill": skill_name}
    skill_dir = extra_skills_root / target
    if (skill_dir / "SKILL.md").exists():
        # Rewrite existing extra skill with utility_low hint.
        judge_hint = {
            "is_correct": False,
            "score": 0.0,
            "rationale": f"utility_low={utility_value:.3f}; request alternative approach",
        }
        feedback = build_failure_feedback_blob(
            task=task,
            user_text=user_text,
            model_answer=model_answer,
            judge=judge_hint,
            trace=trace,
            skill_judgement=skill_judgement,
            include_judge_rationale=include_judge_rationale,
        )
        feedback = append_target_selection_feedback(
            feedback=feedback,
            selection_context=selection_context,
        )
        updates, detail, suggestion = await rewrite_skill_folder_with_feedback(
            skill_name=target,
            skill_dir=skill_dir,
            feedback=feedback,
        )
        if not updates:
            return {
                "status": "discover_failed",
                "skill": target,
                "error": detail,
                "llm_suggestion": suggestion,
            }
        applied = apply_skill_folder_updates(
            skill_dir=skill_dir,
            updates=updates,
            backups_root=extra_skills_root.parent / ".skill_backups",
        )
        if str(applied.get("status") or "") != "ok":
            return {
                "status": "discover_failed",
                "skill": target,
                "error": str(applied.get("reason") or "failed applying folder updates"),
                "llm_suggestion": suggestion,
            }
        return {
            "status": "ok",
            "skill": target,
            "mode": "folder_discover",
            "backup": applied.get("backup"),
            "updated_files": applied.get("updated_files") or [],
            "created_files": applied.get("created_files") or [],
            "llm_suggestion": suggestion,
        }

    # Skill does not exist in extra — instruct the agent to create one.
    if agent is None:
        return {
            "status": "discover_failed",
            "skill": target,
            "error": "skill not in extra and no agent provided for creation",
        }

    # Use the agent-based creation flow (via runner.execute_task_once style).
    discover_feedback = build_failure_feedback_blob(
        task=task,
        user_text=user_text,
        model_answer=model_answer,
        judge={"is_correct": False},
        trace=trace,
        skill_judgement=skill_judgement,
        include_judge_rationale=include_judge_rationale,
    )
    discover_feedback = append_target_selection_feedback(
        feedback=discover_feedback,
        selection_context=selection_context,
    )
    prompt = (
        f"Build a better implementation for skill '{target}' using a different approach.\n"
        f"Current utility is low ({utility_value:.3f}).\n\n"
        f"MUST keep skill_name exactly: {target}\n"
        "MUST use action=create\n"
        f"MUST write to skills_dir={extra_skills_root}\n"
        "Do not modify unrelated skills.\n"
        "Do not overfit to a single task.\n\n"
        "Use `read_skill` with `skill-creator` to learn how to create skills, "
        "then create the skill.\n\n"
        f"Failure feedback:\n{discover_feedback}\n\n"
        + SKILL_API_POLICY
    )
    # This will be handled by the runner — caller passes result back.
    return {
        "status": "discover_via_agent",
        "skill": target,
        "prompt": prompt,
    }
