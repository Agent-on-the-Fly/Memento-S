# SPDX-License-Identifier: Apache-2.0
"""Transactional skill evolution with attribution, tests, and rollback.

The implementation follows the paper's write phase:

1. attribute a failure to one used skill;
2. ask an LLM for targeted file-level replacements;
3. apply them to an isolated copy;
4. run structural, local unit, and synthetic execution gates;
5. atomically deploy on success or retain the original tree on failure.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from core.skill.loader import load_from_dir
from core.skill.schema import Skill
from middleware.llm import LLMClient
from shared.schema import SkillConfig
from utils.logger import get_logger
from utils.strings import to_kebab_case

from .prompts import (
    ATTRIBUTION_PROMPT,
    DISCOVERY_REWRITE_PROMPT,
    REWRITE_PROMPT,
    SYNTHETIC_JUDGE_PROMPT,
)

logger = get_logger(__name__)

CandidateRunner = Callable[[Skill, str], Awaitable[dict[str, Any]]]


def _extract_json(text: str) -> dict[str, Any]:
    """Extract one JSON object without importing the agent package."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.removesuffix("```")
        stripped = stripped.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start : end + 1])
    raise ValueError(f"No JSON object found in response: {stripped[:200]}")


class _AttributionPayload(BaseModel):
    skill_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    failure_mode: str
    rationale: str
    evidence: list[str] = Field(default_factory=list)


class _SyntheticTest(BaseModel):
    request: str
    pass_criteria: str


class _RewritePayload(BaseModel):
    summary: str
    files: dict[str, str]
    synthetic_test: _SyntheticTest | None = None


class _JudgePayload(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


@dataclass(frozen=True)
class AttributionResult:
    skill_name: str
    confidence: float
    failure_mode: str
    rationale: str
    evidence: tuple[str, ...] = ()


@dataclass
class EvolutionResult:
    status: str
    skill_name: str = ""
    event_id: str = ""
    strategy: str = "optimize"
    summary: str = ""
    attribution: AttributionResult | None = None
    utility: dict[str, Any] = field(default_factory=dict)
    candidate_attempts: list[dict[str, Any]] = field(default_factory=list)
    tests: list[dict[str, Any]] = field(default_factory=list)
    backup_path: str = ""
    failed_candidate_path: str = ""
    error: str = ""

    @property
    def deployed(self) -> bool:
        return self.status == "deployed"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["deployed"] = self.deployed
        return data


class SkillEvolutionEngine:
    """Orchestrates guarded mutations of local skill folders."""

    def __init__(
        self,
        config: SkillConfig,
        llm: LLMClient | None,
        candidate_runner: CandidateRunner | None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._candidate_runner = candidate_runner
        self._root = config.skills_dir / ".evolution"
        self._locks: dict[str, asyncio.Lock] = {}
        self._utility_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._config.evolution_enabled)

    @staticmethod
    def _normalize_skill_name(skill_name: str) -> str:
        normalized = to_kebab_case(skill_name.strip())
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", normalized):
            raise ValueError(f"Invalid skill name: {skill_name!r}")
        return normalized

    async def record_outcome(self, skill_name: str, success: bool) -> dict[str, Any]:
        """Persist empirical skill utility using an atomic replace."""
        if not self.enabled or not skill_name:
            return {}
        try:
            normalized = self._normalize_skill_name(skill_name)
        except ValueError:
            return {}
        async with self._utility_lock:
            utility_path = self._root / "utility.json"
            self._root.mkdir(parents=True, exist_ok=True)
            try:
                data = (
                    json.loads(utility_path.read_text(encoding="utf-8"))
                    if utility_path.exists()
                    else {}
                )
            except (json.JSONDecodeError, OSError):
                data = {}
            entry = data.setdefault(normalized, {"success": 0, "failure": 0})
            key = "success" if success else "failure"
            entry[key] = int(entry.get(key, 0)) + 1
            total = int(entry.get("success", 0)) + int(entry.get("failure", 0))
            entry["utility"] = int(entry.get("success", 0)) / total if total else 0.5
            entry["updated_at"] = datetime.now(UTC).isoformat()
            temp = utility_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temp, utility_path)
            return dict(entry)

    async def get_utility(self, skill_name: str) -> dict[str, Any]:
        """Return persisted empirical utility, defaulting to the paper's 0.5 prior."""
        try:
            normalized = self._normalize_skill_name(skill_name)
        except ValueError:
            return {"success": 0, "failure": 0, "samples": 0, "utility": 0.5}
        async with self._utility_lock:
            utility_path = self._root / "utility.json"
            try:
                table = (
                    json.loads(utility_path.read_text(encoding="utf-8"))
                    if utility_path.exists()
                    else {}
                )
            except (json.JSONDecodeError, OSError):
                table = {}
            raw = table.get(normalized, {})
            success = max(0, int(raw.get("success", 0)))
            failure = max(0, int(raw.get("failure", 0)))
            samples = success + failure
            return {
                "success": success,
                "failure": failure,
                "samples": samples,
                "utility": success / samples if samples else 0.5,
                "generation": max(0, int(raw.get("generation", 0))),
            }

    async def evolve_failure(
        self,
        *,
        task: str,
        used_skills: list[str],
        trace: list[dict[str, Any]],
        rationale: str,
        session_id: str = "",
    ) -> EvolutionResult:
        """Attribute and transactionally evolve one skill from a failed trace."""
        if not self.enabled:
            return EvolutionResult(status="disabled")

        protected = {
            to_kebab_case(name) for name in self._config.evolution_protected_skills
        }
        candidates = []
        for name in used_skills:
            try:
                normalized = self._normalize_skill_name(name)
            except ValueError:
                continue
            path = self._config.skills_dir / normalized
            if (
                normalized not in protected
                and path.is_dir()
                and (path / "SKILL.md").is_file()
                and normalized not in candidates
            ):
                candidates.append(normalized)
        if not candidates:
            return EvolutionResult(
                status="skipped",
                error="No mutable local skill was used in the failed trajectory",
            )

        event_id = (
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        )
        try:
            attribution = await self._attribute(task, candidates, trace, rationale)
        except Exception as exc:  # noqa: BLE001 - attribution providers have heterogeneous failures
            result = EvolutionResult(
                status="attribution_failed", event_id=event_id, error=str(exc)
            )
            self._audit(result, task=task, trace=trace, session_id=session_id)
            return result

        result = EvolutionResult(
            status="attributed",
            skill_name=attribution.skill_name,
            event_id=event_id,
            attribution=attribution,
        )
        if attribution.confidence < self._config.evolution_min_attribution_confidence:
            result.status = "skipped_low_confidence"
            result.error = (
                f"Attribution confidence {attribution.confidence:.2f} is below "
                f"{self._config.evolution_min_attribution_confidence:.2f}"
            )
            self._audit(result, task=task, trace=trace, session_id=session_id)
            return result

        result.utility = await self.get_utility(attribution.skill_name)
        if (
            self._config.evolution_utility_discovery_enabled
            and result.utility["samples"] >= self._config.evolution_utility_min_samples
            and result.utility["utility"]
            < self._config.evolution_utility_discovery_threshold
        ):
            # gaia_evolve.py deletes the low-utility skill and recreates the same
            # name.  Production keeps the same policy decision but performs the
            # rebuild on an isolated copy so the active skill is never missing.
            result.strategy = "discover"

        lock = self._locks.setdefault(attribution.skill_name, asyncio.Lock())
        async with lock:
            return await self._evolve_attributed(
                result=result,
                task=task,
                trace=trace,
                session_id=session_id,
            )

    async def rollback_last(self, skill_name: str) -> EvolutionResult:
        """Restore the most recent deployed backup for a skill."""
        try:
            normalized = self._normalize_skill_name(skill_name)
        except ValueError as exc:
            return EvolutionResult(status="rollback_failed", error=str(exc))
        backup_root = self._root / "backups" / normalized
        backups = (
            sorted((p for p in backup_root.iterdir() if p.is_dir()), reverse=True)
            if backup_root.exists()
            else []
        )
        if not backups:
            return EvolutionResult(
                status="rollback_failed", skill_name=normalized, error="No backup found"
            )
        target = self._config.skills_dir / normalized
        rollback_staging = self._root / "staging" / f"rollback-{uuid.uuid4().hex}"
        rollback_staging.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(backups[0], rollback_staging)
        try:
            restored_skill = load_from_dir(rollback_staging)
            if self._normalize_skill_name(restored_skill.name) != normalized:
                raise ValueError("Backup skill identity does not match rollback target")
        except Exception as exc:  # noqa: BLE001 - invalid/corrupt backups must fail closed
            shutil.rmtree(rollback_staging, ignore_errors=True)
            return EvolutionResult(
                status="rollback_failed", skill_name=normalized, error=str(exc)
            )
        displaced = (
            self._root
            / "failed"
            / f"manual-rollback-{normalized}-{uuid.uuid4().hex[:8]}"
        )
        displaced.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not target.is_dir():
                raise FileNotFoundError(f"Active skill not found: {normalized}")
            os.replace(target, displaced)
            os.replace(rollback_staging, target)
        except Exception as exc:  # noqa: BLE001 - atomic swap recovery must handle every failure
            if not target.exists() and displaced.exists():
                os.replace(displaced, target)
            shutil.rmtree(rollback_staging, ignore_errors=True)
            return EvolutionResult(
                status="rollback_failed", skill_name=normalized, error=str(exc)
            )
        result = EvolutionResult(
            status="rolled_back",
            skill_name=normalized,
            event_id=backups[0].name,
            backup_path=str(backups[0]),
            failed_candidate_path=str(displaced),
        )
        self._audit(result, task="manual rollback", trace=[], session_id="")
        return result

    async def _attribute(
        self,
        task: str,
        candidates: list[str],
        trace: list[dict[str, Any]],
        rationale: str,
    ) -> AttributionResult:
        prompt = ATTRIBUTION_PROMPT.format(
            task=task,
            rationale=rationale,
            candidates="\n".join(f"- {name}" for name in candidates),
            trace=self._truncate_json(trace),
        )
        payload = _AttributionPayload(**_extract_json(await self._chat(prompt)))
        normalized = self._normalize_skill_name(payload.skill_name)
        if normalized not in candidates:
            raise ValueError(
                f"Attribution selected non-candidate skill: {payload.skill_name}"
            )
        return AttributionResult(
            skill_name=normalized,
            confidence=payload.confidence,
            failure_mode=payload.failure_mode,
            rationale=payload.rationale,
            evidence=tuple(payload.evidence),
        )

    async def _evolve_attributed(
        self,
        *,
        result: EvolutionResult,
        task: str,
        trace: list[dict[str, Any]],
        session_id: str,
    ) -> EvolutionResult:
        skill_dir = (self._config.skills_dir / result.skill_name).resolve()
        root = self._config.skills_dir.resolve()
        if skill_dir.parent != root or not skill_dir.is_dir():
            result.status = "update_failed"
            result.error = (
                "Resolved skill directory is outside the configured skill root"
            )
            return result

        staging_root = self._root / "staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f"{result.skill_name}-", dir=staging_root)
        )

        try:
            candidate_count = max(1, int(self._config.evolution_candidate_attempts))
            best_candidate: Path | None = None
            best_proposal: _RewritePayload | None = None
            best_tests: list[dict[str, Any]] = []
            best_score = -1.0

            for attempt in range(1, candidate_count + 1):
                candidate_parent = (
                    staging if candidate_count == 1 else staging / f"attempt-{attempt}"
                )
                candidate = candidate_parent / result.skill_name
                candidate.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(skill_dir, candidate)
                attempt_tests: list[dict[str, Any]] = []
                proposal: _RewritePayload | None = None
                try:
                    proposal = await self._rewrite(
                        task,
                        result.attribution,
                        trace,
                        candidate,
                        strategy=result.strategy,
                        utility=result.utility,
                    )
                    changed = self._apply_replacements(candidate, proposal.files)
                    if not changed:
                        raise ValueError(
                            "Rewriter did not make any effective file change"
                        )

                    attempt_tests.extend(
                        await self._static_and_unit_tests(candidate, result.skill_name)
                    )
                    if any(not item["passed"] for item in attempt_tests):
                        raise RuntimeError("Static or local unit-test gate failed")

                    score = 1.0
                    if self._config.evolution_synthetic_test_enabled:
                        if proposal.synthetic_test is None:
                            raise ValueError(
                                "Synthetic test is required by configuration"
                            )
                        synthetic = await self._run_synthetic_test(
                            candidate, proposal.synthetic_test
                        )
                        attempt_tests.append(synthetic)
                        if not synthetic["passed"]:
                            raise RuntimeError("Synthetic execution gate failed")
                        score = float(synthetic.get("score", 0.0))

                    result.candidate_attempts.append(
                        {
                            "attempt": attempt,
                            "passed": True,
                            "score": score,
                            "summary": proposal.summary,
                        }
                    )
                    if best_candidate is None or score > best_score:
                        best_candidate = candidate
                        best_proposal = proposal
                        best_tests = attempt_tests
                        best_score = score
                except Exception as exc:  # noqa: BLE001 - one bad candidate must not abort best-of-N
                    result.candidate_attempts.append(
                        {
                            "attempt": attempt,
                            "passed": False,
                            "error": str(exc),
                            "tests": attempt_tests,
                        }
                    )

            if best_candidate is None or best_proposal is None:
                last = result.candidate_attempts[-1]
                result.tests = list(last.get("tests", []))
                raise RuntimeError(
                    f"All {candidate_count} evolution candidate(s) failed; "
                    f"last error: {last.get('error', 'unknown')}"
                )

            candidate = best_candidate
            result.summary = best_proposal.summary
            result.tests = best_tests

            backup = self._root / "backups" / result.skill_name / result.event_id
            backup.parent.mkdir(parents=True, exist_ok=True)
            os.replace(skill_dir, backup)
            try:
                os.replace(candidate, skill_dir)
                # Validate the deployed path, not only the staging path.
                load_from_dir(skill_dir)
            except Exception:
                if skill_dir.exists():
                    failed_deploy = self._root / "failed" / f"deploy-{result.event_id}"
                    failed_deploy.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(skill_dir, failed_deploy)
                    result.failed_candidate_path = str(failed_deploy)
                os.replace(backup, skill_dir)
                raise

            result.status = "deployed"
            result.backup_path = str(backup)
            shutil.rmtree(staging, ignore_errors=True)
            if result.strategy == "discover":
                try:
                    await self._reset_utility_after_discovery(
                        result.skill_name,
                        event_id=result.event_id,
                        previous=result.utility,
                    )
                except OSError as exc:
                    # The skill is already valid and deployed; a bookkeeping
                    # write must not falsely report that the live tree rolled back.
                    logger.warning(
                        f"Failed to reset utility generation for "
                        f"'{result.skill_name}': {exc}"
                    )
        except Exception as exc:  # noqa: BLE001 - transaction boundary performs rollback
            result.status = "rolled_back"
            result.error = str(exc)
            # Before deployment the original never moved; after a deploy error it
            # was restored in the inner exception handler.  Preserve the rejected
            # candidate for audit when configured.
            if staging.exists():
                if self._config.evolution_keep_failed_candidate:
                    failed = self._root / "failed" / result.event_id
                    failed.parent.mkdir(parents=True, exist_ok=True)
                    if failed.exists():
                        shutil.rmtree(failed)
                    os.replace(staging, failed)
                    result.failed_candidate_path = str(failed)
                else:
                    shutil.rmtree(staging, ignore_errors=True)

        self._audit(result, task=task, trace=trace, session_id=session_id)
        return result

    async def _rewrite(
        self,
        task: str,
        attribution: AttributionResult | None,
        trace: list[dict[str, Any]],
        candidate: Path,
        *,
        strategy: str,
        utility: dict[str, Any],
    ) -> _RewritePayload:
        files = self._read_text_files(candidate)
        prompt_template = (
            DISCOVERY_REWRITE_PROMPT if strategy == "discover" else REWRITE_PROMPT
        )
        prompt = prompt_template.format(
            task=task,
            attribution=json.dumps(
                asdict(attribution) if attribution else {}, ensure_ascii=False
            ),
            utility=json.dumps(utility, ensure_ascii=False),
            trace=self._truncate_json(trace),
            files=json.dumps(files, ensure_ascii=False),
        )
        return _RewritePayload(**_extract_json(await self._chat(prompt)))

    async def _reset_utility_after_discovery(
        self,
        skill_name: str,
        *,
        event_id: str,
        previous: dict[str, Any],
    ) -> None:
        """Start a fresh utility generation after a full low-utility rebuild."""
        async with self._utility_lock:
            utility_path = self._root / "utility.json"
            try:
                table = (
                    json.loads(utility_path.read_text(encoding="utf-8"))
                    if utility_path.exists()
                    else {}
                )
            except (json.JSONDecodeError, OSError):
                table = {}
            generation = max(0, int(previous.get("generation", 0))) + 1
            table[skill_name] = {
                "success": 0,
                "failure": 0,
                "utility": 0.5,
                "generation": generation,
                "reset_from": {
                    "success": int(previous.get("success", 0)),
                    "failure": int(previous.get("failure", 0)),
                    "utility": float(previous.get("utility", 0.5)),
                },
                "reset_event_id": event_id,
                "updated_at": datetime.now(UTC).isoformat(),
            }
            self._root.mkdir(parents=True, exist_ok=True)
            temp = utility_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temp.write_text(
                json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temp, utility_path)

    def _read_text_files(self, skill_dir: Path) -> dict[str, str]:
        allowed = {
            ".md",
            ".txt",
            ".rst",
            ".py",
            ".js",
            ".ts",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".sh",
        }
        output: dict[str, str] = {}
        budget = self._config.evolution_max_prompt_chars
        for path in sorted(skill_dir.rglob("*")):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.suffix.lower() not in allowed
            ):
                continue
            relative = path.relative_to(skill_dir).as_posix()
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            remaining = budget - sum(len(value) for value in output.values())
            if remaining <= 0:
                break
            output[relative] = content[:remaining]
        return output

    @staticmethod
    def _safe_destination(root: Path, relative: str) -> Path:
        if not relative or relative.startswith(("/", "\\")):
            raise ValueError(f"Unsafe replacement path: {relative!r}")
        parts = Path(relative).parts
        if ".." in parts or any(part in {"", "."} for part in parts):
            raise ValueError(f"Unsafe replacement path: {relative!r}")
        destination = (root / relative).resolve()
        if root.resolve() not in destination.parents:
            raise ValueError(f"Replacement escapes skill directory: {relative!r}")
        return destination

    def _apply_replacements(self, root: Path, files: dict[str, str]) -> list[str]:
        changed: list[str] = []
        for relative, content in files.items():
            if not isinstance(content, str):
                raise TypeError(f"Replacement for {relative!r} must be a string")
            destination = self._safe_destination(root, relative)
            if destination.exists() and destination.is_symlink():
                raise ValueError(f"Refusing to replace symlink: {relative!r}")
            old = (
                destination.read_text(encoding="utf-8")
                if destination.exists()
                else None
            )
            if old == content:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
            changed.append(relative)
        return changed

    async def _static_and_unit_tests(
        self,
        candidate: Path,
        expected_skill_name: str,
    ) -> list[dict[str, Any]]:
        tests: list[dict[str, Any]] = []
        try:
            loaded = load_from_dir(candidate)
            if self._normalize_skill_name(loaded.name) != expected_skill_name:
                raise ValueError(
                    "Candidate changed skill identity: "
                    f"expected {expected_skill_name!r}, got {loaded.name!r}"
                )
            for py_file in candidate.rglob("*.py"):
                if py_file.is_symlink():
                    raise ValueError(f"Symlinked Python file is not allowed: {py_file}")
                compile(py_file.read_text(encoding="utf-8"), str(py_file), "exec")
            tests.append({"name": "static_validation", "passed": True})
        except Exception as exc:  # noqa: BLE001 - static gate reports arbitrary parser/I/O failures
            return [{"name": "static_validation", "passed": False, "error": str(exc)}]

        test_files = sorted(candidate.rglob("test_*.py"))
        if not test_files:
            tests.append({"name": "local_unit_tests", "passed": True, "skipped": True})
            return tests
        cmd = [sys.executable, "-m", "pytest", "-q", *[str(p) for p in test_files]]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=candidate,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._config.evolution_test_timeout_sec
            )
            tests.append(
                {
                    "name": "local_unit_tests",
                    "passed": proc.returncode == 0,
                    "command": shlex.join(cmd),
                    "output": (stdout + stderr).decode(errors="replace")[-4000:],
                }
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            tests.append(
                {"name": "local_unit_tests", "passed": False, "error": "timeout"}
            )
        return tests

    async def _run_synthetic_test(
        self, candidate: Path, spec: _SyntheticTest
    ) -> dict[str, Any]:
        if self._candidate_runner is None:
            return {
                "name": "synthetic_execution",
                "passed": False,
                "error": "Candidate runner is not configured",
            }
        skill = load_from_dir(candidate)
        execution = await self._candidate_runner(skill, spec.request)
        prompt = SYNTHETIC_JUDGE_PROMPT.format(
            request=spec.request,
            pass_criteria=spec.pass_criteria,
            result=self._truncate_json(execution),
        )
        try:
            judge = _JudgePayload(**_extract_json(await self._chat(prompt)))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            return {"name": "synthetic_execution", "passed": False, "error": str(exc)}
        execution_ok = bool(execution.get("success", execution.get("ok", False)))
        return {
            "name": "synthetic_execution",
            "passed": execution_ok and judge.passed,
            "score": judge.score,
            "rationale": judge.rationale,
            "request": spec.request,
        }

    async def _chat(self, prompt: str) -> str:
        if self._llm is None:
            raise RuntimeError("LLM client is not configured")
        response = await self._llm.async_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return (response.content or "").strip()

    def _truncate_json(self, value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str)
        limit = self._config.evolution_max_prompt_chars
        return text if len(text) <= limit else text[:limit] + "...[truncated]"

    def _audit(
        self,
        result: EvolutionResult,
        *,
        task: str,
        trace: list[dict[str, Any]],
        session_id: str,
    ) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            event = {
                "timestamp": datetime.now(UTC).isoformat(),
                "session_id": session_id,
                "task": task,
                "trace_skills": [item.get("skill_name") for item in trace],
                **result.to_dict(),
            }
            with (self._root / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.warning(f"Failed to write skill evolution audit event: {exc}")
