# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core.skill.evolution import SkillEvolutionEngine
from middleware.llm.schema import LLMResponse
from shared.schema import SkillConfig


class FakeLLM:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)

    async def async_chat(self, **_kwargs):
        return LLMResponse(content=json.dumps(self.responses.pop(0)))


def _config(tmp_path, **overrides) -> SkillConfig:
    values = {
        "skills_dir": tmp_path / "skills",
        "builtin_skills_dir": tmp_path / "builtin",
        "workspace_dir": tmp_path / "workspace",
        "evolution_enabled": True,
        "evolution_protected_skills": (),
        "evolution_synthetic_test_enabled": True,
    }
    values.update(overrides)
    config = SkillConfig(**values)
    config.skills_dir.mkdir(parents=True)
    config.workspace_dir.mkdir(parents=True)
    return config


def _write_skill(config: SkillConfig, body: str = "Old guidance") -> None:
    skill_dir = config.skills_dir / "demo-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demonstration skill\n---\n\n" + body,
        encoding="utf-8",
    )


def _responses(new_body: str = "New general guidance") -> list[dict]:
    return [
        {
            "skill_name": "demo-skill",
            "confidence": 0.9,
            "failure_mode": "missing edge-case guidance",
            "rationale": "the skill output shows the missing branch",
            "evidence": ["failed output"],
        },
        {
            "summary": "cover the reusable edge case",
            "files": {
                "SKILL.md": (
                    "---\nname: demo-skill\ndescription: Demonstration skill\n---\n\n"
                    + new_body
                )
            },
            "synthetic_test": {
                "request": "exercise a different example of the edge case",
                "pass_criteria": "returns a concrete successful result",
            },
        },
        {"passed": True, "score": 1.0, "rationale": "criteria satisfied"},
    ]


@pytest.mark.asyncio
async def test_evolution_deploys_only_after_all_gates_and_keeps_backup(tmp_path):
    config = _config(tmp_path)
    _write_skill(config)

    async def candidate_runner(_skill, _request):
        return {"success": True, "result": "concrete successful result"}

    engine = SkillEvolutionEngine(config, FakeLLM(_responses()), candidate_runner)
    result = await engine.evolve_failure(
        task="failed task",
        used_skills=["demo-skill"],
        trace=[{"skill_name": "demo-skill", "ok": False}],
        rationale="supervisor requested replan",
        session_id="session-1",
    )

    assert result.status == "deployed"
    assert "New general guidance" in (
        config.skills_dir / "demo-skill" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert result.backup_path and Path(result.backup_path).is_dir()
    assert [test["name"] for test in result.tests] == [
        "static_validation",
        "local_unit_tests",
        "synthetic_execution",
    ]

    rollback = await engine.rollback_last("demo-skill")
    assert rollback.status == "rolled_back"
    assert "Old guidance" in (config.skills_dir / "demo-skill" / "SKILL.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_failed_synthetic_gate_leaves_original_unchanged(tmp_path):
    config = _config(tmp_path)
    _write_skill(config)

    async def candidate_runner(_skill, _request):
        return {"success": False, "error": "candidate failed"}

    engine = SkillEvolutionEngine(config, FakeLLM(_responses()), candidate_runner)
    result = await engine.evolve_failure(
        task="failed task",
        used_skills=["demo-skill"],
        trace=[{"skill_name": "demo-skill", "ok": False}],
        rationale="failed",
    )

    assert result.status == "rolled_back"
    assert "Old guidance" in (config.skills_dir / "demo-skill" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert result.failed_candidate_path


@pytest.mark.asyncio
async def test_rewriter_cannot_escape_skill_directory(tmp_path):
    config = _config(tmp_path, evolution_synthetic_test_enabled=False)
    _write_skill(config)
    responses = _responses()
    responses[1]["files"] = {"../outside.txt": "forbidden"}

    async def candidate_runner(_skill, _request):
        raise AssertionError("candidate must not execute")

    engine = SkillEvolutionEngine(config, FakeLLM(responses[:2]), candidate_runner)
    result = await engine.evolve_failure(
        task="failed task",
        used_skills=["demo-skill"],
        trace=[{"skill_name": "demo-skill", "ok": False}],
        rationale="failed",
    )

    assert result.status == "rolled_back"
    assert not (config.skills_dir / "outside.txt").exists()
    assert "Old guidance" in (config.skills_dir / "demo-skill" / "SKILL.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_rollback_rejects_path_traversal(tmp_path):
    config = _config(tmp_path)

    async def candidate_runner(_skill, _request):
        raise AssertionError("candidate must not execute")

    engine = SkillEvolutionEngine(config, FakeLLM([]), candidate_runner)
    result = await engine.rollback_last("../../outside")

    assert result.status == "rollback_failed"
    assert "Invalid skill name" in result.error


@pytest.mark.asyncio
async def test_utility_updates_are_serialized(tmp_path):
    config = _config(tmp_path)

    async def candidate_runner(_skill, _request):
        raise AssertionError("candidate must not execute")

    engine = SkillEvolutionEngine(config, FakeLLM([]), candidate_runner)
    await asyncio.gather(
        *(engine.record_outcome("demo-skill", index % 2 == 0) for index in range(20))
    )

    utility = json.loads(
        (config.skills_dir / ".evolution" / "utility.json").read_text(encoding="utf-8")
    )["demo-skill"]
    assert utility["success"] == 10
    assert utility["failure"] == 10
    assert utility["utility"] == 0.5


@pytest.mark.asyncio
async def test_low_utility_triggers_transactional_discovery_and_resets_generation(
    tmp_path,
):
    config = _config(
        tmp_path,
        evolution_utility_discovery_threshold=0.2,
        evolution_utility_min_samples=3,
    )
    _write_skill(config)

    async def candidate_runner(_skill, _request):
        return {"success": True, "result": "concrete successful result"}

    llm = FakeLLM(_responses("A fundamentally different reliable workflow"))
    engine = SkillEvolutionEngine(config, llm, candidate_runner)
    for _ in range(3):
        await engine.record_outcome("demo-skill", False)

    result = await engine.evolve_failure(
        task="failed task",
        used_skills=["demo-skill"],
        trace=[{"skill_name": "demo-skill", "ok": False}],
        rationale="repeated methodology failure",
    )

    assert result.status == "deployed"
    assert result.strategy == "discover"
    assert result.utility == {
        "success": 0,
        "failure": 3,
        "samples": 3,
        "utility": 0.0,
        "generation": 0,
    }
    utility = await engine.get_utility("demo-skill")
    assert utility["utility"] == 0.5
    assert utility["samples"] == 0
    assert utility["generation"] == 1


@pytest.mark.asyncio
async def test_best_of_n_candidates_deploys_highest_judge_score(tmp_path):
    config = _config(tmp_path, evolution_candidate_attempts=2)
    _write_skill(config)

    responses = [
        _responses("Candidate one")[0],
        _responses("Candidate one")[1],
        {"passed": True, "score": 0.4, "rationale": "acceptable"},
        _responses("Candidate two")[1],
        {"passed": True, "score": 0.9, "rationale": "more robust"},
    ]

    async def candidate_runner(_skill, _request):
        return {"success": True, "result": "concrete successful result"}

    engine = SkillEvolutionEngine(config, FakeLLM(responses), candidate_runner)
    result = await engine.evolve_failure(
        task="failed task",
        used_skills=["demo-skill"],
        trace=[{"skill_name": "demo-skill", "ok": False}],
        rationale="failed",
    )

    assert result.status == "deployed"
    assert [item["score"] for item in result.candidate_attempts] == [0.4, 0.9]
    assert "Candidate two" in (config.skills_dir / "demo-skill" / "SKILL.md").read_text(
        encoding="utf-8"
    )
