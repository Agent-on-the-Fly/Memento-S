# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from core.skill.retrieval.multi_recall import MultiRecall
from core.skill.retrieval.qwen_recall import QwenRecall
from shared.schema import SkillConfig, SkillSearchResult


def _write_skill(root, name: str, description: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_hybrid_config_installs_keyword_and_qwen_routes(tmp_path):
    _write_skill(tmp_path, "alpha", "Alpha skill")
    config = SkillConfig(
        skills_dir=tmp_path,
        builtin_skills_dir=tmp_path / "builtin",
        workspace_dir=tmp_path / "workspace",
        retrieval_router_backend="hybrid",
        retrieval_embedding_base_url="https://embedding.invalid",
    )

    recall = await MultiRecall.from_config(config)

    assert [route.name for route in recall.get_available_recalls()] == ["local", "qwen"]


def test_placeholder_api_key_does_not_enable_remote_qwen(tmp_path):
    recall = QwenRecall(
        tmp_path,
        base_url="https://embedding.invalid",
        api_key="sk-xxxxxx",
    )

    assert recall.is_available() is False


@pytest.mark.asyncio
async def test_qwen_recall_uses_instruction_for_query_only(tmp_path, monkeypatch):
    _write_skill(tmp_path, "pdf-reader", "Read and extract text from PDF files")
    _write_skill(tmp_path, "weather", "Look up current weather forecasts")
    recall = QwenRecall(tmp_path, base_url="https://embedding.invalid")

    calls: list[list[str]] = []

    async def fake_embed(texts: list[str]):
        calls.append(texts)
        if len(texts) == 1 and texts[0].startswith("Instruct:"):
            return [[1.0, 0.0]]
        return [
            [1.0, 0.0] if "skill: pdf-reader" in text.lower() else [0.0, 1.0]
            for text in texts
        ]

    monkeypatch.setattr(recall, "_embed", fake_embed)
    results = await recall.search("extract a table from this PDF", k=1)

    assert [item.name for item in results] == ["pdf-reader"]
    assert results[0].match_type == "qwen_embedding"
    assert not calls[0][0].startswith("Instruct:")
    assert calls[0][0].startswith("Skill: pdf-reader\nDescription:")
    assert calls[1][0].startswith("Instruct:")


@pytest.mark.asyncio
async def test_qwen_recall_rebuilds_document_embeddings_after_skill_change(
    tmp_path, monkeypatch
):
    _write_skill(tmp_path, "alpha", "first version")
    recall = QwenRecall(tmp_path, base_url="https://embedding.invalid")
    document_calls = 0

    async def fake_embed(texts: list[str]):
        nonlocal document_calls
        if not texts[0].startswith("Instruct:"):
            document_calls += 1
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(recall, "_embed", fake_embed)
    await recall.search("alpha", k=1)
    await recall.search("alpha", k=1)
    assert document_calls == 1

    skill_md = tmp_path / "alpha" / "SKILL.md"
    skill_md.write_text(
        "---\nname: alpha\ndescription: a much longer second version\n---\n",
        encoding="utf-8",
    )
    await recall.search("alpha", k=1)
    assert document_calls == 2


@pytest.mark.asyncio
async def test_qwen_recall_reuses_valid_disk_cache(tmp_path, monkeypatch):
    _write_skill(tmp_path, "alpha", "first version")
    first = QwenRecall(tmp_path, base_url="https://embedding.invalid")

    async def first_embed(texts: list[str]):
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(first, "_embed", first_embed)
    await first.search("alpha", k=1)
    assert list((tmp_path / ".router-cache").glob("qwen.*.npz"))

    second = QwenRecall(tmp_path, base_url="https://embedding.invalid")
    calls: list[list[str]] = []

    async def query_only_embed(texts: list[str]):
        calls.append(texts)
        assert texts[0].startswith("Instruct:")
        return [[1.0, 0.0]]

    monkeypatch.setattr(second, "_embed", query_only_embed)
    results = await second.search("alpha", k=1)

    assert [item.name for item in results] == ["alpha"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_qwen_runtime_failure_falls_back_to_keyword(tmp_path, monkeypatch):
    _write_skill(tmp_path, "alpha-tool", "Handle alpha workflows")
    recall = QwenRecall(tmp_path, base_url="https://embedding.invalid")

    async def failed_embed(_texts: list[str]):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(recall, "_embed", failed_embed)
    results = await recall.search("alpha", k=1)

    assert [item.name for item in results] == ["alpha-tool"]
    assert results[0].match_type == "keyword"
    assert "model unavailable" in results[0].metadata["qwen_fallback"]


@pytest.mark.asyncio
async def test_qwen_only_config_has_keyword_fallback_when_unconfigured(tmp_path):
    _write_skill(tmp_path, "alpha", "Alpha skill")
    config = SkillConfig(
        skills_dir=tmp_path,
        builtin_skills_dir=tmp_path / "builtin",
        workspace_dir=tmp_path / "workspace",
        retrieval_router_backend="qwen",
    )

    recall = await MultiRecall.from_config(config)

    assert [route.name for route in recall.get_available_recalls()] == ["local"]


@pytest.mark.asyncio
async def test_hybrid_fusion_rewards_cross_router_agreement():
    class FakeRecall:
        def __init__(self, name, results):
            self.name = name
            self.results = results

        def is_available(self):
            return True

        async def search(self, _query, k=10, **_kwargs):
            return self.results[:k]

        def get_stats(self):
            return {"name": self.name, "available": True}

    keyword = FakeRecall(
        "local",
        [
            SkillSearchResult("shared", score=0.6, match_type="keyword"),
            SkillSearchResult("keyword-only", score=0.9, match_type="keyword"),
        ],
    )
    qwen = FakeRecall(
        "qwen",
        [SkillSearchResult("shared", score=0.6, match_type="qwen_embedding")],
    )

    results = await MultiRecall([keyword, qwen]).search("goal", k=2)

    assert [result.name for result in results] == ["shared", "keyword-only"]
    assert results[0].match_type == "hybrid"
    assert results[0].metadata["router_scores"] == {
        "local": {"rank": 1, "score": 0.6},
        "qwen": {"rank": 1, "score": 0.6},
    }
