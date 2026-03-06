
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from core.config.config import g_settings
from core.config.logging import get_logger
from .execution import SkillExecutor
from .importers import cjk_query_to_english
from .observability import track_resolve
from .retrieval.stopwords import QUERY_STOPWORDS
from .schema import Skill, SkillExecutionResult
from .skills import SkillCreator, SkillLibrary

logger = get_logger(__name__)


@dataclass
class ResolveResult:

    skill: Skill | None = None
    source: Literal["local", "llm_generated"] | None = None
    execution_result: SkillExecutionResult | None = None
    query_category: str | None = None
    generated_code: str | None = None


async def _to_thread(fn: Callable, *args: Any) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


class SkillResolver:

    def __init__(
        self,
        library: SkillLibrary,
        creator: SkillCreator,
        llm: Any = None,
    ):
        self.library = library
        self.creator = creator

        self.executor = SkillExecutor(llm=llm)
        self.strategy = g_settings.resolve_strategy

    async def resolve(
        self,
        query: str,
        task_name: str | None = None,
        params: dict[str, Any] | None = None,
        execute: bool = True,
        skip_stages: set[str] | None = None,
    ) -> ResolveResult:
        skip_stages = skip_stages or set()
        t0 = time.monotonic()
        logger.info("Resolving: '%s' (strategy=%s, skip=%s)", query, self.strategy, skip_stages or "none")

        query_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", query))
        query_for_search = await cjk_query_to_english(query) if query_has_cjk else query
        if query_has_cjk and re.search(r"[\u4e00-\u9fff]", query_for_search):
            logger.warning("CJK→EN translation incomplete, retrieval will use CJK query")

        skill: Skill | None = None
        source: Literal["local", "llm_generated"] | None = None

        if "local" not in skip_stages:
            skill, source = await _to_thread(self._search_local, query, query_for_search)

        if not skill and "llm" not in skip_stages:
            name = task_name or await self._generate_task_name(query)
            logger.info("Creating skill '%s' via LLM...", name)
            skill = await self.creator.create_skill(name, query)
            source = "llm_generated"
            logger.info("Step 3: LLM generated new skill '%s'", skill.name)

        result = ResolveResult(skill=skill, source=source)

        if execute and skill:
            result.execution_result, result.generated_code = await self.executor.execute(
                skill, query=query, params=params,
            )
            if result.execution_result and result.execution_result.success and source != "local":
                try:
                    self.library.add_skill(skill)
                    logger.info("Hot-loaded skill '%s' (source=%s) into library", skill.name, source)
                except Exception as e:
                    logger.warning("Failed to hot-load skill '%s': %s", skill.name, e)

        latency_ms = (time.monotonic() - t0) * 1000
        track_resolve(
            source or "none",
            bool(result.execution_result and result.execution_result.success),
            latency_ms,
        )
        return result


    def _search_local(
        self,
        original_query: str,
        search_query: str,
    ) -> tuple[Skill | None, str | None]:
        try:
            exact = self._exact_name_match(original_query)
            if exact:
                logger.info("Step 1 HIT (exact name match): '%s'", exact.name)
                return exact, "local"

            extra_queries: list[str] = []
            if search_query != original_query:
                extra_queries.append(original_query)

            scored_results = self.library.retrieve_skills(
                search_query, k=g_settings.retrieval_top_k,
                extra_queries=extra_queries or None,
            )

            if not scored_results:
                logger.info("Step 1: no candidates from local")
                return None, None

            best = scored_results[0]
            logger.info(
                "Step 1 HIT: '%s' (score=%.4f, bm25=%.4f, emb=%.4f, rerank=%s)",
                best.skill.name, best.score, best.bm25_score, best.embedding_score,
                f"{best.reranker_score:.4f}" if best.reranker_score is not None else "N/A",
            )
            return best.skill, "local"

        except Exception as e:
            logger.error("Local search failed: %s", e)
        return None, None

    def _exact_name_match(self, query: str) -> Skill | None:
        q = query.strip().lower()
        if not q:
            return None

        q_snake = q.replace("-", "_").replace(" ", "_")
        q_kebab = q.replace("_", "-").replace(" ", "-")
        q_variants = {q, q_snake, q_kebab}

        prefix_hits: list[Skill] = []
        substr_hits: list[Skill] = []

        for name, skill in self.library.local_cache.items():
            name_lower = name.lower()
            if name_lower in q_variants:
                return skill
            if name_lower.startswith(q_snake + "_") or name_lower.startswith(q_kebab + "-"):
                prefix_hits.append(skill)
            elif len(q) >= 3 and (q_snake in name_lower or q_kebab in name_lower):
                substr_hits.append(skill)

        if len(prefix_hits) == 1:
            return prefix_hits[0]
        if len(substr_hits) == 1:
            return substr_hits[0]

        return None

    async def _generate_task_name(self, query: str) -> str:
        has_non_ascii = any(ord(c) > 127 for c in query)

        if not has_non_ascii:
            words = [
                w for w in query.lower().split()
                if w.isalnum() and w not in QUERY_STOPWORDS and len(w) >= 2
            ][:4]
            name = "_".join(words)
            if name and len(name) >= 3:
                return name

        try:
            prompt = (
                "Convert this task description into a concise English skill name "
                "(2-4 words, lowercase, separated by hyphens, no special characters).\n\n"
                f"Task: {query}\n\n"
                "Return ONLY the skill name, nothing else. "
                "Example: 'auto-publish-blog', 'send-email', 'generate-report'"
            )
            result = (await self.creator._call_llm(prompt, timeout=30)).strip().lower()
            result = re.sub(r'[^a-z0-9\-]', '-', result)
            result = re.sub(r'-+', '-', result).strip('-')
            if result and len(result) >= 3:
                logger.info("Task name generated: '%s' → '%s'", query[:30], result)
                return result.replace("-", "_")
        except Exception as e:
            logger.warning("LLM task name generation failed: %s", e)

        ascii_words = [
            w.lower() for w in re.findall(r'[a-zA-Z]+', query)
            if w.lower() not in QUERY_STOPWORDS and len(w) >= 2
        ][:4]
        if ascii_words:
            return "_".join(ascii_words)
        return "unnamed_skill"
