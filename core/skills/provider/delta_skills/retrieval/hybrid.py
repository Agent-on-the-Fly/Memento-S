
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.config import g_settings
from core.config.logging import get_logger
from .bm25 import BM25Index
from .profile import HybridProfile, get_hybrid_profile
from .reranker import CrossEncoderReranker
from ..schema import Skill

logger = get_logger(__name__)


@dataclass
class ScoredSkill:
    skill: Skill
    score: float
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    reranker_score: Optional[float] = None


class HybridRetriever:

    _embedding_cache: OrderedDict = OrderedDict()
    _embedding_cache_lock: threading.Lock = threading.Lock()
    _embedding_cache_max: int = 128

    def __init__(
        self,
        bm25_index: BM25Index,
        chroma_collection,
        skill_cache: Dict[str, Skill],
        reranker: Optional[CrossEncoderReranker] = None,
        profile: Optional[HybridProfile] = None,
    ):
        self.bm25_index = bm25_index
        self.chroma_collection = chroma_collection
        self.skill_cache = skill_cache
        self.reranker = reranker
        self.profile = profile or get_hybrid_profile()
        logger.debug(
            "HybridRetriever profile='%s': K=%d, weights=(%s/%s/%s)",
            self.profile.name, self.profile.rrf_k,
            self.profile.weights_short, self.profile.weights_mid, self.profile.weights_long,
        )

    @classmethod
    def invalidate_cache(cls):
        with cls._embedding_cache_lock:
            cls._embedding_cache.clear()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float = None,
        extra_queries: List[str] = None,
    ) -> List[ScoredSkill]:
        if min_score is None:
            min_score = self.profile.min_score

        bm25_results = self._bm25_recall(query, k=k * 2)
        embedding_results = self._embedding_recall(query, k=k * 2)

        logger.debug("BM25 recall: %d, Embedding recall: %d", len(bm25_results), len(embedding_results))

        extra_emb_lists: List[List[Tuple[str, float]]] = []
        if extra_queries:
            for eq in extra_queries:
                if eq and eq != query:
                    extra = self._embedding_recall(eq, k=k * 2)
                    if extra:
                        extra_emb_lists.append(extra)

        bm25_scores = dict(bm25_results)
        emb_scores = dict(embedding_results)
        for extra_list in extra_emb_lists:
            for name, score in extra_list:
                emb_scores[name] = max(emb_scores.get(name, 0.0), score)

        bm25_weight, emb_weight = self._get_dynamic_weights(query)
        merged_emb = sorted(emb_scores.items(), key=lambda x: x[1], reverse=True)

        fused = self._score_aware_rrf(
            results_lists=[bm25_results, merged_emb],
            weights=[bm25_weight, emb_weight],
            k=k * 3,
        )

        candidates: List[ScoredSkill] = []
        for name, rrf_score in fused:
            if rrf_score < min_score:
                continue
            if name not in self.skill_cache:
                continue
            candidates.append(ScoredSkill(
                skill=self.skill_cache[name],
                score=rrf_score,
                bm25_score=bm25_scores.get(name, 0.0),
                embedding_score=emb_scores.get(name, 0.0),
            ))

        if self.reranker and candidates:
            try:
                candidates = self.reranker.rerank(
                    query, candidates, min_score=g_settings.reranker_min_score,
                )
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("Reranker failed, using RRF scores: %s", e)

        results = candidates[:k]

        if results:
            best = results[0]
            logger.info(
                "Best match: '%s' (score=%.4f, bm25=%.4f, emb=%.4f, rerank=%s)",
                best.skill.name, best.score, best.bm25_score, best.embedding_score,
                f"{best.reranker_score:.4f}" if best.reranker_score is not None else "N/A",
            )
        else:
            logger.info("No results above threshold %.4f", min_score)

        return results

    def _bm25_recall(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        return self.bm25_index.search(query, k=k)

    def _embedding_recall(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        cache_key = (query, k)

        with self._embedding_cache_lock:
            if cache_key in self._embedding_cache:
                self._embedding_cache.move_to_end(cache_key)
                return self._embedding_cache[cache_key]

        prefixed_query = self.profile.query_instruction + query

        try:
            results = self.chroma_collection.query(
                query_texts=[prefixed_query],
                n_results=k,
            )
            if not results["ids"] or not results["ids"][0]:
                return []

            names = results["ids"][0]
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(names)
            scored = [(name, max(0.0, 1.0 - dist)) for name, dist in zip(names, distances)]

            with self._embedding_cache_lock:
                self._embedding_cache[cache_key] = scored
                if len(self._embedding_cache) > self._embedding_cache_max:
                    self._embedding_cache.popitem(last=False)

            return scored

        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Embedding recall failed: %s", e)
            return []

    def _get_adaptive_rrf_k(self) -> int:
        corpus_size = len(self.skill_cache)
        if corpus_size < self.profile.small_corpus_threshold:
            return self.profile.small_corpus_k
        return self.profile.rrf_k

    def _get_dynamic_weights(self, query: str) -> Tuple[float, float]:
        tokens = BM25Index._tokenize(query)
        token_count = len(tokens)

        if token_count <= 2:
            bm25_w, emb_w = self.profile.weights_short
        elif token_count <= 5:
            bm25_w, emb_w = self.profile.weights_mid
        else:
            bm25_w, emb_w = self.profile.weights_long

        logger.debug(
            "Dynamic weights [%s]: bm25=%.1f, emb=%.1f (tokens=%d)",
            self.profile.name, bm25_w, emb_w, token_count,
        )
        return bm25_w, emb_w

    def _score_aware_rrf(
        self,
        results_lists: List[List[Tuple[str, float]]],
        weights: List[float] = None,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        adaptive_k = self._get_adaptive_rrf_k()

        if weights is None:
            weights = [1.0] * len(results_lists)

        rrf_scores: Dict[str, float] = {}

        for path_idx, results in enumerate(results_lists):
            if not results:
                continue

            w = weights[path_idx] if path_idx < len(weights) else 1.0
            max_score = max(results[0][1], 1e-9)

            for rank, (name, raw_score) in enumerate(results, start=1):
                norm = raw_score / max_score
                score_factor = 0.4 + 0.6 * norm
                rrf_scores[name] = rrf_scores.get(name, 0.0) + w * score_factor / (adaptive_k + rank)

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]
