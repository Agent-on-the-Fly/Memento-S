
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional

from core.config import g_settings
from core.config.logging import get_logger

if TYPE_CHECKING:
    from .hybrid import ScoredSkill

logger = get_logger(__name__)


class CrossEncoderReranker:

    _model_cache: Dict[str, object] = {}
    _instance: CrossEncoderReranker | None = None

    def __init__(self):
        pass

    @classmethod
    def get_instance(cls) -> CrossEncoderReranker:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_model(self, model_name: str):
        if model_name not in self._model_cache:
            from sentence_transformers import CrossEncoder
            self._model_cache[model_name] = CrossEncoder(model_name)
            logger.info("Loaded reranker model: %s", model_name)
        return self._model_cache[model_name]

    def _select_model_name(self, query: str) -> str:
        if g_settings.reranker_model != "auto":
            return g_settings.reranker_model
        return "BAAI/bge-reranker-v2-m3"

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 500:
            return 1.0
        if x <= -500:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def rerank(
        self, query: str, scored_skills: List[ScoredSkill], min_score: float = 0.01,
    ) -> List[ScoredSkill]:
        from .hybrid import ScoredSkill as _ScoredSkill

        if not scored_skills:
            return []

        model_name = self._select_model_name(query)
        model = self._get_model(model_name)

        pairs = [(query, s.skill.to_embedding_text()) for s in scored_skills]
        raw_scores = model.predict(pairs)

        _RERANKER_WEIGHT = 0.6
        max_rrf = max((ss.score for ss in scored_skills), default=1.0)

        reranked: List[_ScoredSkill] = []
        best_candidate: Optional[_ScoredSkill] = None
        best_final_score: float = -1.0

        for ss, raw in zip(scored_skills, raw_scores):
            norm_score = self._sigmoid(float(raw))
            rrf_norm = ss.score / max_rrf
            final_score = _RERANKER_WEIGHT * norm_score + (1 - _RERANKER_WEIGHT) * rrf_norm

            entry = _ScoredSkill(
                skill=ss.skill,
                score=final_score,
                bm25_score=ss.bm25_score,
                embedding_score=ss.embedding_score,
                reranker_score=norm_score,
            )

            if final_score > best_final_score:
                best_final_score = final_score
                best_candidate = entry

            if final_score < min_score:
                logger.debug(
                    "Reranker filtered '%s': ce=%.4f, rrf=%.4f, final=%.4f < min=%.4f",
                    ss.skill.name, norm_score, rrf_norm, final_score, min_score,
                )
                continue
            reranked.append(entry)
        if not reranked and best_candidate is not None:
            logger.info(
                "Reranker: all filtered, keeping best '%s' (final=%.4f)",
                best_candidate.skill.name, best_final_score,
            )
            reranked.append(best_candidate)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked
