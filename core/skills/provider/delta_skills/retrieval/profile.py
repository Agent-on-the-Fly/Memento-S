
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from core.config import g_settings
from core.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HybridProfile:
    name: str
    query_instruction: str
    rrf_k: int
    min_score: float
    weights_short: Tuple[float, float]
    weights_mid: Tuple[float, float]
    weights_long: Tuple[float, float]
    small_corpus_k: int = 15
    small_corpus_threshold: int = 50


_PROFILES: Dict[str, HybridProfile] = {
    "bge-m3": HybridProfile(
        name="bge-m3",
        query_instruction="Represent this sentence for searching relevant passages: ",
        rrf_k=60,
        min_score=0.012,
        weights_short=(1.4, 0.6),
        weights_mid=(1.0, 1.0),
        weights_long=(0.7, 1.3),
    ),
    "qwen3": HybridProfile(
        name="qwen3",
        query_instruction=(
            "Instruct: Given a user query, retrieve relevant skill "
            "descriptions that match the query\nQuery:"
        ),
        rrf_k=20,
        min_score=0.012,
        weights_short=(0.4, 1.6),
        weights_mid=(0.3, 1.7),
        weights_long=(0.2, 1.8),
    ),
}

_PROFILE_ALIASES: Dict[str, str] = {
    "auto": "bge-m3",
    "BAAI/bge-m3": "bge-m3",
    "memento-qwen": "qwen3",
}

_DEFAULT_PROFILE = "bge-m3"


def get_hybrid_profile(model_name: str = None) -> HybridProfile:
    if model_name is None:
        model_name = g_settings.embedding_model

    key = model_name.lower().strip()

    if key in _PROFILES:
        return _PROFILES[key]

    if key in _PROFILE_ALIASES:
        return _PROFILES[_PROFILE_ALIASES[key]]

    for profile_key in _PROFILES:
        if profile_key in key:
            logger.info("Fuzzy matched profile '%s' for model '%s'", profile_key, model_name)
            return _PROFILES[profile_key]

    logger.warning("No HybridProfile for model '%s', falling back to '%s'", model_name, _DEFAULT_PROFILE)
    return _PROFILES[_DEFAULT_PROFILE]
