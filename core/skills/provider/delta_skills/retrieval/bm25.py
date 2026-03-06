
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    import jieba
from rank_bm25 import BM25Okapi

from core.config.logging import get_logger
from .stopwords import ALL_STOPWORDS as _ALL_STOPWORDS
from ..schema import Skill

logger = get_logger(__name__)

jieba.setLogLevel(jieba.logging.WARNING)


class BM25Index:

    _IDF_FLOOR = 0.1

    def __init__(self):
        self._corpus_tokens: List[List[str]] = []
        self._skill_names: List[str] = []
        self._bm25: Optional[BM25Okapi] = None

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None and len(self._skill_names) > 0

    def _create_bm25(self, corpus_tokens: List[List[str]]) -> BM25Okapi:
        bm25 = BM25Okapi(corpus_tokens, epsilon=0)
        patched = 0
        for word, idf_val in bm25.idf.items():
            if idf_val < self._IDF_FLOOR:
                bm25.idf[word] = self._IDF_FLOOR
                patched += 1
        if patched > 0:
            logger.debug("BM25: clamped %d negative/zero IDF values (corpus=%d)", patched, len(corpus_tokens))
        return bm25

    def build(self, skills: Dict[str, Skill]) -> None:
        self._corpus_tokens = []
        self._skill_names = []

        for name, skill in skills.items():
            tokens = self._tokenize(skill.to_embedding_text())
            self._corpus_tokens.append(tokens)
            self._skill_names.append(name)

        if self._corpus_tokens:
            self._bm25 = self._create_bm25(self._corpus_tokens)
            logger.info("BM25 index built with %d documents", len(self._skill_names))
        else:
            self._bm25 = None
            logger.warning("BM25 index built with 0 documents")

    def add(self, skill: Skill) -> None:
        tokens = self._tokenize(skill.to_embedding_text())

        if skill.name in self._skill_names:
            idx = self._skill_names.index(skill.name)
            self._corpus_tokens[idx] = tokens
        else:
            self._corpus_tokens.append(tokens)
            self._skill_names.append(skill.name)

        self._bm25 = self._create_bm25(self._corpus_tokens)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if not self.is_built:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        scored_pairs = sorted(
            zip(self._skill_names, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(name, score) for name, score in scored_pairs[:k] if score > 0]

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        lower = text.lower()
        raw_tokens = list(jieba.cut_for_search(lower))

        result: List[str] = []
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue

            if re.search(r"[a-zA-Z]", token):
                for st in re.split(r"[_\-.\s/|:,;()\[\]{}]+", token):
                    st = st.strip()
                    if len(st) >= 2 and st not in _ALL_STOPWORDS:
                        result.append(st)
            else:
                if len(token) >= 2 and token not in _ALL_STOPWORDS:
                    if re.search(r"[\u4e00-\u9fff]", token):
                        result.append(token)

        return result
