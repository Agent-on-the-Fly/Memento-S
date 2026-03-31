"""Information saturation detector for search results.

Analyzes content quality to detect when search has reached diminishing returns.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """Record of a search/fetch result."""

    tool_name: str
    query: str
    content: str
    turn: int
    entities: list[str] = field(default_factory=list)


class InfoSaturationDetector:
    """Detects when search results are no longer providing new information."""

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        entity_overlap_threshold: float = 0.8,
        min_results_for_analysis: int = 3,
    ):
        """
        Args:
            similarity_threshold: Content similarity threshold (0-1)
            entity_overlap_threshold: Entity overlap threshold (0-1)
            min_results_for_analysis: Minimum results before analyzing
        """
        self.similarity_threshold = similarity_threshold
        self.entity_overlap_threshold = entity_overlap_threshold
        self.min_results_for_analysis = min_results_for_analysis
        self.history: list[SearchResult] = []
        self.entity_registry: dict[str, int] = {}  # entity -> count
        self.content_fingerprints: set[str] = set()

    def record(
        self,
        tool_name: str,
        query: str,
        content: str,
        turn: int,
    ) -> None:
        """Record a search/fetch result."""
        # Extract entities (URLs, names, case names, etc.)
        entities = self._extract_entities(content)

        result = SearchResult(
            tool_name=tool_name,
            query=query,
            content=content[:2000],  # Truncate for storage
            turn=turn,
            entities=entities,
        )
        self.history.append(result)

        # Update entity registry
        for entity in entities:
            self.entity_registry[entity] = self.entity_registry.get(entity, 0) + 1

        # Store content fingerprint
        fingerprint = self._content_fingerprint(content)
        self.content_fingerprints.add(fingerprint)

    def check_saturation(self) -> dict[str, Any] | None:
        """Check if information saturation has been reached.

        Returns:
            Saturation info dict if saturated, None otherwise
        """
        if len(self.history) < self.min_results_for_analysis:
            return None

        recent = self.history[-3:]  # Check last 3 results

        # Test 1: High content similarity between consecutive results
        similarity_score = self._calculate_similarity(recent)
        if similarity_score > self.similarity_threshold:
            return {
                "type": "content_similarity",
                "severity": "high",
                "score": similarity_score,
                "message": (
                    f"Search results are {similarity_score:.0%} similar to previous results. "
                    "You're finding the same information from different sources. "
                    "Current knowledge is sufficient to proceed with deliverable creation."
                ),
                "recommendation": "synthesize_and_create",
            }

        # Test 2: Low new entity discovery rate
        new_entity_ratio = self._calculate_new_entity_ratio(recent)
        if new_entity_ratio < 0.2:  # Less than 20% new entities
            return {
                "type": "low_new_entities",
                "severity": "medium",
                "new_entity_ratio": new_entity_ratio,
                "message": (
                    f"Only {new_entity_ratio:.0%} of entities in recent results are new. "
                    "Information discovery has reached diminishing returns. "
                    "You have gathered sufficient unique cases and examples."
                ),
                "recommendation": "proceed_to_creation",
            }

        # Test 3: Repeated search queries
        query_variation = self._check_query_variation()
        if query_variation < 0.3:  # Very similar queries
            return {
                "type": "repetitive_queries",
                "severity": "medium",
                "message": (
                    "Search queries are becoming repetitive with only minor keyword changes. "
                    "This indicates topic exhaustion. "
                    "Use the information already collected to create the deliverable."
                ),
                "recommendation": "use_existing_info",
            }

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get information gathering statistics."""
        if not self.history:
            return {}

        total_entities = len(self.entity_registry)
        repeated_entities = sum(
            1 for count in self.entity_registry.values() if count > 1
        )

        return {
            "total_searches": len(self.history),
            "unique_entities": total_entities,
            "repeated_entities": repeated_entities,
            "entity_reuse_rate": repeated_entities / total_entities
            if total_entities > 0
            else 0,
            "unique_content_pieces": len(self.content_fingerprints),
        }

    def _extract_entities(self, content: str) -> list[str]:
        """Extract key entities from content."""
        entities = []

        # URLs
        urls = re.findall(r'https?://[^\s\]\)"\'>]+', content)
        entities.extend(urls[:5])  # Top 5 URLs

        # Chinese case names (XX诉XX案)
        cases = re.findall(r"[\u4e00-\u9fa5]{2,}诉[\u4e00-\u9fa5]{2,}案", content)
        entities.extend(cases[:3])

        # Company names (limited liability companies)
        companies = re.findall(r"[\u4e00-\u9fa5]{2,}(?:公司|集团|企业)", content)
        entities.extend(companies[:5])

        # Years (2020-2025)
        years = re.findall(r"202[0-5]", content)
        entities.extend(years[:3])

        return list(set(entities))  # Deduplicate

    def _content_fingerprint(self, content: str) -> str:
        """Generate a fingerprint for content similarity comparison."""
        # Normalize: lowercase, remove punctuation, keep Chinese and English
        normalized = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", content.lower())
        # Take first 200 chars as fingerprint
        return hashlib.md5(normalized[:200].encode()).hexdigest()[:16]

    def _calculate_similarity(self, results: list[SearchResult]) -> float:
        """Calculate average content similarity between consecutive results."""
        if len(results) < 2:
            return 0.0

        similarities = []
        for i in range(len(results) - 1):
            sim = self._jaccard_similarity(results[i].content, results[i + 1].content)
            similarities.append(sim)

        return sum(similarities) / len(similarities)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        # Extract words/bigrams
        words1 = set(self._extract_words(text1))
        words2 = set(self._extract_words(text2))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _extract_words(self, text: str) -> list[str]:
        """Extract significant words from text."""
        # Chinese characters
        chinese = re.findall(r"[\u4e00-\u9fa5]{2,}", text)
        # English words
        english = re.findall(r"[a-zA-Z]{3,}", text)
        return chinese + english

    def _calculate_new_entity_ratio(self, results: list[SearchResult]) -> float:
        """Calculate ratio of new entities in recent results."""
        if not results:
            return 1.0

        all_recent_entities = []
        for result in results:
            all_recent_entities.extend(result.entities)

        if not all_recent_entities:
            return 1.0

        # Count how many are new (first appearance in recent results)
        new_count = 0
        seen = set()

        for entity in all_recent_entities:
            if entity not in self.entity_registry or self.entity_registry[entity] <= 1:
                if entity not in seen:
                    new_count += 1
                    seen.add(entity)

        return new_count / len(set(all_recent_entities))

    def _check_query_variation(self) -> float:
        """Check how much search queries have varied."""
        if len(self.history) < 3:
            return 1.0

        recent_queries = [r.query for r in self.history[-5:]]
        if not recent_queries:
            return 1.0

        # Calculate pairwise similarity
        similarities = []
        for i in range(len(recent_queries)):
            for j in range(i + 1, len(recent_queries)):
                sim = self._query_similarity(recent_queries[i], recent_queries[j])
                similarities.append(sim)

        if not similarities:
            return 1.0

        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity  # Return variation (dissimilarity)

    def _query_similarity(self, q1: str, q2: str) -> float:
        """Calculate similarity between two search queries."""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0
