# SPDX-License-Identifier: Apache-2.0
"""Information saturation detector for search results.

Analyzes content quality to detect when information gathering has reached diminishing returns.
Uses tool-agnostic metrics (content fingerprint diversity, length changes, Jaccard similarity)
so it works for any tool that returns structured or text content, not just search_web/fetch_webpage.
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
    content_fingerprint: str = ""
    length: int = 0


class InfoSaturationDetector:
    """Detects when information gathering has reached diminishing returns.

    All signals are tool-agnostic:
    - Content fingerprint diversity (MD5 of normalized content)
    - Content length delta
    - Jaccard word similarity between consecutive results
    - Query variation (for tools that accept queries)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        entity_overlap_threshold: float = 0.8,
        min_results_for_analysis: int = 3,
    ):
        """
        Args:
            similarity_threshold: Content similarity threshold (0-1). Above this = saturated.
            entity_overlap_threshold: Not used (kept for API compat).
            min_results_for_analysis: Minimum results before analyzing.
        """
        self.similarity_threshold = similarity_threshold
        self.entity_overlap_threshold = entity_overlap_threshold
        self.min_results_for_analysis = min_results_for_analysis
        self.history: list[SearchResult] = []
        self.content_fingerprints: set[str] = set()

    def record(
        self,
        tool_name: str,
        query: str,
        content: str,
        turn: int,
    ) -> None:
        """Record a tool result for saturation analysis."""
        fingerprint = self._content_fingerprint(content)
        self.content_fingerprints.add(fingerprint)

        result = SearchResult(
            tool_name=tool_name,
            query=query,
            content=content[:2000],
            turn=turn,
            content_fingerprint=fingerprint,
            length=len(content),
        )
        self.history.append(result)

    def check_saturation(self) -> dict[str, Any] | None:
        """Check if information saturation has been reached.

        Returns:
            Saturation info dict if saturated, None otherwise.
        """
        if len(self.history) < self.min_results_for_analysis:
            return None

        recent = self.history[-3:]

        # Test 1: High content similarity between consecutive results (tool-agnostic)
        similarity_score = self._calculate_similarity(recent)
        if similarity_score > self.similarity_threshold:
            return {
                "type": "content_similarity",
                "severity": "high",
                "score": similarity_score,
                "message": (
                    f"Recent tool results are {similarity_score:.0%} similar to previous ones. "
                    "The same information is being returned repeatedly. "
                    "Proceed to synthesize and create the deliverable."
                ),
                "recommendation": "synthesize_and_create",
            }

        # Test 2: No new content fingerprints (tool-agnostic — replaces entity extraction)
        recent_fingerprints = {r.content_fingerprint for r in recent}
        total_fingerprints = len(self.content_fingerprints)
        new_fingerprints = len(recent_fingerprints - (self.content_fingerprints - recent_fingerprints))
        if total_fingerprints > 0 and new_fingerprints == 0:
            return {
                "type": "no_new_content",
                "severity": "medium",
                "message": (
                    "No new unique content has been discovered in recent tool calls. "
                    "The output is identical or near-identical to previous results. "
                    "Stop gathering and start creating."
                ),
                "recommendation": "proceed_to_creation",
            }

        # Test 3: Diminishing length returns (tool-agnostic)
        if self._check_diminishing_length(recent):
            return {
                "type": "diminishing_returns",
                "severity": "medium",
                "message": (
                    "Tool output length is shrinking across recent calls, "
                    "indicating information exhaustion. Use what you have to create the deliverable."
                ),
                "recommendation": "proceed_to_creation",
            }

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get information gathering statistics."""
        if not self.history:
            return {}
        return {
            "total_calls": len(self.history),
            "unique_content_pieces": len(self.content_fingerprints),
        }

    def _content_fingerprint(self, content: str) -> str:
        """Generate a fingerprint for content similarity comparison."""
        normalized = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", content.lower())
        return hashlib.md5(normalized[:200].encode()).hexdigest()[:16]

    def _calculate_similarity(self, results: list[SearchResult]) -> float:
        """Calculate average Jaccard similarity between consecutive results."""
        if len(results) < 2:
            return 0.0
        similarities = []
        for i in range(len(results) - 1):
            sim = self._jaccard_similarity(results[i].content, results[i + 1].content)
            similarities.append(sim)
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(self._extract_words(text1))
        words2 = set(self._extract_words(text2))
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _extract_words(self, text: str) -> list[str]:
        """Extract significant words from text."""
        chinese = re.findall(r"[\u4e00-\u9fa5]{2,}", text)
        english = re.findall(r"[a-zA-Z]{3,}", text)
        return chinese + english

    def _check_diminishing_length(self, results: list[SearchResult]) -> bool:
        """Check if content length is consistently shrinking (diminishing returns)."""
        if len(results) < 3:
            return False
        lengths = [r.length for r in results]
        decreases = sum(1 for i in range(1, len(lengths)) if lengths[i] < lengths[i - 1])
        return decreases >= len(lengths) - 1
