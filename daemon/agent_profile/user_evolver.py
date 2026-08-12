# SPDX-License-Identifier: Apache-2.0
"""UserEvolver — USER.md evolution engine via dialectic multi-pass reasoning.

Pass 0: Observation — cold start, comprehensive observation
Pass 1: Blind-spot critique — conditional review of Pass 0
Pass 2: Synthesis — final JSON output of sectioned atomic facts
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from utils.logger import get_logger

from daemon.agent_profile.constants import FACT_MAX_CHARS
from daemon.agent_profile.user_prompts import (
    PASS0_SYSTEM,
    PASS1_SYSTEM,
    pass0_user,
    pass1_user,
    pass2_system,
    pass2_user,
)

if TYPE_CHECKING:
    from middleware.llm.llm_client import LLMClient

logger = get_logger(__name__)

_VALID_SECTIONS = frozenset([
    "Identity & Preferences",
    "Communication Style",
    "Expertise & Background",
    "Current Goals & Context",
    "Agreements & Corrections",
])


def _signal_sufficient(result: str) -> bool:
    """Check whether Pass 0 signal is strong enough to skip Pass 1."""
    if not result or len(result.strip()) < 100:
        return False
    if "\n" in result and (
        "##" in result
        or re.search(r"^[*-] ", result, re.MULTILINE)
        or re.search(r"^\s*\d+\. ", result, re.MULTILINE)
    ):
        return True
    return len(result.strip()) > 500


class UserEvolver:
    """Dialectic multi-pass reasoning engine — extracts sectioned user facts from conversation history."""

    def __init__(self, llm_client: "LLMClient") -> None:
        self._llm = llm_client

    async def extract(
        self,
        transcript: str,
        existing_context: str,
        max_facts: int,
    ) -> dict[str, list[str]]:
        """Run dialectic passes to extract sectioned user facts."""
        # Pass 0: observation
        pass0 = await self._pass_0(transcript)
        if not pass0:
            return {}

        # Pass 1: conditional blind-spot critique
        pass1 = ""
        if not _signal_sufficient(pass0):
            pass1 = await self._pass_1(pass0, transcript)
            if pass1:
                pass0 = pass0 + "\n\n---\n\nSupplementary analysis:\n" + pass1

        # Pass 2: synthesis
        return await self._pass_2(pass0, existing_context, max_facts)

    async def _pass_0(self, transcript: str) -> str:
        """Pass 0: cold-start observation across all dimensions."""
        return await self._call_llm(PASS0_SYSTEM, pass0_user(transcript))

    async def _pass_1(self, pass0_result: str, transcript: str) -> str:
        """Pass 1: blind-spot critique — conditionally supplement Pass 0."""
        result = await self._call_llm(PASS1_SYSTEM, pass1_user(pass0_result, transcript))
        if result and result.strip().lower() in ("no additions needed", "none", "n/a", "no supplement"):
            return ""
        return result

    async def _pass_2(
        self,
        analysis: str,
        existing_context: str,
        max_facts: int,
    ) -> dict[str, list[str]]:
        """Pass 2: synthesize into sectioned facts."""
        return await self._call_llm_json(
            pass2_system(max_facts),
            pass2_user(analysis, existing_context, max_facts),
            max_facts,
        )

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Send request to LLM."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            result = await self._llm.async_chat(messages)
            content = result.content if hasattr(result, "content") else str(result)
            return content.strip()
        except Exception as e:
            logger.warning("[UserEvolver] LLM call failed: {}", e)
            return ""

    async def _call_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_facts: int,
    ) -> dict[str, list[str]]:
        """Send request to LLM and parse JSON result into {section: [fact, ...]}."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            result = await self._llm.async_chat(messages)
            raw = result.content if hasattr(result, "content") else str(result)
            raw = raw.strip()

            code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if code_block_match:
                raw = code_block_match.group(1).strip()

            data: dict = json.loads(raw)
            if not isinstance(data, dict):
                return {}
            total = 0
            out: dict[str, list[str]] = {}
            for section, facts in data.items():
                if section not in _VALID_SECTIONS:
                    continue
                if not isinstance(facts, list):
                    continue
                trimmed = [str(f)[:FACT_MAX_CHARS].strip() for f in facts if f]
                if trimmed:
                    budget = max_facts - total
                    if budget <= 0:
                        break
                    chosen = trimmed[:budget]
                    out[section] = chosen
                    total += len(chosen)
            return out
        except json.JSONDecodeError:
            logger.debug("[UserEvolver] JSON parse failed: {}", raw[:200])
            return {}
        except Exception as e:
            logger.warning("[UserEvolver] LLM JSON call failed: {}", e)
            return {}
