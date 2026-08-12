# SPDX-License-Identifier: Apache-2.0
"""MemoryConsolidationEngine — 唯一的整合引擎，只负责"怎么整合"。

由 Dream、Agent 或独立后台循环触发。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.logger import get_logger

from middleware.config.schemas.config_models import MemoryConsolidationConfig

from .prompts import DEEP_CONSOLIDATION_PROMPT, MAX_ENTRYPOINT_LINES, QUICK_CONSOLIDATION_PROMPT
from .result_applier import ResultApplier

logger = get_logger(__name__)

LOCK_FILE = ".consolidation.lock"
LOCK_TIMEOUT = 3600  # seconds


@dataclass
class ConsolidationContext:
    """整合所需的上下文数据。"""
    index_content: str
    staging_content: str
    topics: list[dict[str, str]]
    mode: str = "quick"


class MemoryConsolidationEngine:
    """唯一的整合引擎 — 只负责"怎么整合"，不关心"谁触发"。

    使用方式:
        engine = MemoryConsolidationEngine(
            memory_dir=path,
            llm_provider=llm,
            config=MemoryConsolidationConfig(...),
        )

        # 触发检查
        if engine.check_should_consolidate():
            await engine.run()

        # 或直接触发
        await engine.quick_run()
        await engine.deep_run()
    """

    def __init__(
        self,
        memory: "LongTermMemory | Path",
        config: MemoryConsolidationConfig,
    ) -> None:
        from pathlib import Path

        if isinstance(memory, Path):
            self._memory_dir = memory
            from infra.memory.impl.long_term_memory import LongTermMemory
            self._memory = LongTermMemory(memory, model="")
        else:
            self._memory_dir = memory._dir
            self._memory = memory

        self._result_applier = ResultApplier(self._memory_dir)
        self._config = config
        self.__llm_client = None  # lazy load

    @property
    def _llm(self):
        """Lazy load LLM client."""
        if self.__llm_client is None:
            from middleware.llm.llm_client import chat_completions_async
            self.__llm_client = chat_completions_async
        return self.__llm_client

    @_llm.setter
    def _llm(self, value):
        self.__llm_client = value

    # ---- 并发锁 ----

    def _lock_path(self) -> Path:
        return self._memory_dir / LOCK_FILE

    def _acquire_lock(self) -> bool:
        """Filesystem lock to prevent concurrent consolidation runs."""
        lock_path = self._lock_path()
        try:
            if lock_path.exists():
                lock_age = time.time() - lock_path.stat().st_mtime
                if lock_age < LOCK_TIMEOUT:
                    return False
                lock_path.unlink(missing_ok=True)
            lock_path.write_text(str(time.time()), encoding="utf-8")
            return True
        except OSError:
            return False

    def _release_lock(self) -> None:
        try:
            self._lock_path().unlink(missing_ok=True)
        except OSError:
            pass

    # ---- 触发检查 ----

    def check_should_consolidate(self) -> bool:
        """检查是否应触发整合（基于累积量阈值）。"""
        staging = self._memory.get_staging_content()
        if not staging.strip():
            return False

        session_count = staging.count("## Session ")
        if session_count >= self._config.min_staging_sessions:
            return True

        if len(staging.encode("utf-8")) >= self._config.min_staging_bytes:
            return True

        return False

    # ---- 整合执行 ----

    async def run(self, context: ConsolidationContext) -> bool:
        """根据 context.mode 执行对应整合。"""
        if context.mode == "deep":
            return await self.deep_run(context)
        return await self.quick_run(context)

    async def quick_run(self, context: ConsolidationContext | None = None) -> bool:
        """轻量整合：只看 staging 内容，快速生成 topic 更新。"""
        if not self._acquire_lock():
            logger.debug("MemoryConsolidationEngine: consolidation already running, skipping quick_run")
            return False
        try:
            if context is None:
                staging = self._memory.get_staging_content()
                if not staging.strip():
                    logger.debug("MemoryConsolidationEngine: no staging content, skipping")
                    return False

                index = self._memory.get_index_content()
                context = ConsolidationContext(
                    index_content=index or "(empty)",
                    staging_content=staging,
                    topics=[],
                    mode="quick",
                )

            prompt = QUICK_CONSOLIDATION_PROMPT.format(
                staging_content=context.staging_content,
                index_content=context.index_content,
                max_lines=80,
            )

            raw = await self._llm(
                system="You are a memory consolidator. Output valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._config.max_tokens_per_call,
            )

            result = self._parse_result(raw)
            if result is None:
                return False

            changed = self._apply_result(result)
            self._memory.clear_staging()
            self._last_consolidation_time = time.monotonic()

            topic_count = (
                len(result.get("updated_topics", []))
                + len(result.get("new_topics", []))
            )
            logger.info(
                "MemoryConsolidationEngine.quick_run: {} topics changed, {} deleted",
                topic_count,
                len(result.get("deleted_topics", [])),
            )
            return True

        except Exception:
            logger.opt(exception=True).warning("MemoryConsolidationEngine.quick_run failed")
            return False
        finally:
            self._release_lock()

    async def deep_run(self, context: ConsolidationContext | None = None) -> bool:
        """深度整合：读取所有 topic + staging + index，全量 LLM 分析。"""
        if not self._acquire_lock():
            logger.debug("MemoryConsolidationEngine: consolidation already running, skipping deep_run")
            return False
        try:
            if context is None:
                staging = self._memory.get_staging_content()
                if not staging.strip():
                    logger.debug("MemoryConsolidationEngine: no staging content, skipping")
                    return False

                context = ConsolidationContext(
                    index_content=self._memory.get_index_content(),
                    staging_content=staging,
                    topics=self._memory.list_topics_with_content(),
                    mode="deep",
                )

            topics_text = ""
            for t in context.topics:
                topics_text += f"\n### {t['slug']}\n```\n{t['content'][:2000]}\n```\n"

            prompt = DEEP_CONSOLIDATION_PROMPT.format(
                index=context.index_content or "(empty)",
                topics=topics_text or "(no existing topics)",
                staging=context.staging_content,
                max_lines=MAX_ENTRYPOINT_LINES,
            )

            raw = await self._llm(
                system="You are a memory consolidator. Output valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
            )

            result = self._parse_result(raw)
            if result is None:
                return False

            changed = self._apply_result(result)
            self._memory.clear_staging()
            self._last_consolidation_time = time.monotonic()

            topic_count = (
                len(result.get("updated_topics", []))
                + len(result.get("new_topics", []))
            )
            logger.info(
                "MemoryConsolidationEngine.deep_run: {} topics changed, {} deleted",
                topic_count,
                len(result.get("deleted_topics", [])),
            )
            return True

        except Exception:
            logger.opt(exception=True).warning("MemoryConsolidationEngine.deep_run failed")
            return False
        finally:
            self._release_lock()

    # ---- 内部工具 ----

    def _apply_result(self, result: dict[str, Any]) -> int:
        """应用整合结果，返回变更的 topic 数量。"""
        return self._result_applier.apply(result)

    @staticmethod
    def _parse_result(raw: str) -> dict[str, Any] | None:
        """从 LLM 响应中解析 JSON。"""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            start = 1
            end = len(lines) - 1
            if lines[0].startswith("```json"):
                start = 1
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end = i
                    break
            text = "\n".join(lines[start:end])

        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = text[brace_start:i + 1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            logger.warning("MemoryConsolidationEngine: JSON parse failed")
                            return None
        return None
