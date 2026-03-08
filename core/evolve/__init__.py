"""Evolve loop — self-evolution benchmark integration for Memento-S."""

from __future__ import annotations

from core.llm import LLM

_evolve_llm: LLM | None = None


def get_evolve_llm(model: str | None = None) -> LLM:
    """获取 evolve 模块专用的 LLM 实例。"""
    global _evolve_llm
    if _evolve_llm is None or model:
        _evolve_llm = LLM(model=model)
    return _evolve_llm
