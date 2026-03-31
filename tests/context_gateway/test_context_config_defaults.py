"""ContextConfig 默认值 + 自定义覆盖测试。"""
from __future__ import annotations

from core.context.schemas import ContextConfig
from core.memento_s.schemas import AgentConfig


def test_context_config_defaults():
    """所有 ContextConfig 默认值正确。"""
    cfg = ContextConfig()

    assert cfg.compaction_trigger_ratio == 0.7
    assert cfg.compress_threshold_ratio == 0.5
    assert cfg.summary_ratio == 0.15


def test_agent_config_embeds_context_config():
    """AgentConfig 内嵌 ContextConfig。"""
    cfg = AgentConfig()

    assert isinstance(cfg.context, ContextConfig)
    assert cfg.context.compaction_trigger_ratio == 0.7


def test_context_config_custom_values():
    """ContextConfig 支持自定义覆盖。"""
    cfg = ContextConfig(
        compaction_trigger_ratio=0.5,
        compress_threshold_ratio=0.5,
        summary_ratio=0.2,
    )

    assert cfg.compaction_trigger_ratio == 0.5
    assert cfg.compress_threshold_ratio == 0.5
    assert cfg.summary_ratio == 0.2
