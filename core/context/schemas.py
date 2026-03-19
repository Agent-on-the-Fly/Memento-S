"""Context 模块配置。

ContextConfig 是 ContextManager 的唯一配置依赖。
compress / compact 的 token 阈值在 init_budget() 中
由 input_budget * ratio 直接派生。

input_budget = LLMProfile.context_window - LLMProfile.max_tokens
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextConfig:
    """ContextManager 可配置参数。

    由 AgentConfig.context 持有，在 agent 创建 ContextManager 时传入。
    所有 ratio 均相对于 input_budget（= context_window - max_tokens）计算。
    """

    compaction_trigger_ratio: float = 0.7
    """total_tokens > input_budget * ratio 时触发 compact。"""

    compress_threshold_ratio: float = 0.5
    """单条消息 > input_budget * ratio 时触发 compress。"""

    summary_ratio: float = 0.15
    """compress / compact 摘要输出上限 = input_budget * ratio。"""
