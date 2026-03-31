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

    history_load_limit: int = 20
    """load_history() 从 DB 读取的最大条数（取代硬编码 80）。"""

    recent_rounds_keep: int = 3
    """两层窗口中保留原文的最近对话轮数。"""

    history_budget_ratio: float = 0.5
    """历史占 input_budget 的上限比例。"""

    embedding_enabled: bool = False
    """是否启用 embedding 语义相关性过滤（需要 embedding 服务可用）。"""

    memory_enabled: bool = False
    """是否启用长期记忆（MEMORY.md + daily notes）。"""

    max_memory_prompt_chars: int = 2000
    """注入 system prompt 的 memory 内容最大字符数。"""

    daily_notes_show_days: int = 3
    """daily notes 回看天数。"""

    artifact_fold_char_limit: int = 4000
    """tool result 字符数超过此阈值时，存为 artifact 文件，prompt 里只留 ref + preview。"""

    artifact_fold_line_limit: int = 120
    """tool result 行数超过此阈值时，同样触发 artifact fold。"""

    artifact_preview_max_lines: int = 5
    """artifact preview 保留的最大行数。"""

    artifact_preview_max_chars: int = 500
    """artifact preview 保留的最大字符数。"""

    bounded_prompt_enabled: bool = True
    """启用 bounded prompt 模式：从 block 视图组装 prompt，不再依赖 DB 全量历史。"""

    bounded_recent_events_k: int = 8
    """bounded 模式下，active block 保留最近 K 轮（每轮 = assistant + tool_call/result）注入 prompt。"""

    bounded_max_ref_previews: int = 3
    """bounded 模式下，runtime_state.recent_refs 最多注入几条 artifact preview。"""

    bounded_past_blocks_k: int = 2
    """bounded 模式下，注入最近 K 个 sealed block 的事件作为跨轮上下文。
    tool_result 会被 slim，tool_result_ref 原样保留。"""

    block_compact_threshold: int = 16
    """active block 的 event 数超过此阈值时，触发 compact_old_events。"""

    def __post_init__(self) -> None:
        for name in (
            "compaction_trigger_ratio",
            "compress_threshold_ratio",
            "summary_ratio",
            "history_budget_ratio",
        ):
            val = getattr(self, name)
            if not (0.0 < val < 1.0):
                raise ValueError(f"{name} must be in (0, 1), got {val}")
        if self.history_load_limit < 1:
            raise ValueError(f"history_load_limit must be >= 1, got {self.history_load_limit}")
        if self.recent_rounds_keep < 0:
            raise ValueError(f"recent_rounds_keep must be >= 0, got {self.recent_rounds_keep}")
