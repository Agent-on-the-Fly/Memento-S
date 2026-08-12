# SPDX-License-Identifier: Apache-2.0
"""Constants for AgentProfileEvolver."""

# USER.md 定时进化间隔
EVOLVE_INTERVAL_SECONDS = 3600  # 每小时定时检查一次

# SOUL.md 定时进化间隔（更保守，跨会话验证需要时间积累）
EVOLVE_SOUL_INTERVAL_SECONDS = 7200  # 每 2 小时一次

# USER.md 每条事实的最大字符数
FACT_MAX_CHARS = 120

# 单次进化最多追加的事实数（会话结束时触发）
MAX_FACTS_PER_SESSION = 3

# 每次进化最多追加的事实数（定时触发，更保守）
MAX_FACTS_PER_TIMED = 2

# 消息截断长度
MSG_TRUNCATE_CHARS = 400

# 输入 LLM 的最大字符数（防止上下文溢出）
INPUT_MAX_CHARS = 8000
