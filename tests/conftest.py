# SPDX-License-Identifier: Apache-2.0
"""Pytest 配置文件

注册自定义标记和全局 fixtures。
"""

import pytest


# These files are legacy, executable diagnostics for the pre-Conversation
# storage/provider APIs.  They perform live/manual checks and are not part of
# the current pytest contract.  Current equivalents live in the focused test
# packages under tests/ and middleware/storage/tests/.
collect_ignore = [
    "test_model_schema_consistency.py",
    "test_prompt_components.py",
    "test_search_execute_flow.py",
    "test_storage_models.py",
    "test_storage_performance.py",
    "test_storage_service.py",
]


def pytest_configure(config):
    """配置 pytest，注册自定义标记"""
    config.addinivalue_line(
        "markers", "smoke: 冒烟测试 - 快速验证核心功能"
    )
    config.addinivalue_line(
        "markers", "slow: 耗时测试 - 涉及 LLM API 调用或大量数据处理"
    )
    config.addinivalue_line(
        "markers", "integration: 集成测试 - 端到端完整流程测试"
    )
