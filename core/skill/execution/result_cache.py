# SPDX-License-Identifier: Apache-2.0
"""ResultCache — 纯内存的结构化结果缓存。

设计原则：
- 不落文件，完全在内存中
- 只在 skill 执行期间存活
- 不感知具体数据类型，只提供通用的 KV + 列表存储
- 用于 outcome 输出时的结构化汇总

语义分层：
- Observation（观察层）: 每次 tool call 的原始结果
- ResultCache（结果缓存层）: 从 observation 中提取的结构化数据（汇总用）
- Context（上下文层）: 传递给 LLM 的处理后上下文
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResultCache:
    """纯内存的结构化结果缓存。

    工具通过 register() 方法注册结果，用于生成结构化的 outcome 汇总。
    """
    # 标量存储
    _values: dict[str, Any] = field(default_factory=dict)

    # 列表存储
    _lists: dict[str, list[Any]] = field(default_factory=dict)

    # 注册历史
    _registry: list[dict[str, str]] = field(default_factory=list)

    # 配置
    max_list_items: int = 1000

    def register(
        self,
        key: str,
        value: Any,
        *,
        list_item: bool = False,
    ) -> None:
        """注册结果。

        Args:
            key: 结果的唯一标识符
            value: 结果值
            list_item: 是否作为列表项追加
        """
        entry: dict[str, str] = {"key": key, "action": None}

        if list_item:
            if key not in self._lists:
                self._lists[key] = []

            # 去重
            val_key = self._make_key(value)
            existing_keys = [self._make_key(v) for v in self._lists[key]]
            if val_key in existing_keys:
                entry["action"] = "skip_duplicate"
            elif len(self._lists[key]) < self.max_list_items:
                self._lists[key].append(value)
                entry["action"] = "append"
            else:
                entry["action"] = "skip_full"
        else:
            self._values[key] = value
            entry["action"] = "store"

        self._registry.append(entry)

    @staticmethod
    def _make_key(value: Any) -> str:
        """生成去重键"""
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        return str(value)

    def get(self, key: str) -> Any | None:
        """获取值"""
        return self._values.get(key)

    def get_list(self, key: str) -> list[Any]:
        """获取列表"""
        return self._lists.get(key, [])

    def has(self, key: str) -> bool:
        """检查是否存在"""
        return key in self._values or key in self._lists

    def count(self, key: str) -> int | None:
        """获取列表长度"""
        if key in self._lists:
            return len(self._lists[key])
        return None

    def to_prompt_section(self) -> str:
        """生成注入 prompt 的摘要（概览）"""
        lines = ["## Result Cache"]

        if self._lists:
            lines.append("\nCached lists:")
            for key, items in self._lists.items():
                lines.append(f"- {key}: {len(items)} items")

        if self._values:
            lines.append("\nCached values:")
            for key, value in self._values.items():
                if not isinstance(value, (dict, list)):
                    lines.append(f"- {key}: {str(value)[:100]}")

        if not self._lists and not self._values:
            return "## Result Cache\n(no results cached yet)"

        return "\n".join(lines)

    def to_structured_output(self) -> dict[str, Any]:
        """生成结构化输出（用于 outcome）"""
        return {
            "lists": {k: len(v) for k, v in self._lists.items()},
            "values": {k: v for k, v in self._values.items() if not isinstance(v, (dict, list))},
            "registry_count": len(self._registry),
        }
