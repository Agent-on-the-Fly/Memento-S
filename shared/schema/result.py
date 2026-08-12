# SPDX-License-Identifier: Apache-2.0
"""统一的 Result 类型 — 用于跨层错误传播。

提供 Ok / Err 两种结果变体，模仿 Rust 的 Result 模式。
使用方式：
    from shared.schema import Result, Ok, Err

    def fetch_data() -> Result[str, str]:
        if success:
            return Ok("data")
        else:
            return Err("error message")

    result = fetch_data()
    if result.is_ok():
        print(result.value)
    else:
        print(result.error)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Generic, Union

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """成功结果。"""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        """解包成功值（若为 Err 则抛异常）。"""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """解包成功值，失败则返回默认值。"""
        return self.value

    def map(self, fn, **kwargs) -> Ok:
        """对成功值应用函数变换。"""
        return Ok(fn(self.value, **kwargs))


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """错误结果。"""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> T:
        """解包成功值（若为 Err 则抛异常）。"""
        raise ValueError(f"Attempted to unwrap Err: {self.error}")

    def unwrap_or(self, default: T) -> T:
        """解包成功值，失败则返回默认值。"""
        return default

    def map(self, fn, **kwargs) -> Err:
        """对成功值应用函数变换（Err 直接透传）。"""
        return self


# Union 别名（Python 3.14 原生支持 | 语法，但为兼容性保留 Union）
Result = Ok[T] | Err[E]


__all__ = ["Result", "Ok", "Err"]
