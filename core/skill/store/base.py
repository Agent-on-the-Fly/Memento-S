"""Skill 存储抽象基类 - 定义通用接口"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Set


class SkillStorage(ABC):
    """Skill 存储抽象基类 - 所有存储实现的通用接口"""

    def __init__(self) -> None:
        self._initialized = False

    @property
    def is_ready(self) -> bool:
        """存储是否已初始化就绪"""
        return self._initialized

    @abstractmethod
    async def init(self) -> None:
        """初始化存储（创建目录/连接/表等）"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭存储（释放连接等）"""
        pass

    @abstractmethod
    async def save(self, name: str, data: Any) -> None:
        """保存数据（各实现决定 data 类型）"""
        pass

    @abstractmethod
    async def load(self, name: str) -> Any | None:
        """加载数据"""
        pass

    @abstractmethod
    async def delete(self, name: str) -> bool:
        """删除数据"""
        pass

    @abstractmethod
    async def list_names(self) -> Set[str]:
        """返回所有存储的名称集合"""
        pass
