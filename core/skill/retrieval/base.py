"""retrieval/base.py — 召回基类定义

定义召回策略的抽象基类，统一召回接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schema import RecallCandidate


class BaseRecall(ABC):
    """召回策略基类

    所有召回策略（本地文件、本地数据库、远程等）都应继承此类，
    实现统一的 search 接口。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """召回策略名称，用于日志和调试"""
        ...

    @abstractmethod
    async def search(self, query: str, k: int = 10, **kwargs) -> list[RecallCandidate]:
        """执行召回搜索

        Args:
            query: 搜索查询字符串
            k: 返回的最大结果数
            **kwargs: 额外参数，由具体实现决定

        Returns:
            RecallCandidate 列表
        """
        ...

    def is_available(self) -> bool:
        """检查召回策略是否可用

        Returns:
            True 如果可用，False 如果不可用
        """
        return True

    def get_stats(self) -> dict:
        """获取召回策略统计信息

        Returns:
            包含统计信息的字典
        """
        return {"name": self.name, "available": self.is_available()}
