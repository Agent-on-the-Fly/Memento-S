"""Skill VectorStorage - 同时继承 middleware VectorStorage 和 SkillStorage"""

from __future__ import annotations

from typing import Any, List, Set

from core.skill.store.base import SkillStorage
from middleware.storage.vector_storage import (
    SQLITE_VEC_AVAILABLE,
    VectorStorage as MiddlewareVectorStorage,
    deserialize_f32,
    serialize_f32,
)

__all__ = [
    "SQLITE_VEC_AVAILABLE",
    "VectorStorage",
    "deserialize_f32",
    "serialize_f32",
]


class VectorStorage(MiddlewareVectorStorage, SkillStorage):
    """Skill 向量存储 - 同时满足 middleware 和 core.skill 接口"""

    def __init__(
        self,
        db_path,
        dimension: int | None = None,
        table_name: str = "skill_embeddings",
        id_column: str = "skill_name",
    ) -> None:
        # 初始化 middleware VectorStorage
        MiddlewareVectorStorage.__init__(
            self,
            db_path=db_path,
            dimension=dimension,
            table_name=table_name,
            id_column=id_column,
        )
        # 初始化 SkillStorage
        SkillStorage.__init__(self)

    async def save(self, name: str, data: Any) -> None:
        """保存向量数据（实现 SkillStorage 接口）

        Args:
            name: skill 名称
            data: 向量数据 (List[float])
        """
        if not isinstance(data, list):
            raise TypeError(f"VectorStorage.save expects list, got {type(data)}")
        await MiddlewareVectorStorage.save(self, name, data)

    async def load(self, name: str) -> Any | None:
        """加载向量数据（实现 SkillStorage 接口）

        Args:
            name: skill 名称

        Returns:
            向量数据 (List[float]) 或 None
        """
        return await MiddlewareVectorStorage.load(self, name)

    async def delete(self, name: str) -> bool:
        """删除向量数据（实现 SkillStorage 接口）

        Args:
            name: skill 名称

        Returns:
            是否删除成功
        """
        return await MiddlewareVectorStorage.delete(self, name)

    async def list_names(self) -> Set[str]:
        """返回所有已索引的 skill 名称（实现 SkillStorage 接口）

        Returns:
            skill 名称集合
        """
        return await MiddlewareVectorStorage.list_names(self)

    async def init(self) -> None:
        """初始化存储（实现 SkillStorage 接口）"""
        await MiddlewareVectorStorage.init(self)
        # 同步初始化状态
        self._initialized = self.is_ready

    async def close(self) -> None:
        """关闭存储（实现 SkillStorage 接口）"""
        await MiddlewareVectorStorage.close(self)

    async def search(self, query_vector: List[float], k: int = 10):
        """向量相似度搜索"""
        return await MiddlewareVectorStorage.search(self, query_vector, k)
