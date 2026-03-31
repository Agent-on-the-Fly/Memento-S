"""向量存储 - sqlite-vec 数据库管理 embedding

通用组件，可用于 skill embedding 和 conversation embedding 等场景。
通过 table_name / id_column 参数区分不同用途的虚拟表。
"""

from __future__ import annotations

import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any, List, Set, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import sqlite_vec

    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    sqlite_vec = None


def serialize_f32(vec: List[float]) -> bytes:
    """将 float list 序列化为 little-endian float32 bytes（sqlite-vec 格式）"""
    return struct.pack(f"<{len(vec)}f", *vec)


def deserialize_f32(data: bytes) -> List[float]:
    """将 little-endian float32 bytes 反序列化为 float list"""
    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data))


class VectorStorage:
    """向量存储 - sqlite-vec 数据库"""

    def __init__(
        self,
        db_path: Path,
        dimension: int | None = None,
        table_name: str = "skill_embeddings",
        id_column: str = "skill_name",
    ) -> None:
        self._db_path = Path(db_path)
        self._dimension = dimension
        self._table_name = table_name
        self._id_column = id_column
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._initialized = False

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @classmethod
    async def from_config(cls, config: Any) -> "VectorStorage":
        """从配置创建 VectorStorage 实例（向量存储与主数据库使用同一个文件）。"""
        db_path = Path(config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        dimension = getattr(config, "embedding_dimension", 1536)
        storage = cls(db_path=db_path, dimension=dimension)
        await storage.init()
        logger.info(
            "VectorStorage created from config: {} (shared with main DB, dimension={})",
            db_path,
            dimension,
        )
        return storage

    async def init(self) -> None:
        """初始化 sqlite-vec 数据库"""
        if not SQLITE_VEC_AVAILABLE:
            logger.warning("sqlite-vec not installed, vector storage disabled")
            return

        try:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)

            if self._dimension:
                self._create_table(self._dimension)

            self._initialized = True
            logger.info("VectorStorage initialized: {}", self._db_path)

        except Exception as e:
            logger.error("VectorStorage init failed: {}", e)
            raise

    def _create_table(self, dim: int) -> None:
        """创建 vec0 虚拟表"""
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self._table_name} USING vec0(
                {self._id_column} TEXT PRIMARY KEY,
                embedding float[{dim}] distance_metric=cosine
            )
        """)
        self._conn.commit()

    async def close(self) -> None:
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._initialized = False

    async def save(self, name: str, vector: List[float]) -> None:
        """保存向量"""
        if not self.is_ready or not self._conn:
            raise RuntimeError("VectorStorage not initialized")

        if not self._dimension:
            self._dimension = len(vector)
            self._create_table(self._dimension)

        vec_bytes = serialize_f32(vector)

        with self._lock:
            self._conn.execute("BEGIN")
            self._conn.execute(
                f"DELETE FROM {self._table_name} WHERE {self._id_column} = ?", (name,)
            )
            self._conn.execute(
                f"INSERT INTO {self._table_name} ({self._id_column}, embedding) VALUES (?, ?)",
                (name, vec_bytes),
            )
            self._conn.commit()

        logger.debug("Saved vector for '{}' in table '{}'", name, self._table_name)

    async def load(self, name: str) -> List[float] | None:
        """加载向量"""
        if not self.is_ready or not self._conn:
            return None

        if not self._table_exists():
            return None

        row = self._conn.execute(
            f"SELECT embedding FROM {self._table_name} WHERE {self._id_column} = ?",
            (name,),
        ).fetchone()

        if row:
            return deserialize_f32(row[0])
        return None

    def _table_exists(self) -> bool:
        """检查表是否存在"""
        if not self._conn:
            return False
        try:
            self._conn.execute(f"SELECT 1 FROM {self._table_name} LIMIT 1")
            return True
        except sqlite3.OperationalError:
            return False

    async def delete(self, name: str) -> bool:
        """删除向量"""
        if not self.is_ready or not self._conn:
            return False

        if not self._table_exists():
            return False

        with self._lock:
            cursor = self._conn.execute(
                f"DELETE FROM {self._table_name} WHERE {self._id_column} = ?", (name,)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    async def list_names(self) -> Set[str]:
        """返回所有已索引的名称"""
        if not self.is_ready or not self._conn:
            return set()

        if not self._table_exists():
            return set()

        rows = self._conn.execute(
            f"SELECT {self._id_column} FROM {self._table_name}"
        ).fetchall()
        return {row[0] for row in rows}

    async def search(
        self, query_vector: List[float], k: int = 10
    ) -> List[Tuple[str, float]]:
        """向量相似度搜索"""
        if not self.is_ready or not self._conn:
            return []

        vec_bytes = serialize_f32(query_vector)

        rows = self._conn.execute(
            f"""
            SELECT {self._id_column}, distance
            FROM {self._table_name}
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (vec_bytes, k),
        ).fetchall()

        return [(name, 1.0 - distance) for name, distance in rows]
