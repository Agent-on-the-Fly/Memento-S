"""local_db_recall — 本地向量检索

使用 sqlite-vec 扩展进行 skill embedding 的本地向量检索。
通过 EmbeddingClient 获取 query embedding，在本地 sqlite-vec 中检索 top-k。

注意：系统初始化由 bootstrap 保证，此类假设数据库已初始化。
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import sqlite_vec

from middleware.llm.embedding_client import EmbeddingClient
from middleware.storage.utils import serialize_f32
from utils.logger import get_logger
from .base import BaseRecall
from .schema import RecallCandidate

logger = get_logger(__name__)


class LocalDbRecall(BaseRecall):
    """本地向量检索

    管理独立的 sqlite-vec 数据库，执行本地向量检索（cosine similarity）。
    假设数据库已由系统启动流程（bootstrap）初始化完成。

    Args:
        db_path: sqlite 数据库文件路径
        embedding_client: OpenAI 兼容 embedding API 客户端（可选，无则禁用）
    """

    def __init__(self, db_path: Path, embedding_client=None):
        self._db_path = db_path
        self._embedding_client = embedding_client
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._thread_id: int | None = None

    @classmethod
    def from_config(cls, config: "SkillConfig") -> "LocalDbRecall":
        """从配置创建 LocalDbRecall 实例

        Args:
            config: SkillConfig 配置

        Returns:
            LocalDbRecall 实例（如果 embedding 配置可用）
        """
        embedding_client = EmbeddingClient.from_config()
        return cls(config.db_path, embedding_client)

    @property
    def name(self) -> str:
        """召回策略名称"""
        return "local_db"

    def is_available(self) -> bool:
        """检查数据库连接是否可用"""
        try:
            return self._ensure_connection()
        except Exception:
            return False

    def _ensure_connection(self) -> bool:
        """确保 SQLite 连接在当前线程中有效，跨线程时重建连接。"""
        current_thread_id = threading.current_thread().ident
        if self._conn is not None and self._thread_id == current_thread_id:
            return True

        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        try:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._thread_id = current_thread_id
            logger.debug(
                "Created new SQLite connection for thread {}", current_thread_id
            )
        except ImportError:
            logger.error("sqlite-vec not installed, cannot create embedding connection")
            return False
        except Exception as e:
            logger.error("Failed to create SQLite connection with sqlite-vec: {}", e)
            return False

        return self._conn is not None

    async def search(
        self,
        query: str,
        k: int = 10,
        min_score: float = 0.0,
        **kwargs,
    ) -> list["RecallCandidate"]:
        """语义检索 top-k skills，返回 RecallCandidate 列表。"""
        if not self._embedding_client:
            return []

        try:
            query_vec = await self._embed_query(query)
            vec_bytes = serialize_f32(query_vec)

            if not self._ensure_connection():
                logger.error("Cannot search: failed to get database connection")
                return []

            rows = self._conn.execute(
                """
                SELECT skill_name, distance
                FROM skill_embeddings
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
                """,
                (vec_bytes, k),
            ).fetchall()

            candidates: list[RecallCandidate] = []
            for name, distance in rows:
                score = max(0.0, 1.0 - distance)
                if score >= min_score:
                    candidates.append(
                        RecallCandidate(
                            name=name,
                            source="local",
                            score=score,
                            match_type="embedding",
                        )
                    )
            return candidates

        except Exception as e:
            logger.warning("Local search failed: {}", e)
            return []

    def get_indexed_names(self) -> set[str]:
        """返回所有已索引的 skill name。"""
        if not self._ensure_connection():
            logger.error("Cannot get indexed names: failed to get database connection")
            return set()

        rows = self._conn.execute("SELECT skill_name FROM skill_embeddings").fetchall()
        return {row[0] for row in rows}

    async def _embed_query(self, query: str) -> list[float]:
        if not self._embedding_client:
            return []
        return await self._embedding_client.embed_query(query)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = super().get_stats()
        stats.update(
            {
                "db_path": str(self._db_path),
                "embedding_ready": self._embedding_client is not None,
                "indexed_count": len(self.get_indexed_names())
                if self.is_available()
                else 0,
            }
        )
        return stats
