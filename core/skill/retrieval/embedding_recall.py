"""embedding_recall — sqlite-vec 向量检索

使用 sqlite-vec 扩展进行 skill embedding 的向量检索。
通过 EmbeddingClient 获取 query embedding，在本地 sqlite-vec 中检索 top-k。
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from utils.logger import get_logger
from core.skill.embedding.utils import serialize_f32

logger = get_logger(__name__)


@dataclass
class EmbeddingMatch:
    """Embedding 检索匹配结果"""

    name: str
    score: float  # similarity score (1 - cosine_distance)


class EmbeddingRecall:
    """sqlite-vec 向量检索

    管理独立的 sqlite-vec 数据库，存储 skill embedding 向量。
    通过 EmbeddingClient 生成 embedding，sqlite-vec 执行 cosine 检索。

    Args:
        db_path: sqlite 数据库文件路径
        embedding_client: OpenAI 兼容 embedding API 客户端（可选，无则禁用）
    """

    def __init__(self, db_path: Path, embedding_client=None):
        self._db_path = db_path
        self._embedding_client = embedding_client
        self._dim: int | None = None
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._initialized = False
        self._thread_id: int | None = None

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._conn is not None

    @property
    def dimension(self) -> int | None:
        return self._dim

    def _ensure_connection(self) -> bool:
        """确保 SQLite 连接在当前线程中有效，跨线程时重建连接。"""
        if not self._initialized:
            return False

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
            import sqlite_vec

            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._thread_id = current_thread_id
            logger.debug("Created new SQLite connection for thread {}", current_thread_id)
        except ImportError:
            logger.error("sqlite-vec not installed, cannot create embedding connection")
            return False
        except Exception as e:
            logger.error("Failed to create SQLite connection with sqlite-vec: {}", e)
            return False

        return self._conn is not None

    def init(self, dim: int) -> None:
        """初始化 vec0 虚拟表。"""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._dim = dim

            try:
                import sqlite_vec
            except ImportError:
                logger.warning("sqlite-vec not installed, embedding recall disabled")
                return

            try:
                self._conn = sqlite3.connect(str(self._db_path))
                self._conn.enable_load_extension(True)
                sqlite_vec.load(self._conn)
                self._conn.enable_load_extension(False)

                self._conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS skill_embeddings USING vec0(
                        skill_name TEXT PRIMARY KEY,
                        embedding float[{dim}] distance_metric=cosine
                    )
                """)
                self._conn.commit()

                count = self._conn.execute(
                    "SELECT COUNT(*) FROM skill_embeddings"
                ).fetchone()[0]

                self._initialized = True
                logger.info(
                    "EmbeddingRecall initialized: db={}, dim={}, existing={}",
                    self._db_path, dim, count,
                )
            except Exception as e:
                logger.error("sqlite-vec init failed: {}", e)
                if self._conn:
                    self._conn.close()
                    self._conn = None

    async def ensure_ready_async(self) -> bool:
        """确保 embedding recall 可用，自动检测维度并初始化。"""
        if self._initialized:
            return True
        if not self._embedding_client:
            return False

        try:
            dim = self._embedding_client.dimension
            if dim is None:
                await self.embed_texts(["dimension probe"])
                dim = self._embedding_client.dimension

            if dim:
                self.init(dim)
                return True
        except Exception as e:
            logger.warning("EmbeddingRecall auto-init failed: {}", e)

        return False

    async def search(
        self,
        query: str,
        k: int = 10,
        min_score: float = 0.0,
    ) -> list[EmbeddingMatch]:
        """语义检索 top-k skills，按相似度降序返回。"""
        if not self.is_ready or not self._embedding_client:
            return []

        try:
            query_vec = await self.embed_query(query)
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

            return [
                EmbeddingMatch(name=name, score=max(0.0, 1.0 - distance))
                for name, distance in rows
                if max(0.0, 1.0 - distance) >= min_score
            ]

        except Exception as e:
            logger.warning("Embedding search failed: {}", e)
            return []

    async def upsert_async(self, skill_name: str, text: str) -> bool:
        """为 skill 生成 embedding 并 upsert 到 vec store。"""
        if not self.is_ready or not self._embedding_client:
            return False

        try:
            vecs = await self.embed_texts([text])
            if not vecs:
                return False

            vec_bytes = serialize_f32(vecs[0])

            if not self._ensure_connection():
                logger.error("Cannot upsert '{}': failed to get database connection", skill_name)
                return False

            with self._lock:
                try:
                    self._conn.execute("BEGIN")
                    self._conn.execute(
                        "DELETE FROM skill_embeddings WHERE skill_name = ?", (skill_name,)
                    )
                    self._conn.execute(
                        "INSERT INTO skill_embeddings (skill_name, embedding) VALUES (?, ?)",
                        (skill_name, vec_bytes),
                    )
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    raise

            return True

        except Exception as e:
            logger.warning("Embedding upsert failed for '{}': {}", skill_name, e)
            return False

    async def upsert_batch_async(self, items: list[tuple[str, str]]) -> int:
        """批量 upsert embedding，返回成功 upsert 的数量。"""
        if not self.is_ready or not self._embedding_client or not items:
            return 0

        names = [name for name, _ in items]
        texts = [text for _, text in items]

        try:
            vecs = await self.embed_texts(texts)
            if not vecs or len(vecs) != len(names):
                return 0

            if not self._ensure_connection():
                logger.error("Cannot batch upsert: failed to get database connection")
                return 0

            with self._lock:
                try:
                    self._conn.execute("BEGIN")
                    for name, vec in zip(names, vecs):
                        vec_bytes = serialize_f32(vec)
                        self._conn.execute(
                            "DELETE FROM skill_embeddings WHERE skill_name = ?", (name,)
                        )
                        self._conn.execute(
                            "INSERT INTO skill_embeddings (skill_name, embedding) VALUES (?, ?)",
                            (name, vec_bytes),
                        )
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    raise

            return len(names)

        except Exception as e:
            logger.warning("Batch embedding upsert failed: {}", e)
            return 0

    def delete(self, skill_names: list[str]) -> None:
        """删除指定 skill 的 embedding。"""
        if not self.is_ready or not skill_names:
            return

        if not self._ensure_connection():
            logger.error("Cannot delete: failed to get database connection")
            return

        with self._lock:
            placeholders = ",".join("?" * len(skill_names))
            self._conn.execute(
                f"DELETE FROM skill_embeddings WHERE skill_name IN ({placeholders})",
                skill_names,
            )
            self._conn.commit()

    def get_indexed_names(self) -> set[str]:
        """返回所有已索引的 skill name。"""
        if not self.is_ready:
            return set()

        if not self._ensure_connection():
            logger.error("Cannot get indexed names: failed to get database connection")
            return set()

        rows = self._conn.execute("SELECT skill_name FROM skill_embeddings").fetchall()
        return {row[0] for row in rows}

    def cleanup_orphans(self, valid_names: set[str]) -> int:
        """删除不在 valid_names 中的孤儿记录。"""
        if not self.is_ready:
            return 0

        orphans = self.get_indexed_names() - valid_names
        if orphans:
            self.delete(list(orphans))
            logger.info("Removed {} orphan(s) from embedding index", len(orphans))
        return len(orphans)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self._embedding_client:
            return []
        return await self._embedding_client.embed(texts)

    async def embed_query(self, query: str) -> list[float]:
        if not self._embedding_client:
            return []
        return await self._embedding_client.embed_query(query)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            self._initialized = False
