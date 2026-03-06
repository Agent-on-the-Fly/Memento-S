
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from rank_bm25 import BM25Okapi

from core.config.logging import get_logger
from .bm25 import BM25Index

logger = get_logger(__name__)


@dataclass
class CloudSkillEntry:

    id: str
    name: str
    author: str
    description: str
    github_url: str
    stars: int
    updated_at: int = 0


class CloudCatalog:

    _IDF_FLOOR = 0.1
    _EMBED_BATCH_SIZE = 200

    def __init__(self, jsonl_path: Path):
        self._jsonl_path = jsonl_path
        self._entries: dict[str, CloudSkillEntry] = {}      # id -> entry
        self._name_index: dict[str, CloudSkillEntry] = {}    # normalized_name -> entry
        self._entry_ids: list[str] = []                      #  BM25 corpus 
        self._bm25: Optional[BM25Okapi] = None

        self._embedding_fn = None
        self._collection: chromadb.Collection | None = None
        self._embedding_ready = False
        self._embedding_lock = threading.Lock()

        self._load_and_index(jsonl_path)

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def embedding_ready(self) -> bool:
        return self._embedding_ready


    def _load_and_index(self, jsonl_path: Path) -> None:
        if not jsonl_path.exists():
            logger.warning("Cloud catalog not found: %s", jsonl_path)
            return

        entries: list[CloudSkillEntry] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = CloudSkillEntry(
                        id=data.get("id", ""),
                        name=data.get("name", ""),
                        author=data.get("author", ""),
                        description=data.get("description", ""),
                        github_url=data.get("githubUrl", ""),
                        stars=data.get("stars", 0),
                        updated_at=data.get("updatedAt", 0),
                    )
                    if entry.id and entry.name:
                        entries.append(entry)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug("Skipped malformed line %d: %s", line_no, e)

        if not entries:
            logger.warning("Cloud catalog is empty: %s", jsonl_path)
            return

        corpus_tokens: list[list[str]] = []
        entry_ids: list[str] = []

        for entry in entries:
            self._entries[entry.id] = entry
            norm_name = entry.name.lower().replace("-", "_").replace(" ", "_")
            self._name_index[norm_name] = entry
            self._name_index[entry.name.lower()] = entry

            text = f"{entry.name} {entry.description}"
            tokens = BM25Index._tokenize(text)
            corpus_tokens.append(tokens)
            entry_ids.append(entry.id)

        self._entry_ids = entry_ids
        self._bm25 = BM25Okapi(corpus_tokens, epsilon=0)

        patched = 0
        for word, idf_val in self._bm25.idf.items():
            if idf_val < self._IDF_FLOOR:
                self._bm25.idf[word] = self._IDF_FLOOR
                patched += 1
        if patched > 0:
            logger.debug(
                "CloudCatalog BM25: clamped %d negative/zero IDF values (corpus=%d)",
                patched, len(corpus_tokens),
            )

        logger.info("Cloud catalog loaded: %d skills indexed", len(entries))


    def search(self, query: str, k: int = 10) -> List[Tuple[CloudSkillEntry, float]]:
        if self._bm25 is None or not self._entry_ids:
            return []

        tokens = BM25Index._tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        scored_pairs = sorted(
            zip(self._entry_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results: List[Tuple[CloudSkillEntry, float]] = []
        for entry_id, score in scored_pairs[:k]:
            if score <= 0:
                break
            entry = self._entries.get(entry_id)
            if entry:
                results.append((entry, score))

        return results

    def get_by_name(self, name: str) -> Optional[CloudSkillEntry]:
        norm = name.lower().replace("-", "_").replace(" ", "_")
        return self._name_index.get(norm)


    def init_embedding_async(self, embedding_fn) -> None:
        self._embedding_fn = embedding_fn
        t = threading.Thread(target=self._build_embedding_index, daemon=True)
        t.start()

    def _build_embedding_index(self) -> None:
        if not self._entries or not self._embedding_fn:
            return

        try:
            chroma_dir = self._jsonl_path.parent / ".cloud_chroma"
            chroma_dir.mkdir(parents=True, exist_ok=True)
            mtime_file = chroma_dir / ".catalog_mtime"

            current_mtime = ""
            if self._jsonl_path.exists():
                current_mtime = str(self._jsonl_path.stat().st_mtime)

            client = chromadb.PersistentClient(path=str(chroma_dir))
            self._collection = client.get_or_create_collection(
                name="cloud_skills",
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

            stored_mtime = ""
            if mtime_file.exists():
                stored_mtime = mtime_file.read_text(encoding="utf-8").strip()

            need_rebuild = (
                self._collection.count() != len(self._entries)
                or stored_mtime != current_mtime
            )

            if need_rebuild:
                logger.info(
                    "Cloud embedding index: rebuilding (%d entries, count=%d, mtime_changed=%s)",
                    len(self._entries),
                    self._collection.count(),
                    stored_mtime != current_mtime,
                )
                self._rebuild_collection()
                mtime_file.write_text(current_mtime, encoding="utf-8")
            else:
                logger.info(
                    "Cloud embedding index: loaded from cache (%d entries)",
                    self._collection.count(),
                )

            self._embedding_ready = True

        except Exception as e:
            logger.warning("Cloud embedding index build failed: %s", e)

    def _rebuild_collection(self) -> None:
        if not self._collection:
            return

        existing = self._collection.count()
        if existing > 0:
            all_ids = self._collection.get()["ids"]
            if all_ids:
                for i in range(0, len(all_ids), self._EMBED_BATCH_SIZE):
                    self._collection.delete(ids=all_ids[i:i + self._EMBED_BATCH_SIZE])

        all_entries = list(self._entries.values())
        total = len(all_entries)

        for batch_start in range(0, total, self._EMBED_BATCH_SIZE):
            batch = all_entries[batch_start:batch_start + self._EMBED_BATCH_SIZE]
            ids = [e.id for e in batch]
            documents = [f"{e.name} | {e.description}" for e in batch]
            metadatas = [{"name": e.name, "idx": str(i)} for i, e in enumerate(batch, start=batch_start)]

            self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

            if (batch_start + self._EMBED_BATCH_SIZE) % 1000 < self._EMBED_BATCH_SIZE:
                logger.info(
                    "Cloud embedding: %d/%d entries indexed",
                    min(batch_start + self._EMBED_BATCH_SIZE, total), total,
                )

        logger.info("Cloud embedding index: built %d entries", total)

    def search_embedding(self, query: str, k: int = 3) -> List[Tuple[CloudSkillEntry, float]]:
        if not self._embedding_ready or not self._collection:
            return self.search(query, k)

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, self._collection.count()),
            )

            if not results or not results["ids"] or not results["ids"][0]:
                return []

            output: List[Tuple[CloudSkillEntry, float]] = []
            ids = results["ids"][0]
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

            for entry_id, distance in zip(ids, distances):
                entry = self._entries.get(entry_id)
                if entry:
                    score = max(0.0, 1.0 - distance)
                    output.append((entry, score))

            return output

        except Exception as e:
            logger.warning("Cloud embedding search failed, falling back to BM25: %s", e)
            return self.search(query, k)
