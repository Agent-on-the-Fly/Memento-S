
from __future__ import annotations

import json
from pathlib import Path

import chromadb

from core.config import g_settings
from core.config.logging import get_logger
from ...schema import Skill
from .persistence import get_skill_mtime, to_kebab_case

logger = get_logger(__name__)


class EmbeddingStore:

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self.ef = None
        self.client = None
        self.collection = None
        self._initialized = False

        self.mtime_cache_file = workspace / "data" / ".mtime_cache.json"
        self.mtime_cache: dict = {}
        if self.mtime_cache_file.exists():
            try:
                self.mtime_cache = json.loads(self.mtime_cache_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Could not load mtime cache: %s", e)

    def ensure(
        self,
        local_cache: dict[str, Skill],
        skills_directory: Path,
    ) -> None:
        if self._initialized:
            return

        chosen_model = self._select_embedding_model()

        if chosen_model == "none":
            self._initialized = True
            self.ef = None
            self.client = None
            self.collection = None
            logger.info("ChromaDB skipped (embedding disabled)")
            return

        chroma_directory = g_settings.chroma_directory
        model_lock_file = chroma_directory / ".embedding_model"

        self.ef = self._load_embedding_function(chosen_model)
        self.client = chromadb.PersistentClient(path=str(chroma_directory))

        self._init_or_rebuild_collection(chosen_model, model_lock_file)

        try:
            chroma_directory.mkdir(parents=True, exist_ok=True)
            model_lock_file.write_text(chosen_model, encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write embedding model lock file: %s", e)

        self._initialized = True

        if self.collection.count() == 0 and local_cache:
            logger.info(
                "ChromaDB collection is empty, forcing full resync of %d skills",
                len(local_cache),
            )
            self.mtime_cache = {}

        self._cleanup_orphans(local_cache)
        self._sync_updated(local_cache, skills_directory)

    def preflight_check(
        self,
        local_cache: dict[str, Skill],
        skills_directory: Path,
    ) -> None:
        model = g_settings.embedding_model.strip().lower()
        if model in ("none", "disabled", "off", "false"):
            return

        chroma_directory = g_settings.chroma_directory
        model_lock_file = chroma_directory / ".embedding_model"

        need_eager_init = False

        try:
            if model_lock_file.exists():
                previous_model = model_lock_file.read_text(encoding="utf-8").strip()
                current_model = g_settings.embedding_model
                if current_model == "auto":
                    current_model = "BAAI/bge-m3"
                if previous_model and previous_model != current_model:
                    logger.warning(
                        "Preflight: embedding model changed '%s' → '%s', triggering eager ChromaDB init",
                        previous_model, current_model,
                    )
                    need_eager_init = True
        except (OSError, ValueError) as e:
            logger.debug("Preflight: failed to read model lock file: %s", e)

        if not need_eager_init:
            cached_names = set(self.mtime_cache.keys())
            loaded_names = set(local_cache.keys())
            new_skills = loaded_names - cached_names
            deleted_skills = cached_names - loaded_names
            if new_skills or deleted_skills:
                logger.info(
                    "Preflight: skill set changed (new=%d, removed=%d), triggering eager ChromaDB init",
                    len(new_skills), len(deleted_skills),
                )
                need_eager_init = True

        if need_eager_init:
            self.ensure(local_cache, skills_directory)

    def destroy(self) -> None:
        if self._initialized and self.client:
            try:
                self.client.reset()
            except Exception as e:
                logger.debug("ChromaDB reset failed: %s", e)

        self._initialized = False
        self.client = None
        self.collection = None

    def upsert(self, skill: Skill) -> None:
        if self.collection is None:
            return
        chroma_metadata = skill.model_dump(exclude={"code", "files"})
        for key in ("parameters", "dependencies", "tags"):
            if key in chroma_metadata and isinstance(chroma_metadata[key], (dict, list)):
                chroma_metadata[key] = json.dumps(chroma_metadata[key])

        self.collection.upsert(
            documents=[skill.to_embedding_text()],
            metadatas=[chroma_metadata],
            ids=[skill.name],
        )

    def delete(self, ids: list[str]) -> None:
        if self.collection is None:
            return
        self.collection.delete(ids=ids)

    def query(
        self,
        query: str,
        k: int,
        skill_cache: dict[str, Skill],
    ) -> list:
        from ...retrieval import ScoredSkill

        try:
            results = self.collection.query(query_texts=[query], n_results=k)
            if results["ids"] and results["ids"][0]:
                names = results["ids"][0]
                distances = results.get("distances", [[]])[0]
                scored: list[ScoredSkill] = []
                for i, name in enumerate(names):
                    if name not in skill_cache:
                        continue
                    dist = distances[i] if i < len(distances) else 0.0
                    sim = max(0.0, 1.0 - dist)
                    scored.append(ScoredSkill(
                        skill=skill_cache[name],
                        score=sim,
                        embedding_score=sim,
                    ))
                return scored
        except Exception as e:
            logger.warning("ChromaDB search failed: %s", e)
        return []

    def _cleanup_orphans(self, local_cache: dict[str, Skill]) -> None:
        try:
            existing_ids = (
                set(self.collection.get()["ids"])
                if self.collection.count() > 0
                else set()
            )
            orphan_ids = existing_ids - set(local_cache.keys())
            if orphan_ids:
                self.collection.delete(ids=list(orphan_ids))
                for oid in orphan_ids:
                    self.mtime_cache.pop(oid, None)
                logger.info(
                    "Removed %d orphan(s) from ChromaDB: %s",
                    len(orphan_ids), list(orphan_ids),
                )
        except Exception as e:
            logger.debug("Orphan cleanup skipped: %s", e)

    def _sync_updated(
        self,
        local_cache: dict[str, Skill],
        skills_directory: Path,
    ) -> None:
        updated_skills = []
        for name, skill in local_cache.items():
            skill_dir = skills_directory / to_kebab_case(name)
            current_mtime = get_skill_mtime(skill_dir)
            if current_mtime > self.mtime_cache.get(name, 0):
                updated_skills.append(skill)
                self.mtime_cache[name] = current_mtime

        if not updated_skills:
            return

        logger.info("Syncing %d updated skill(s) to ChromaDB...", len(updated_skills))
        for skill in updated_skills:
            self.upsert(skill)

        self.save_mtime_cache()

    def save_mtime_cache(self) -> None:
        try:
            self.mtime_cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.mtime_cache_file.write_text(json.dumps(self.mtime_cache))
        except Exception as e:
            logger.warning("Failed to save mtime cache: %s", e)

    @staticmethod
    def _select_embedding_model() -> str:
        model = g_settings.embedding_model.strip().lower()
        if model in ("none", "disabled", "off", "false"):
            logger.info("Embedding disabled (EMBEDDING_MODEL=%s), using BM25 only", g_settings.embedding_model)
            return "none"
        if model != "auto":
            logger.info("Using manually configured embedding model: %s", g_settings.embedding_model)
            return g_settings.embedding_model
        chosen = "BAAI/bge-m3"
        logger.info("Using unified embedding model: %s", chosen)
        return chosen

    @staticmethod
    def _load_embedding_function(model_name: str):

        base_url = (g_settings.embedding_base_url or "").strip()
        if base_url:
            return EmbeddingStore._load_api_embedding_function(model_name, base_url)

        if model_name == "qwen3":
            try:
                from ...retrieval.qwen3_embedding import Qwen3EmbeddingFunction

                tokenizer_path = g_settings.qwen3_tokenizer_path
                model_path = g_settings.qwen3_model_path

                if not tokenizer_path or not model_path:
                    raise ValueError(
                        "QWEN3_TOKENIZER_PATH and QWEN3_MODEL_PATH must be set "
                        "when EMBEDDING_MODEL=qwen3"
                    )

                ef = Qwen3EmbeddingFunction(
                    tokenizer_path=tokenizer_path,
                    model_path=model_path,
                )
                logger.info(
                    "Qwen3 embedding loaded: tokenizer=%s, model=%s",
                    tokenizer_path, model_path,
                )
                return ef
            except Exception as e:
                logger.error("Failed to load Qwen3 embedding: %s", e)
                logger.warning("Falling back to BAAI/bge-m3")
                model_name = "BAAI/bge-m3"

        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            ef = SentenceTransformerEmbeddingFunction(model_name=model_name)
            logger.info("Embedding model loaded: %s", model_name)
            return ef
        except Exception as e:
            logger.warning(
                "Failed to load '%s', falling back to ChromaDB default: %s",
                model_name, e,
            )
            from chromadb.utils import embedding_functions
            return embedding_functions.DefaultEmbeddingFunction()

    @staticmethod
    def _load_api_embedding_function(model_name: str, base_url: str):
        if base_url.rstrip("/").endswith("/embeddings"):
            base_url = base_url.rstrip("/").rsplit("/embeddings", 1)[0]
        base_url = base_url.rstrip("/")

        api_key = g_settings.embedding_api_key or "no-key-required"

        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

            ef = OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=base_url,
                model_name=model_name,
            )
            logger.info(
                "API embedding loaded: model=%s, base_url=%s",
                model_name, base_url,
            )
            return ef
        except Exception as e:
            logger.error(
                "Failed to load API embedding (model=%s, base_url=%s): %s",
                model_name, base_url, e,
            )
            logger.warning("Falling back to local SentenceTransformer")
            try:
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                fallback_model = "BAAI/bge-m3"
                ef = SentenceTransformerEmbeddingFunction(model_name=fallback_model)
                logger.info("Fallback embedding model loaded: %s", fallback_model)
                return ef
            except Exception as e2:
                logger.warning("Fallback also failed: %s", e2)
                from chromadb.utils import embedding_functions
                return embedding_functions.DefaultEmbeddingFunction()

    def _init_or_rebuild_collection(self, chosen_model: str, model_lock_file: Path):
        previous_model = None
        need_rebuild = False
        try:
            if model_lock_file.exists():
                previous_model = model_lock_file.read_text(encoding="utf-8").strip()
                if previous_model and previous_model != chosen_model:
                    need_rebuild = True
                    logger.warning(
                        "Embedding model changed: '%s' → '%s', will rebuild ChromaDB",
                        previous_model, chosen_model,
                    )
        except (OSError, ValueError) as e:
            logger.debug("Failed to read embedding model lock file: %s", e)

        try:
            self.collection = self.client.get_or_create_collection(
                name="agent_skills",
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"},
            )
            if not need_rebuild and self.collection.count() > 0:
                try:
                    self.collection.query(query_texts=["test"], n_results=1)
                except Exception as e:
                    if "dimension" in str(e).lower():
                        need_rebuild = True
                        logger.warning("Embedding dimension mismatch detected: %s", e)
                    else:
                        raise

            if not need_rebuild:
                actual_space = (self.collection.metadata or {}).get("hnsw:space", "l2")
                if actual_space != "cosine":
                    logger.warning(
                        "ChromaDB distance metric mismatch: expected 'cosine', got '%s' → rebuilding",
                        actual_space,
                    )
                    need_rebuild = True
        except ValueError:
            need_rebuild = True

        if need_rebuild:
            logger.warning("Recreating ChromaDB collection with model: %s", chosen_model)
            try:
                self.client.delete_collection("agent_skills")
            except Exception as e:
                logger.debug("delete_collection failed (may not exist): %s", e)
            self.collection = self.client.get_or_create_collection(
                name="agent_skills",
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"},
            )
            self.mtime_cache = {}
