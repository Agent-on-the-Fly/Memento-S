# SPDX-License-Identifier: Apache-2.0
"""Behaviour-aligned Qwen embedding recall for local skills.

The router follows the Memento-S reference implementation: skill documents are
embedded without an instruction while the routing goal receives the Qwen3
retrieval instruction, embeddings are last-token pooled and L2 normalised, and
cosine similarity supplies the one-step soft-Q score.

Two backends are supported:

* ``local``: a Qwen3-Embedding base model or Memento-Qwen checkpoint loaded by
  ``transformers``;
* ``api``: an OpenAI-compatible embedding endpoint, useful for packaged builds
  where shipping torch is undesirable.

Heavy dependencies and model weights are loaded lazily.  If configuration is
missing, ``is_available`` is false and MultiRecall keeps the keyword route.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import httpx

from shared.schema import SkillConfig, SkillSearchResult
from utils.logger import get_logger

from .base import BaseRecall
from .local_recall import LocalRecall, _CacheEntry

logger = get_logger(__name__)

_PLACEHOLDER_API_KEYS = {"sk-xxxxxx", "your-api-key", "replace-me"}


class QwenRecall(BaseRecall):
    """Dense local-skill recall backed by Qwen3 embeddings."""

    def __init__(
        self,
        skills_dir: Path | str,
        *,
        tokenizer_path: str = "Qwen/Qwen3-Embedding-0.6B",
        model_path: str = "",
        base_url: str = "",
        api_key: str = "",
        api_model: str = "Qwen3-Embedding-0.6B",
        device: str = "auto",
        max_length: int = 8192,
        batch_size: int = 32,
        timeout_sec: float = 30.0,
        query_instruction: str = (
            "Instruct: Given a user query, retrieve relevant skill descriptions "
            "that match the query\nQuery:"
        ),
    ) -> None:
        self._skills_dir = Path(skills_dir)
        self._tokenizer_path = tokenizer_path.strip()
        self._model_path = model_path.strip()
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._api_model = api_model.strip() or "Qwen3-Embedding-0.6B"
        self._device_setting = device
        self._max_length = max(1, int(max_length))
        self._batch_size = max(1, int(batch_size))
        self._timeout_sec = max(1.0, float(timeout_sec))
        self._query_instruction = query_instruction

        self._catalog = LocalRecall(self._skills_dir)
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"
        self._doc_signature = ""
        self._doc_names: list[str] = []
        self._doc_embeddings: Any = None
        self._load_lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: SkillConfig) -> QwenRecall:
        return cls(
            config.skills_dir,
            tokenizer_path=config.qwen_tokenizer_path,
            model_path=config.qwen_model_path,
            base_url=config.retrieval_embedding_base_url or "",
            api_key=config.retrieval_embedding_api_key or "",
            api_model=config.retrieval_embedding_model or "Qwen3-Embedding-0.6B",
            device=config.qwen_device,
            max_length=config.qwen_max_length,
            batch_size=config.qwen_batch_size,
            timeout_sec=config.qwen_timeout_sec,
            query_instruction=config.qwen_query_instruction,
        )

    @property
    def name(self) -> str:
        return "qwen"

    @property
    def backend(self) -> str:
        return "local" if self._model_path else "api"

    def is_available(self) -> bool:
        api_key = self._api_key.strip().lower()
        api_configured = bool(self._base_url and api_key not in _PLACEHOLDER_API_KEYS)
        configured = bool(self._model_path or api_configured)
        return configured and self._skills_dir.is_dir()

    def _entries(self) -> list[_CacheEntry]:
        if self._catalog._has_changes():
            self._catalog._refresh_cache()
        return [
            self._catalog._state.entries[k]
            for k in sorted(self._catalog._state.entries)
        ]

    @staticmethod
    def _document(entry: _CacheEntry) -> str:
        # Match the reference catalog_embedding.py representation.  The
        # behaviour-tuned checkpoint was trained against name + description;
        # unrelated frontmatter tags can move documents off that distribution.
        return (f"Skill: {entry.skill_name}\nDescription: {entry.description}").strip()

    def _signature(self, entries: list[_CacheEntry]) -> str:
        payload = {
            "backend": self.backend,
            "model": self._model_path or self._api_model,
            "base_url": self._base_url,
            "tokenizer": self._tokenizer_path,
            "max_length": self._max_length,
            "entries": [(e.skill_name, e.mtime, e.size) for e in entries],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def _cache_file(self) -> Path:
        model_id = "|".join(
            [
                self.backend,
                self._model_path or self._api_model,
                self._base_url,
                self._tokenizer_path,
                str(self._max_length),
            ]
        )
        model_hash = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:12]
        return self._skills_dir / ".router-cache" / f"qwen.{model_hash}.npz"

    def _load_disk_cache(
        self,
        *,
        signature: str,
        names: list[str],
    ) -> Any | None:
        try:
            import numpy as np
        except ImportError:
            return None
        cache_file = self._cache_file()
        if not cache_file.is_file():
            return None
        try:
            with np.load(cache_file, allow_pickle=False) as data:
                if str(data["signature"].item()) != signature:
                    return None
                if [str(item) for item in data["names"].tolist()] != names:
                    return None
                embeddings = data["embeddings"].astype("float32")
                if embeddings.shape[0] != len(names):
                    return None
                return embeddings
        except (OSError, KeyError, ValueError):
            return None

    def _save_disk_cache(
        self,
        *,
        signature: str,
        names: list[str],
        embeddings: Any,
    ) -> None:
        try:
            import numpy as np
        except ImportError:
            return
        cache_file = self._cache_file()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temp = cache_file.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            array = (
                embeddings.detach().cpu().numpy()
                if hasattr(embeddings, "detach")
                else np.asarray(embeddings)
            )
            with temp.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    signature=np.asarray(signature),
                    names=np.asarray(names, dtype=str),
                    embeddings=np.asarray(array, dtype="float32"),
                )
            os.replace(temp, cache_file)
        except (OSError, TypeError, ValueError):
            temp.unlink(missing_ok=True)

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(float(v) * float(v) for v in vector))
        if norm <= 0:
            return [0.0 for _ in vector]
        return [float(v) / norm for v in vector]

    async def _embed_api(self, texts: list[str]) -> list[list[float]]:
        output: list[list[float]] = []
        async with httpx.AsyncClient(
            timeout=self._timeout_sec,
            trust_env=False,
        ) as client:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key or 'no-key-required'}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self._api_model, "input": batch},
                )
                response.raise_for_status()
                data = response.json().get("data") or []
                data.sort(key=lambda item: item.get("index", 0))
                output.extend(self._normalize(item["embedding"]) for item in data)
        if len(output) != len(texts):
            raise RuntimeError(
                "Qwen embedding API returned "
                f"{len(output)} vectors for {len(texts)} texts"
            )
        return output

    def _ensure_local_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local Qwen router requires `pip install -e '.[qwen-router]'`"
            ) from exc

        requested = self._device_setting
        if requested == "auto":
            if torch.cuda.is_available():
                requested = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                requested = "mps"
            else:
                requested = "cpu"
        dtype = (
            torch.bfloat16
            if requested.startswith("cuda")
            else torch.float16
            if requested == "mps"
            else torch.float32
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_path or self._model_path,
            padding_side="left",
        )
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if requested.startswith("cuda"):
            kwargs["device_map"] = "auto"
            errors: list[str] = []
            for attention in ("flash_attention_2", "sdpa", None):
                try:
                    attempt = dict(kwargs)
                    if attention:
                        attempt["attn_implementation"] = attention
                    self._model = AutoModel.from_pretrained(self._model_path, **attempt)
                    break
                except Exception as exc:  # noqa: BLE001 - retry alternate attention backends
                    errors.append(f"{attention or 'default'}: {exc}")
            if self._model is None:
                raise RuntimeError(
                    "Failed to load Qwen embedding model: " + " | ".join(errors)
                )
        else:
            self._model = AutoModel.from_pretrained(self._model_path, **kwargs).to(
                requested
            )
        self._model.eval()
        self._torch = torch
        self._device = str(next(self._model.parameters()).device)

    @staticmethod
    def _last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
        import torch

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _embed_local_sync(self, texts: list[str]) -> Any:
        self._ensure_local_model()
        torch = self._torch
        chunks: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(self._device)
                result = self._model(**inputs)
                pooled = self._last_token_pool(
                    result.last_hidden_state, inputs["attention_mask"]
                )
                chunks.append(
                    torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().float()
                )
        return torch.cat(chunks, dim=0)

    async def _embed(self, texts: list[str]) -> Any:
        if self.backend == "api":
            return await self._embed_api(texts)
        return await asyncio.to_thread(self._embed_local_sync, texts)

    async def _ensure_documents(self, entries: list[_CacheEntry]) -> None:
        signature = self._signature(entries)
        if signature == self._doc_signature:
            return
        async with self._load_lock:
            if signature == self._doc_signature:
                return
            self._doc_names = [entry.skill_name for entry in entries]
            cached = await asyncio.to_thread(
                self._load_disk_cache,
                signature=signature,
                names=self._doc_names,
            )
            if cached is None:
                self._doc_embeddings = await self._embed(
                    [self._document(e) for e in entries]
                )
                await asyncio.to_thread(
                    self._save_disk_cache,
                    signature=signature,
                    names=self._doc_names,
                    embeddings=self._doc_embeddings,
                )
            else:
                self._doc_embeddings = cached
            self._doc_signature = signature

    async def _keyword_fallback(
        self,
        query: str,
        k: int,
        *,
        reason: str,
        **kwargs: Any,
    ) -> list[SkillSearchResult]:
        logger.warning(f"[QWEN_RECALL] dense route failed; keyword fallback: {reason}")
        fallback = await self._catalog.search(query, k=k, **kwargs)
        return [
            SkillSearchResult(
                name=item.name,
                description=item.description,
                source=item.source,
                score=item.score,
                match_type=item.match_type,
                metadata={**item.metadata, "qwen_fallback": reason[:500]},
            )
            for item in fallback
        ]

    async def search(
        self, query: str, k: int = 10, **kwargs: Any
    ) -> list[SkillSearchResult]:
        if not query.strip() or not self.is_available():
            return []
        entries = self._entries()
        if not entries:
            return []
        try:
            await self._ensure_documents(entries)
            query_text = f"{self._query_instruction}{query}"
            query_embeddings = await self._embed([query_text])
        except Exception as exc:  # noqa: BLE001 - dense routing degrades to keyword recall
            return await self._keyword_fallback(
                query,
                k,
                reason=f"{type(exc).__name__}: {exc}",
                **kwargs,
            )

        try:
            if self.backend == "local":
                documents = self._doc_embeddings
                if not self._torch.is_tensor(documents):
                    documents = self._torch.as_tensor(
                        documents,
                        dtype=query_embeddings.dtype,
                        device=query_embeddings.device,
                    )
                scores = (query_embeddings[0] @ documents.T).tolist()
            else:
                query_vector = query_embeddings[0]
                scores = [
                    sum(a * b for a, b in zip(query_vector, doc, strict=True))
                    for doc in self._doc_embeddings
                ]

            entry_map = {entry.skill_name: entry for entry in entries}
            ranked = sorted(
                enumerate(scores),
                key=lambda item: (-float(item[1]), self._doc_names[item[0]]),
            )
            results: list[SkillSearchResult] = []
            for index, score in ranked[: max(1, int(k))]:
                name = self._doc_names[index]
                entry = entry_map[name]
                results.append(
                    SkillSearchResult(
                        name=name,
                        description=entry.description,
                        source="local",
                        # Embeddings are normalized, so their dot product is the
                        # one-step soft-Q/cosine score used by the reference router.
                        score=max(0.0, min(1.0, float(score))),
                        match_type="qwen_embedding",
                        metadata={"router_backend": self.backend},
                    )
                )
            return results
        except Exception as exc:  # noqa: BLE001 - dense routing degrades to keyword recall
            return await self._keyword_fallback(
                query,
                k,
                reason=f"{type(exc).__name__}: {exc}",
                **kwargs,
            )

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        stats.update(
            {
                "backend": self.backend,
                "model": self._model_path or self._api_model,
                "indexed_skills": len(self._doc_names),
            }
        )
        return stats
