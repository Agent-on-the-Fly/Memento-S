# Core Architecture

This directory contains the runtime engine used by the CLI.

## Module map

- `config.py`: environment/config constants.
- `llm.py`: LLM transport and retry logic.
- `router.py`: step routing and semantic candidate selection.
- `utils/`: shared helpers (`json_utils`, `path_utils`, `logging_utils`).
- `skill_engine/`: planning and skill execution internals.

## skill_engine layout

- `skill_runner.py`: compatibility facade (stable import path).
- `planning.py`: plan generation/validation/normalization.
- `execution.py`: single-skill loop execution + continuation heuristics.
- `summarization.py`: long output summarization.
- `create_on_miss.py`: create-on-miss decision and orchestration.
- `skill_executor.py`: bridge op execution (filesystem/terminal/web/uv).
- `executor_utils.py`: shared executor parsing/canonicalization helpers.
- `skill_resolver.py`: local skill lookup + dynamic fetch/install.
- `skill_catalog.py`: catalog parsing, semantic indexing/ranking.
  - `catalog_jsonl.py`: shared JSONL catalog parsing/normalization helpers.
  - router retriever methods: `tfidf` / `bm25` / `qwen` / `memento_qwen`
  - embedding cache precompute path: `router_data/embeddings/`
  - optional async prewarm on first route: `SEMANTIC_ROUTER_EMBED_PREWARM=1`
- `skill_utils.py`: helper for call-skill bridge invocation.

## Design notes

- Keep CLI-facing imports stable via `skill_runner.py` exports.
- Prefer adding new logic in focused modules, not the facade.
- Avoid printing directly from core modules unless it is intentional UX output.
- Use `core.utils.logging_utils.log_event` for diagnostics and traceability.
