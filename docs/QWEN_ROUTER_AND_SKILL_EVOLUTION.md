# Qwen Router and Guarded Skill Evolution

This implementation connects the paper's Read and Write phases to the production
runtime.

## Read: Qwen routing

Before planning, `MementoSAgent` calls `SkillGateway.route(goal)`. The local
router can run in three modes:

| Mode | Behaviour |
| --- | --- |
| `keyword` | Existing lightweight local keyword recall. |
| `qwen` | Qwen3 embedding recall only. |
| `hybrid` | Keyword and Qwen candidates combined by score-aware reciprocal rank fusion. |

The Qwen implementation matches the Memento-S reference router:

- skill documents use the checkpoint-aligned `Skill: <name>\nDescription: <description>` format;
- only the query receives the Qwen retrieval instruction;
- a local model uses last-token pooling followed by L2 normalisation;
- the similarity score is the one-step routing Q score;
- document embeddings are persisted under `.router-cache/` and invalidated by
  model/catalog signatures;
- local loading falls back through `flash_attention_2`, `sdpa`, and the default
  attention implementation;
- a base Qwen3-Embedding model or a behaviour-fine-tuned Memento-Qwen checkpoint
  can be supplied through the same interface.

The router supports either a local Transformers checkpoint or an
OpenAI-compatible embedding endpoint. Heavy local dependencies are optional and
loaded only on the first Qwen route:

```bash
pip install -e '.[qwen-router]'
```

Example user configuration (`~/memento_s/config.json`):

```json
{
  "skills": {
    "retrieval": {
      "router_backend": "hybrid",
      "qwen_tokenizer_path": "Qwen/Qwen3-Embedding-0.6B",
      "qwen_model_path": "/models/memento-qwen",
      "qwen_device": "auto",
      "qwen_timeout_sec": 30
    }
  }
}
```

For an embedding service, leave `qwen_model_path` empty and configure
`embedding_base_url`, `embedding_api_key`, and `embedding_model`. If Qwen is not
configured or fails, the router retains a keyword fallback (including `qwen`-only
mode). Placeholder keys such as `sk-xxxxxx` are treated as unconfigured so they
do not add network latency.

## Write: automatic attribution, update, tests, and rollback

When an execution is judged to require replanning, or the runtime stops after
repeated skill failures, the execution trace enters this guarded pipeline:

1. **Attribution** - the LLM must select exactly one mutable skill used in the
   trace and provide a confidence, reusable failure mode, and concrete evidence.
2. **Utility decision** - empirical utility is persisted as
   `success/(success+failure)`. With at least 3 samples, utility below `0.2`
   triggers a full same-name methodology rebuild; otherwise the engine applies a
   targeted repair. This mirrors `gaia_evolve.py` without deleting the live skill.
3. **Targeted update/discovery** - the rewriter returns complete file replacements scoped
   to that skill directory. Optional best-of-N attempts retain the highest-scoring
   candidate that passes every gate. Path traversal and symlink replacement are rejected.
4. **Isolated candidate** - edits are applied to a copied directory; the active
   skill has not changed yet.
5. **Automatic gates** - parse `SKILL.md`, compile Python sources, run any
   `test_*.py` tests, synthesize a new test request, execute the candidate through
   `SkillAgent`, and have the judge score the observable result.
6. **Deploy or rollback** - all gates passing causes an atomic directory swap and
   preserves the old tree as a backup. Any failure discards/archives the
   candidate while the original remains active. A deployment-time error restores
   the backup immediately.

Core built-in skills are protected by default. Configuration:

```json
{
  "skills": {
    "evolution": {
      "enabled": true,
      "protected_skills": [
        "filesystem",
        "skill-creator",
        "uv-pip-install",
        "web-search"
      ],
      "min_attribution_confidence": 0.5,
      "max_updates_per_step": 1,
      "candidate_attempts": 1,
      "utility_discovery_enabled": true,
      "utility_discovery_threshold": 0.2,
      "utility_min_samples": 3,
      "test_timeout_sec": 180,
      "synthetic_test_enabled": true,
      "keep_failed_candidate": true
    }
  }
}
```

Runtime data lives under the configured skills directory:

```text
.evolution/
├── utility.json                 # empirical success/failure utility
├── events.jsonl                 # attribution, gate, deployment audit log
├── backups/<skill>/<event>/     # pre-update snapshots
└── failed/<event>/              # rejected candidates when retention is enabled
```

After a successful low-utility rebuild, that skill starts a new utility generation
at the neutral `0.5` prior; the previous generation remains recorded in
`reset_from` for audit.

Restore the newest snapshot explicitly with:

```bash
memento skills rollback <skill-name>
```
