SKILL_DYNAMIC_FETCH_ENABLED=1 \
SKILL_DYNAMIC_FETCH_CATALOG_JSONL=./router_data/skills_catalog.jsonl \
uv run python -m evolve.main \
    --experiment read-write-optimize \
    --data hle_data/sampled_train100_test50_no_replace/train.jsonl \
    --local-skills-dir skills \
    --skill-extra-dir experiments/rwo/skill_extra \
    --tip-file experiments/rwo/TIP.md \
    --start 0 --max-tasks 20 \
    --run-dir experiments/rwo/runs \
    --judge-model openai/o3-mini \
    --skip-task-exceptions \
    --max-steps 50 \
    --optimize-attempts 3 \
    --optimize-unit-test-gate
