uv run python -m core.evolve.main \
    --experiment read-write-optimize \
    --data hle_data/sampled_train100_test50_no_replace/train.jsonl \
    --start 0 --max-tasks 30 \
    --judge-model openai/o3-mini \
    --skip-task-exceptions \
    --optimize-attempts 3 \
    --optimize-unit-test-gate

uv run python -m core.evolve.main \
    --experiment read-write-optimize \
    --data gaia_data/data/split_by_level_60_40/train.jsonl \
    --start 0 --max-tasks 30 \
    --judge-model openai/o3-mini \
    --skip-task-exceptions \
    --optimize-attempts 3 \
    --optimize-unit-test-gate